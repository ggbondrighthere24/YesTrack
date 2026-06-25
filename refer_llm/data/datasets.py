from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

# 确保可以从项目根目录导入兄弟包（例如 data）
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.refer_kitti_mot import ReferKittiMOT
from refer_llm.crop_utils import crop_with_margin


def build_refer_dataset(
        data_root: str,
        split: str = "train",
        train_ids_override: Optional[List[int]] = None,
        val_ids_override: Optional[List[int]] = None,
) -> ReferKittiMOT:
    return ReferKittiMOT(
        data_root=data_root,
        split=split,
        load_annotation=True,
        expression_sub_dir="expression",
        labels_with_ids_sub_dir="labels_with_ids/image_02",
        train_ids_override=train_ids_override,
        val_ids_override=val_ids_override,
    )


class QwenReferYesNoDataset(Dataset):
    def __init__(
            self,
            refer_dataset: ReferKittiMOT,
            image_size: int = 384,
            margin_ratio: float = 0.2,
            margin_px: Optional[int] = None,
            min_side: int = 8,
            max_text_len: int = 128,
            negative_downsample: float = 1.0,
            coord_mode: str = "xy",
            coord_decimals: int = 3,
            prompt_single_tpl: Optional[str] = None,
            oversample_seq: Optional[str] = None,
            oversample_factor: int = 1,
    ):
        super().__init__()
        self.ds = refer_dataset
        self.image_size = image_size
        self.margin_ratio = margin_ratio
        self.margin_px = margin_px
        self.min_side = min_side
        self.max_text_len = max_text_len
        self.negative_downsample = negative_downsample
        self.coord_mode = coord_mode  # "xy" or "xywh"
        self.coord_decimals = coord_decimals
        self.prompt_single_tpl = prompt_single_tpl
        self.oversample_seq = oversample_seq
        self.oversample_factor = max(1, int(oversample_factor or 1))
        if self.prompt_single_tpl is None or (isinstance(self.prompt_single_tpl, str) and len(self.prompt_single_tpl.strip()) == 0):
            raise RuntimeError("prompt_single_tpl 未提供或为空")

        # 预展开索引：每个样本 = (seq, frame_idx, ann_idx, sentence_idx)
        self.samples: List[Tuple[str, int, int, int]] = []
        for seq in self.ds.sequence_names:
            num_frames = self.ds.sequence_infos[seq]["length"]
            exprs = self.ds._load_expressions_for_sequence(seq)
            if len(exprs) == 0:
                raise RuntimeError(f"训练数据缺少文本表达: sequence={seq}")
            for frame_idx in range(num_frames):
                ann = self.ds.annotations[seq][frame_idx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                for sentence_idx, expr in enumerate(exprs):
                    label_map = expr.get("label", {})
                    ids = label_map.get(str(frame_idx), [])
                    try:
                        pos_ids = set(int(i) for i in ids)
                    except Exception:
                        pos_ids = set()
                    for ann_idx in range(M):
                        obj_id = int(ann["id"][ann_idx].item())
                        is_pos = obj_id in pos_ids
                        if not is_pos and self.negative_downsample < 1.0:
                            if torch.rand(1).item() > self.negative_downsample:
                                continue
                        self.samples.append((seq, frame_idx, ann_idx, sentence_idx))

        self.expr_cache: Dict[str, List[Dict[str, Any]]] = {}

        if len(self.samples) == 0:
            raise RuntimeError("训练数据构建失败：未生成任何样本（可能缺少图像或文本）")
        
        # 简单序列过采样（例如行人稀少的 0016）
        if self.oversample_seq and self.oversample_factor > 1:
            seq_tag = str(self.oversample_seq)
            to_dup = [s for s in self.samples if s[0] == seq_tag]
            if len(to_dup) > 0:
                extra = []
                for _ in range(self.oversample_factor - 1):
                    extra.extend(to_dup)
                self.samples.extend(extra)

    def __len__(self):
        return len(self.samples)

    def _get_expressions(self, seq: str) -> List[Dict[str, Any]]:
        if seq not in self.expr_cache:
            self.expr_cache[seq] = self.ds._load_expressions_for_sequence(seq)
        return self.expr_cache[seq]

    def __getitem__(self, idx: int):
        seq, frame_idx, ann_idx, sentence_idx = self.samples[idx]
        img_path = self.ds.image_paths[seq][frame_idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"训练阶段找不到图像或无法打开: {img_path}") from e
        W, H = image.size

        ann = self.ds.annotations[seq][frame_idx]
        x, y, w, h = ann["bbox"][ann_idx].tolist()

        patch = crop_with_margin(
            image=image,
            bbox_xywh=(x, y, w, h),
            margin_ratio=self.margin_ratio,
            margin_px=self.margin_px,
            min_side=self.min_side,
        )
        if patch is None:
            raise RuntimeError(
                f"crop_with_margin failed: seq={seq}, frame={frame_idx}, ann_idx={ann_idx}, bbox=({x},{y},{w},{h})"
            )

        if self.image_size is not None:
            patch = patch.resize((self.image_size, self.image_size))

        exprs = self._get_expressions(seq)
        if sentence_idx >= len(exprs):
            raise RuntimeError(f"训练阶段找不到对应的文本表达: sequence={seq}, sentence_idx={sentence_idx}")
        sentence = exprs[sentence_idx].get("sentence", "")
        if not isinstance(sentence, str) or len(sentence.strip()) == 0:
            raise RuntimeError(f"训练阶段文本为空: sequence={seq}, frame={frame_idx}, sentence_idx={sentence_idx}")
        label_map = exprs[sentence_idx].get("label", {})
        ids = label_map.get(str(frame_idx), [])
        try:
            pos_ids = set(int(i) for i in ids)
        except Exception:
            pos_ids = set()
        obj_id = int(ann["id"][ann_idx].item())
        y_label = 1 if obj_id in pos_ids else 0

        cx = x + 0.5 * w
        cy = y + 0.5 * h
        nx1 = max(0.0, min(1.0, cx / float(W)))
        ny1 = max(0.0, min(1.0, cy / float(H)))
        fmt = "{:." + str(max(0, int(self.coord_decimals))) + "f}"
        if self.coord_mode == "xywh":
            nw = max(0.0, min(1.0, w / float(W)))
            nh = max(0.0, min(1.0, h / float(H)))
            coord_str = f"{fmt.format(nx1)} {fmt.format(ny1)} {fmt.format(nw)} {fmt.format(nh)}"
        else:
            coord_str = f"{fmt.format(nx1)} {fmt.format(ny1)}"
        try:
            prompt = self.prompt_single_tpl.format(coord=coord_str, sentence=sentence)
        except Exception as e:
            raise RuntimeError(f"单帧 prompt 模板格式化失败: {e}")

        return {
            "image": patch,
            "text": prompt,
            "label": y_label,
        }


class QwenReferVideoYesNoDataset(Dataset):
    def __init__(
            self,
            refer_dataset: ReferKittiMOT,
            image_size: int = 384,
            margin_ratio: float = 0.1,
            margin_px: Optional[int] = None,
            min_side: int = 8,
            max_text_len: int = 128,
            negative_downsample: float = 1.0,
            video_n_frames: int = 4,
            prompt_video_tpl: Optional[str] = None,
            oversample_seq: Optional[str] = None,
            oversample_factor: int = 1,
            coord_mode: str = "xy",
            coord_decimals: int = 3,
    ):
        super().__init__()
        self.ds = refer_dataset
        self.image_size = image_size
        self.margin_ratio = margin_ratio
        self.margin_px = margin_px
        self.min_side = min_side
        self.max_text_len = max_text_len
        self.negative_downsample = negative_downsample
        self.video_n_frames = int(video_n_frames)
        self.prompt_video_tpl = prompt_video_tpl
        self.oversample_seq = oversample_seq
        self.oversample_factor = max(1, int(oversample_factor or 1))
        self.coord_mode = coord_mode  # "xy" or "xywh"
        self.coord_decimals = coord_decimals
        if self.prompt_video_tpl is None or (isinstance(self.prompt_video_tpl, str) and len(self.prompt_video_tpl.strip()) == 0):
            raise RuntimeError("prompt_video_tpl 未提供或为空")

        self.samples: List[Tuple[str, int, int, int]] = []
        for seq in self.ds.sequence_names:
            num_frames = self.ds.sequence_infos[seq]["length"]
            exprs = self.ds._load_expressions_for_sequence(seq)
            if len(exprs) == 0:
                raise RuntimeError(f"训练数据缺少文本表达: sequence={seq}")
            for frame_idx in range(num_frames):
                ann = self.ds.annotations[seq][frame_idx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                for sentence_idx, expr in enumerate(exprs):
                    label_map = expr.get("label", {})
                    ids = label_map.get(str(frame_idx), [])
                    try:
                        pos_ids = set(int(i) for i in ids)
                    except Exception:
                        pos_ids = set()
                    for ann_idx in range(M):
                        obj_id = int(ann["id"][ann_idx].item())
                        is_pos = obj_id in pos_ids
                        if not is_pos and self.negative_downsample < 1.0:
                            if torch.rand(1).item() > self.negative_downsample:
                                continue
                        self.samples.append((seq, frame_idx, ann_idx, sentence_idx))

        if len(self.samples) == 0:
            raise RuntimeError("训练数据构建失败：未生成任何样本（可能缺少图像或文本）")
        
        # 简单序列过采样（例如行人稀少的 0016）
        if self.oversample_seq and self.oversample_factor > 1:
            seq_tag = str(self.oversample_seq)
            to_dup = [s for s in self.samples if s[0] == seq_tag]
            if len(to_dup) > 0:
                extra = []
                for _ in range(self.oversample_factor - 1):
                    extra.extend(to_dup)
                self.samples.extend(extra)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, frame_idx, ann_idx, sentence_idx = self.samples[idx]

        img_path = self.ds.image_paths[seq][frame_idx]
        try:
            image_cur = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"训练阶段找不到图像或无法打开: {img_path}") from e

        ann = self.ds.annotations[seq][frame_idx]
        x, y, w, h = ann["bbox"][ann_idx].tolist()
        obj_id = int(ann["id"][ann_idx].item())

        images_seq: List[Image.Image] = []
        start_f = max(0, frame_idx - self.video_n_frames + 1)
        for fidx in range(start_f, frame_idx + 1):
            ann_f = self.ds.annotations[seq][fidx]
            ids_f = ann_f["id"].tolist()

            if obj_id not in ids_f:
                # 目标在该帧缺失，跳过此帧以保证训练不中断
                continue
            idx_f = ids_f.index(obj_id)

            x_f, y_f, w_f, h_f = ann_f["bbox"][idx_f].tolist()
            img_path_f = self.ds.image_paths[seq][fidx]

            try:
                image_f = Image.open(img_path_f).convert("RGB")
            except Exception:
                raise ValueError(f"Invalid image: {img_path_f}")

            patch_f = crop_with_margin(
                image=image_f,
                bbox_xywh=(x_f, y_f, w_f, h_f),
                margin_ratio=self.margin_ratio,
                margin_px=self.margin_px,
                min_side=self.min_side,
            )
            if patch_f is None:
                raise RuntimeError(
                    f"crop_with_margin failed in video clip: seq={seq}, frame={fidx}, ann_idx={idx_f}, bbox=({x_f},{y_f},{w_f},{h_f})"
                )
            if self.image_size is not None:
                patch_f = patch_f.resize((self.image_size, self.image_size))
            images_seq.append(patch_f)
        if len(images_seq) == 0:
            patch_cur = crop_with_margin(
                image=image_cur,
                bbox_xywh=(x, y, w, h),
                margin_ratio=self.margin_ratio,
                margin_px=self.margin_px,
                min_side=self.min_side,
            )
            if patch_cur is None:
                raise RuntimeError(
                    f"crop_with_margin failed for current frame: seq={seq}, frame={frame_idx}, ann_idx={ann_idx}, bbox=({x},{y},{w},{h})"
                )
            if self.image_size is not None:
                patch_cur = patch_cur.resize((self.image_size, self.image_size))
            images_seq = [patch_cur]

        W, H = image_cur.size
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        nx = max(0.0, min(1.0, cx / float(W)))
        ny = max(0.0, min(1.0, cy / float(H)))
        fmt = "{:." + str(max(0, int(self.coord_decimals))) + "f}"
        if self.coord_mode == "xywh":
            nw = max(0.0, min(1.0, w / float(W)))
            nh = max(0.0, min(1.0, h / float(H)))
            coord_str = f"{fmt.format(nx)} {fmt.format(ny)} {fmt.format(nw)} {fmt.format(nh)}"
        else:
            coord_str = f"{fmt.format(nx)} {fmt.format(ny)}"
        coords_str = f"<{coord_str}>"

        exprs = self.ds._load_expressions_for_sequence(seq)
        if sentence_idx >= len(exprs):
            raise RuntimeError(f"训练阶段找不到对应的文本表达: sequence={seq}, sentence_idx={sentence_idx}")
        sentence = exprs[sentence_idx].get("sentence", "")
        if not isinstance(sentence, str) or len(sentence.strip()) == 0:
            raise RuntimeError(f"训练阶段文本为空: sequence={seq}, frame={frame_idx}, sentence_idx={sentence_idx}")

        label_map = exprs[sentence_idx].get("label", {})
        ids = label_map.get(str(frame_idx), [])
        
        try:
            pos_ids = set(int(i) for i in ids)
        except Exception:
            raise ValueError(f"Invalid ids: {ids}")
            # pos_ids = set()
        y_label = 1 if obj_id in pos_ids else 0

        try:
            prompt = self.prompt_video_tpl.format(sentence=sentence, coord=coord_str, coords=coords_str)
        except Exception as e:
            raise RuntimeError(f"视频 prompt 模板格式化失败: {e}")

        return {
            "images": images_seq,
            "text": prompt,
            "label": y_label,
        }


class FilteredSingleYesNoDataset(Dataset):
    """
    从筛选器输出目录加载单帧样本：
    期望结构: filtered_root/single/<seq>/*.json，其中 meta 包含:
      - saved_image: 图像路径
      - prompt: 已格式化的 prompt
      - y_true: 0/1
    """
    def __init__(
        self,
        filtered_root: str,
        image_size: Optional[int] = 320,
    ):
        super().__init__()
        self.filtered_root = os.path.abspath(filtered_root)
        self.image_size = image_size
        self.items: List[Tuple[str, str, int]] = []  # (image_path, prompt, label)
        single_root = os.path.join(self.filtered_root, "single")
        if not os.path.isdir(single_root):
            raise RuntimeError(f"FilteredSingleYesNoDataset: 目录不存在: {single_root}")
        for seq in sorted(os.listdir(single_root)):
            seq_dir = os.path.join(single_root, seq)
            if not os.path.isdir(seq_dir):
                continue
            for fn in sorted(os.listdir(seq_dir)):
                if not fn.endswith(".json"):
                    # 允许同目录下存在 .jpg 等图像文件；仅对 .json 进行严格校验
                    continue
                meta_path = os.path.join(seq_dir, fn)
                import json as _json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                img_path = meta.get("saved_image") or meta.get("image")
                prompt = meta.get("prompt")
                if img_path is None or prompt is None:
                    raise RuntimeError(f"FilteredSingleYesNoDataset: 元数据缺少字段(saved_image/prompt): {meta_path}")
                try:
                    label = int(meta.get("y_true", 0))
                except Exception as e:
                    raise RuntimeError(f"FilteredSingleYesNoDataset: y_true 非法: {meta_path}") from e
                if not os.path.isabs(img_path):
                    img_path = os.path.join(os.path.dirname(meta_path), os.path.basename(img_path))
                if not os.path.isfile(img_path):
                    raise RuntimeError(f"FilteredSingleYesNoDataset: 图像不存在: {img_path} (meta: {meta_path})")
                # 轻量校验能否打开
                try:
                    _ = Image.open(img_path).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"FilteredSingleYesNoDataset: 图像无法打开: {img_path}") from e
                self.items.append((img_path, prompt, label))
        if len(self.items) == 0:
            raise RuntimeError(f"FilteredSingleYesNoDataset: 未找到任何样本于 {single_root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, prompt, label = self.items[idx]
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"过滤单帧样本无法打开图像: {img_path}") from e
        if self.image_size is not None:
            im = im.resize((self.image_size, self.image_size))
        return {
            "image": im,
            "text": prompt,
            "label": int(label),
        }


class FilteredVideoYesNoDataset(Dataset):
    """
    从筛选器输出目录加载视频样本：
    期望结构: filtered_root/video/<seq>/<case_dir>/meta.json，其中 meta 包含:
      - items: [{image: 路径, frame_idx: int, bbox_xywh: [x,y,w,h]}, ...]
      - prompt: 已格式化的 prompt
      - sequence, obj_id, sentence_idx, frames
    标签从 ReferKittiMOT 中恢复（以最后一帧 frame_idx 的 label_map 判定）。
    """
    def __init__(
        self,
        filtered_root: str,
        refer_dataset: ReferKittiMOT,
        image_size: Optional[int] = 320,
    ):
        super().__init__()
        self.filtered_root = os.path.abspath(filtered_root)
        self.ds = refer_dataset
        self.image_size = image_size
        self.items: List[Dict[str, Any]] = []
        video_root = os.path.join(self.filtered_root, "video")
        if not os.path.isdir(video_root):
            raise RuntimeError(f"FilteredVideoYesNoDataset: 目录不存在: {video_root}")
        for seq in sorted(os.listdir(video_root)):
            seq_dir = os.path.join(video_root, seq)
            if not os.path.isdir(seq_dir):
                raise RuntimeError(f"FilteredVideoYesNoDataset: 非目录项: {seq_dir}")
            for case_dir in sorted(os.listdir(seq_dir)):
                case_path = os.path.join(seq_dir, case_dir)
                if not os.path.isdir(case_path):
                    raise RuntimeError(f"FilteredVideoYesNoDataset: 非目录项: {case_path}")
                meta_path = os.path.join(case_path, "meta.json")
                if not os.path.isfile(meta_path):
                    raise RuntimeError(f"FilteredVideoYesNoDataset: 缺少 meta.json: {case_path}")
                import json as _json
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                prompt = meta.get("prompt")
                frames = meta.get("frames", [])
                items = meta.get("items", [])
                if prompt is None or not isinstance(items, list) or len(items) == 0:
                    raise RuntimeError(f"FilteredVideoYesNoDataset: 元数据缺少 prompt/items: {meta_path}")
                try:
                    obj_id = int(meta.get("obj_id"))
                    sentence_idx = int(meta.get("sentence_idx"))
                except Exception as e:
                    raise RuntimeError(f"FilteredVideoYesNoDataset: obj_id/sentence_idx 非法: {meta_path}") from e
                sequence = str(meta.get("sequence") or seq)
                # 计算标签：以最后一帧为准
                if len(frames) > 0:
                    try:
                        last_frame = int(frames[-1])
                    except Exception as e:
                        raise RuntimeError(f"FilteredVideoYesNoDataset: frames[-1] 非法: {meta_path}") from e
                else:
                    if "frame_idx" not in items[-1]:
                        raise RuntimeError(f"FilteredVideoYesNoDataset: 缺少 frames 且 items[-1].frame_idx 不可用: {meta_path}")
                    last_frame = int(items[-1].get("frame_idx"))
                exprs = self.ds._load_expressions_for_sequence(sequence)
                if sentence_idx >= len(exprs):
                    raise RuntimeError(f"FilteredVideoYesNoDataset: sentence_idx 越界: {meta_path}")
                label_map = exprs[sentence_idx].get("label", {})
                ids = label_map.get(str(last_frame), [])
                try:
                    pos_ids = set(int(i) for i in ids)
                except Exception as e:
                    raise RuntimeError(f"FilteredVideoYesNoDataset: label_map 非法: {meta_path}") from e
                y_label = 1 if obj_id in pos_ids else 0
                # 收集并校验图像路径
                imgs = []
                for it in items:
                    ip = it.get("image")
                    if not ip:
                        raise RuntimeError(f"FilteredVideoYesNoDataset: items 中缺少 image 字段: {meta_path}")
                    if not os.path.isabs(ip):
                        ip = os.path.join(case_path, os.path.basename(ip))
                    if not os.path.isfile(ip):
                        raise RuntimeError(f"FilteredVideoYesNoDataset: 图像不存在: {ip} (meta: {meta_path})")
                    # 轻量校验
                    try:
                        _ = Image.open(ip).convert("RGB")
                    except Exception as e:
                        raise RuntimeError(f"FilteredVideoYesNoDataset: 图像无法打开: {ip}") from e
                    imgs.append(ip)
                self.items.append({
                    "images": imgs,
                    "prompt": prompt,
                    "label": y_label,
                })
        if len(self.items) == 0:
            raise RuntimeError(f"FilteredVideoYesNoDataset: 未找到任何样本于 {video_root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        rec = self.items[idx]
        imgs_paths = rec["images"]
        prompt = rec["prompt"]
        label = int(rec["label"])
        images: List[Image.Image] = []
        for p in imgs_paths:
            # try:
            im = Image.open(p).convert("RGB")
            # except Exception:
            #     continue
            if self.image_size is not None:
                im = im.resize((self.image_size, self.image_size))
            images.append(im)
        if len(images) == 0:
            # 构造兜底
            # images = [Image.new("RGB", (self.image_size or 320, self.image_size or 320), (128, 128, 128))]
            raise RuntimeError(f"FilteredVideoYesNoDataset: 无法加载图像: {imgs_paths}")
        return {
            "images": images,
            "text": prompt,
            "label": label,
        }


