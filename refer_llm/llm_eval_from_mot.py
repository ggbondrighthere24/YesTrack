from __future__ import annotations

import os
import re
import argparse
import glob
import math
import time
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
from collections import deque

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging; logger = logging.getLogger(__name__)

# 全局幻觉统计计数器（按样本数计）
HALLUCINATION_TOTAL_SAMPLES = 0
HALLUCINATION_TOTAL_HITS = 0

# 项目内导入
import sys
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoProcessor
from refer_llm.llm_eval import build_model_and_processor_for_eval, get_yes_no_token_ids
from refer_llm.crop_utils import crop_with_margin, xywh_to_xyxy_with_margin
from refer_llm.llm_train import _save_batch_visualization  # 复用训练中的可视化实现（带自动换行）
from data.refer_kitti_mot import ReferKittiMOT


def setup_distributed():
    """初始化分布式环境（推理使用，多进程多卡划分任务）"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)


def _natural_sentence_from_dirname(name: str) -> str:
    # 目录名如 "black-cars-in-the-left" -> "black cars in the left"
    s = re.sub(r"[_\-]+", " ", name).strip()
    return s


def _normalize_sentence_for_compare(text: str) -> str:
    """规范化句子用于比对：小写、连字符转空格、仅保留字母数字与空格、压缩多空格。"""
    t = str(text or "").strip().lower()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _check_expressions_match_dataset(
    data_root: str,
    results_root: str,
) -> None:
    """检查每个序列下的 expression 目录与数据集 expressions 数量与内容一致；否则抛错。"""
    ds_all = ReferKittiMOT(
        data_root=data_root,
        split="all",  # 包含全部序列
        load_annotation=False,
        expression_sub_dir="expression",
        labels_with_ids_sub_dir="labels_with_ids/image_02",
    )
    seqs = sorted([d for d in os.listdir(results_root) if os.path.isdir(os.path.join(results_root, d))])
    if len(seqs) == 0:
        raise RuntimeError(f"results_root 下未发现任何序列: {results_root}")
    for seq in seqs:
        seq_dir = os.path.join(results_root, seq)
        expr_dirs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
        expr_from_results = [_natural_sentence_from_dirname(d) for d in expr_dirs]
        set_results = {_normalize_sentence_for_compare(s) for s in expr_from_results}
        # 加载数据集表达式
        exprs_ds = ds_all._load_expressions_for_sequence(seq)
        if len(exprs_ds) == 0:
            raise RuntimeError(f"数据集中序列 {seq} 未找到 expressions（expression/{seq} 为空）")
        expr_from_ds = [str(e.get("sentence", "") or "").strip() for e in exprs_ds]
        set_ds = {_normalize_sentence_for_compare(s) for s in expr_from_ds}
        if set_results != set_ds:
            missing_in_results = sorted(list(set_ds - set_results))[:5]
            extra_in_results = sorted(list(set_results - set_ds))[:5]
            raise RuntimeError(
                f"表达式不一致: 序列 {seq} | "
                f"数据集={len(set_ds)} 项, 结果目录={len(set_results)} 项 | "
                f"结果缺失(示例)={missing_in_results} | 多余(示例)={extra_in_results}"
            )


def _dirname_from_sentence(sentence: str) -> str:
    """
    将自然语言句子转为目录名（尽量与训练/结果目录风格一致）：
      - 小写
      - 空白与大多数标点 -> '-'
      - **保留逗号 ','**（为了与 GT/TrackEval 的表达式命名保持一致，例如 "in-silver,-cars-..."）
      - 压缩重复 '-'
      - 去掉首尾 '-'
    """
    s = str(sentence or "").strip().lower()
    # Keep commas to match GT naming; normalize everything else to '-'
    s = re.sub(r"[^a-z0-9,]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s if len(s) > 0 else "expr"


def _iter_seq_files_from_bytetrack_dir(
    bytetrack_dir: str,
    file_glob: str = "*.txt",
) -> List[Tuple[str, str]]:
    """
    ByteTrack 输出模式：
      bytetrack_dir 下包含若干文件：{seq}.txt
    返回 [(seq, mot_path), ...]
    """
    bytetrack_dir = os.path.abspath(bytetrack_dir)
    if not os.path.isdir(bytetrack_dir):
        raise FileNotFoundError(f"bytetrack_dir 不存在或不是目录: {bytetrack_dir}")
    pattern = os.path.join(bytetrack_dir, file_glob)
    files = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
    if len(files) == 0:
        raise RuntimeError(f"bytetrack_dir 下未找到文件: {pattern}")

    out: List[Tuple[str, str]] = []
    for pth in files:
        base = os.path.basename(pth)
        name, ext = os.path.splitext(base)
        if ext.lower() != ".txt":
            continue
        seq = str(name).strip()
        if len(seq) == 0:
            continue
        out.append((seq, pth))
    if len(out) == 0:
        raise RuntimeError(f"bytetrack_dir 下未找到合法的 {file_glob} 结果文件: {bytetrack_dir}")
    return out


def _load_mot_file(
    mot_path: str,
    *,
    frame_offset: int = 0,
) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    """
    读取 MOT 检测文件，返回 {frame -> [(id,x,y,w,h,score), ...]}
    支持逗号或空格分隔。
    """
    frame_to_dets: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
    off = int(frame_offset)
    with open(mot_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            parts = [p for p in re.split(r"[, \t]+", line) if len(p) > 0]
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
                score = float(parts[6]) if len(parts) > 6 else 1.0
            except Exception:
                continue
            if off != 0:
                frame = int(frame) + off
                if frame < 0:
                    raise ValueError(
                        f"frame_offset 导致出现负帧号：mot_path={mot_path}, raw_frame={parts[0]}, frame_offset={off}, shifted_frame={frame}"
                    )
            frame_to_dets.setdefault(frame, []).append((tid, x, y, w, h, score))
    return frame_to_dets


def _max_frame_in_mot_output(out_path: str) -> Optional[int]:
    """读取 predict_with_conf.txt，返回其中出现的最大 frame（第一列）；读不到则返回 None。"""
    try:
        mx: Optional[int] = None
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = [p for p in re.split(r"[, \t]+", s) if p]
                if not parts:
                    continue
                try:
                    fr = int(float(parts[0]))
                except Exception:
                    continue
                mx = fr if mx is None else max(mx, fr)
        return mx
    except Exception:
        return None


def _forward_probs_for_patches(
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    patches: List[Optional[Image.Image]],
    prompts: List[str],
    yes_id: int,
    no_id: int,
    *,
    context: str = "",
) -> List[float]:
    """对已裁剪好的 patch 进行前向，返回 Yes 概率（与输入顺序对齐）。"""
    assert len(patches) == len(prompts)
    # Ensure left padding for batch inference so the last token is the prediction point
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    batch_messages = []
    kept_indices: List[int] = []
    for idx, (img, pr_) in enumerate(zip(patches, prompts)):
        if img is None:
            # 之前这里会导致最终 out[idx]=0.0；现在改为遇到就报错，避免“很多 0”
            raise RuntimeError(
                f"Invalid patch (None) at index={idx}. This usually means crop/resize failed. {context}".strip()
            )
        batch_messages.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": pr_}]}
        ])
        kept_indices.append(idx)
    if len(batch_messages) == 0:
        raise RuntimeError(f"No valid inputs for model forward (batch_messages is empty). {context}".strip())

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]
    two_logits = last_logits[:, [yes_id, no_id]]
    probs = torch.softmax(two_logits, dim=-1)[:, 0].detach().cpu().tolist()
    out = [0.0 for _ in prompts]
    for k, idx in enumerate(kept_indices):
        out[idx] = probs[k]
    return out


def _prepare_patches_for_frame(
    img_path: str,
    dets_xywh: List[Tuple[float, float, float, float]],
    image_size: Optional[int],
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    preprocess_workers: int = 0,
    *,
    context: str = "",
) -> Tuple[List[Optional[Image.Image]], int, int]:
    """
    评估加速：同一帧只读一次图像，然后按 bbox 裁剪 patch（可选多线程）。
    返回 patches（与 dets_xywh 对齐）以及 (W,H)。
    """
    try:
        with Image.open(img_path) as im0:
            im = im0.convert("RGB")
    except Exception:
        # 之前会导致该帧所有样本 confidence=0；现在改为直接报错
        raise FileNotFoundError(f"Failed to open image: {img_path} {context}".strip())
    W, H = im.size
    # 转 numpy 以确保线程并发裁剪安全（避免共享 PIL 对象的线程安全问题）
    np_img = np.array(im)  # H x W x 3, uint8

    def _crop_one(b):
        x, y, w, h = b
        # 基础合法性检查：避免 NaN/Inf 或负尺寸导致 silent 0
        for v, nm in [(x, "x"), (y, "y"), (w, "w"), (h, "h")]:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                raise ValueError(f"Invalid bbox value {nm}={v} for img={img_path} bbox={b} {context}".strip())
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bbox with non-positive size for img={img_path} bbox={b} {context}".strip())
        x1, y1, x2, y2 = xywh_to_xyxy_with_margin(
            bbox_xywh=(x, y, w, h),
            image_size=(W, H),
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
        )
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid crop region (x2<=x1 or y2<=y1) for img={img_path} "
                f"bbox={b} -> xyxy=({x1},{y1},{x2},{y2}) {context}".strip()
            )
        patch_np = np_img[y1:y2, x1:x2]
        if patch_np is None or patch_np.size == 0:
            raise ValueError(
                f"Empty crop array for img={img_path} bbox={b} -> xyxy=({x1},{y1},{x2},{y2}) {context}".strip()
            )
        try:
            patch = Image.fromarray(patch_np)
        except Exception:
            raise RuntimeError(f"Failed to create PIL Image from crop for img={img_path} bbox={b} {context}".strip())
        if image_size is not None:
            patch = patch.resize((int(image_size), int(image_size)))
        return patch

    n_workers = int(preprocess_workers) if preprocess_workers is not None else 0
    if n_workers > 0 and len(dets_xywh) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            patches = list(ex.map(_crop_one, dets_xywh))
    else:
        patches = [_crop_one(b) for b in dets_xywh]
    return patches, int(W), int(H)


def _load_images_np_cache(
    image_paths: List[str],
    preprocess_workers: int = 0,
) -> Dict[str, Tuple[np.ndarray, int, int]]:
    """批量加载若干图像到 numpy 缓存：path -> (np_img, W, H)。失败的路径将被跳过。"""
    uniq = sorted({str(p) for p in image_paths if p is not None and len(str(p)) > 0})
    cache: Dict[str, Tuple[np.ndarray, int, int]] = {}

    def _load_one(pth: str):
        try:
            with Image.open(pth) as im0:
                im = im0.convert("RGB")
            W, H = im.size
            np_img = np.array(im)  # uint8
            return pth, (np_img, int(W), int(H))
        except Exception:
            return pth, None

    n_workers = int(preprocess_workers) if preprocess_workers is not None else 0
    if n_workers > 0 and len(uniq) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for pth, out in ex.map(_load_one, uniq):
                if out is not None:
                    cache[pth] = out
    else:
        for pth in uniq:
            _, out = _load_one(pth)
            if out is not None:
                cache[pth] = out
    return cache


def _prepare_video_patches_for_samples(
    image_paths_per_sample: List[List[str]],
    dets_per_sample: List[List[Tuple[float, float, float, float]]],
    image_size: Optional[int],
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    preprocess_workers: int = 0,
) -> List[List[Image.Image]]:
    """
    视频/二阶段提速：将所有样本涉及到的帧图像去重加载（每帧只解码一次），并并行裁剪/resize。
    返回每个样本的 frames_imgs（与 paths_one/boxes_one 的顺序一致，失败帧会被跳过）。
    """
    all_paths: List[str] = []
    for paths_one in image_paths_per_sample:
        if paths_one:
            all_paths.extend(paths_one)
    cache = _load_images_np_cache(all_paths, preprocess_workers=preprocess_workers)

    # 生成所有 crop 任务
    tasks: List[Tuple[int, int, str, Tuple[float, float, float, float]]] = []
    for si, (paths_one, boxes_one) in enumerate(zip(image_paths_per_sample, dets_per_sample)):
        for fi, (pth, box) in enumerate(zip(paths_one, boxes_one)):
            tasks.append((si, fi, pth, box))

    # 输出容器：先用 None 占位，再填充 Image
    out: List[List[Optional[Image.Image]]] = []
    for paths_one in image_paths_per_sample:
        out.append([None for _ in range(len(paths_one))])

    def _crop_task(t):
        si, fi, pth, (x, y, w, h) = t
        entry = cache.get(pth)
        if entry is None:
            return si, fi, None
        np_img, W, H = entry
        x1, y1, x2, y2 = xywh_to_xyxy_with_margin(
            bbox_xywh=(x, y, w, h),
            image_size=(W, H),
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
        )
        if x2 <= x1 or y2 <= y1:
            return si, fi, None
        patch_np = np_img[y1:y2, x1:x2]
        if patch_np is None or patch_np.size == 0:
            return si, fi, None
        try:
            patch = Image.fromarray(patch_np)
        except Exception:
            return si, fi, None
        if image_size is not None:
            patch = patch.resize((int(image_size), int(image_size)))
        return si, fi, patch

    n_workers = int(preprocess_workers) if preprocess_workers is not None else 0
    if n_workers > 0 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for si, fi, patch in ex.map(_crop_task, tasks):
                if patch is not None and si < len(out) and fi < len(out[si]):
                    out[si][fi] = patch
    else:
        for t in tasks:
            si, fi, patch = _crop_task(t)
            if patch is not None and si < len(out) and fi < len(out[si]):
                out[si][fi] = patch

    # 每个样本压缩掉 None（模拟原逻辑：读图失败的帧直接跳过）
    final: List[List[Image.Image]] = []
    for frames in out:
        final.append([im for im in frames if im is not None])
    return final


@torch.no_grad()
def _forward_probs_for_detections(
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    image_paths: List[str],
    dets: List[Tuple[float, float, float, float]],  # 每个 (x,y,w,h) 对应一张 image_paths 的图
    prompts: List[str],
    yes_id: int,
    no_id: int,
    image_size: Optional[int],
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
) -> List[float]:
    """
    针对若干单帧样本前向，返回 Yes 概率。
    每个样本一张图（已裁剪）。
    """
    assert len(image_paths) == len(dets) == len(prompts)
    images: List[Image.Image] = []
    for img_path, (x, y, w, h) in zip(image_paths, dets):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            raise FileNotFoundError(f"Failed to open image: {img_path}")
        patch = crop_with_margin(
            image=img,
            bbox_xywh=(x, y, w, h),
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
        )
        if patch is None:
            patch = img.crop((int(x), int(y), int(x + w), int(y + h)))
        if image_size is not None:
            patch = patch.resize((int(image_size), int(image_size)))
        images.append(patch)

    # Ensure left padding
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    batch_messages = []
    kept_indices: List[int] = []
    for idx, (img, pr_) in enumerate(zip(images, prompts)):
        if img is None:
            raise RuntimeError(f"Invalid patch (None) at index={idx} path={image_paths[idx]}")
        batch_messages.append([
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": pr_}]}
        ])
        kept_indices.append(idx)

    if len(batch_messages) == 0:
        raise RuntimeError("No valid inputs for model forward (batch_messages is empty).")

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]
    two_logits = last_logits[:, [yes_id, no_id]]
    probs = torch.softmax(two_logits, dim=-1)[:, 0].detach().cpu().tolist()

    # 回填到原顺序
    out = [0.0 for _ in image_paths]
    for k, idx in enumerate(kept_indices):
        out[idx] = probs[k]
    return out


@torch.no_grad()
def _forward_probs_for_detections_video(
    model: torch.nn.Module,
    processor: AutoProcessor,
    device: torch.device,
    image_paths_per_sample: List[List[str]],  # 每个样本多帧的图像路径
    dets_per_sample: List[List[Tuple[float, float, float, float]]],  # 对应每帧的 bbox
    prompts: List[str],  # 每个样本一个 prompt（视频模板）
    yes_id: int,
    no_id: int,
    image_size: Optional[int],
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    preprocess_workers: int = 0,
    *,
    context: str = "",
) -> List[float]:
    """
    针对若干视频样本前向，返回 Yes 概率。每个样本可包含多帧图像。
    """
    # 预处理提速：去重加载帧 + 并行 crop/resize
    frames_imgs_per_sample = _prepare_video_patches_for_samples(
        image_paths_per_sample=image_paths_per_sample,
        dets_per_sample=dets_per_sample,
        image_size=image_size,
        margin_ratio=margin_ratio,
        margin_px=margin_px,
        min_side=min_side,
        preprocess_workers=int(preprocess_workers or 0),
    )

    # Ensure left padding
    if processor.tokenizer.padding_side != "left":
        processor.tokenizer.padding_side = "left"

    batch_messages = []
    kept_indices: List[int] = []
    for idx, (frames_imgs, pr_) in enumerate(zip(frames_imgs_per_sample, prompts)):
        if len(frames_imgs) == 0:
            raise RuntimeError(
                f"Video sample has 0 valid frames after load/crop. sample_idx={idx} "
                f"(this previously produced confidence=0 for that sample). {context}".strip()
            )
        batch_messages.append([
            {"role": "user", "content": [*([{"type": "image", "image": im} for im in frames_imgs]), {"type": "text", "text": pr_}]}
        ])
        kept_indices.append(idx)

    if len(batch_messages) == 0:
        raise RuntimeError(f"No valid video inputs for model forward (batch_messages is empty). {context}".strip())

    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(device)
    outputs = model(**inputs)
    last_logits = outputs.logits[:, -1, :]

    # ── 幻觉检测统计 ──────────────────────────────────────────────────
    global HALLUCINATION_TOTAL_SAMPLES, HALLUCINATION_TOTAL_HITS
    if last_logits.dim() == 2 and last_logits.size(0) > 0:
        argmax_ids = last_logits.argmax(dim=-1)  # [B]
        is_yes = argmax_ids == yes_id
        is_no = argmax_ids == no_id
        hallucination_mask = ~(is_yes | is_no)   # 既不是 Yes 也不是 No

        batch_size = int(argmax_ids.size(0))
        n_hallucination = int(hallucination_mask.sum().item())

        HALLUCINATION_TOTAL_SAMPLES += batch_size
        HALLUCINATION_TOTAL_HITS += n_hallucination

        if n_hallucination > 0:
            hallucination_indices = hallucination_mask.nonzero(as_tuple=True)[0].tolist()
            hallucination_token_ids = argmax_ids[hallucination_mask].tolist()
            # 解码出实际 token 方便排查
            hallucination_tokens = processor.tokenizer.convert_ids_to_tokens(hallucination_token_ids)
            logger.warning(
                f"[Hallucination] {n_hallucination}/{batch_size} samples' top-1 logit is "
                f"neither Yes({yes_id}) nor No({no_id}). "
                f"batch_indices={hallucination_indices}, "
                f"token_ids={hallucination_token_ids}, "
                f"tokens={hallucination_tokens}. "
                f"{context}".strip()
            )
    # ─────────────────────────────────────────────────────────────────

    two_logits = last_logits[:, [yes_id, no_id]]
    probs = torch.softmax(two_logits, dim=-1)[:, 0].detach().cpu().tolist()

    out = [0.0 for _ in prompts]
    for k, idx in enumerate(kept_indices):
        out[idx] = probs[k]
    return out


def main():
    p = argparse.ArgumentParser()
    # 模型/权重
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--lora_path", type=str, default='v1bestweight/lora_step_10000', help="LoRA 目录（训练保存的适配器路径）")
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--fp16", action="store_true")
    # 数据
    p.add_argument("--data_root", type=str, default="/data/sq_2023/refer_kitti")
    # p.add_argument("--dataset_version", type=str, default="dancetrack", choices=["v1", "v2", "dancetrack"])
    p.add_argument("--results_root", type=str, default="track_result_old/v1best_track_result", help="结果目录（如 track_result/results_epoch58）")
    p.add_argument("--mot_filename", type=str, default="predict.txt", help="MOT 文件名；若不存在则跳过该 expression（绝不回退 gt）")
    p.add_argument("--mot_frame_start_one", type=int, default=1, help="MOT 文件帧号是否从 1 开始（1 表示从1开始，0 表示从0开始）")
    p.add_argument(
        "--image_frame_start_one",
        type=int,
        default=0,
        choices=[0, 1],
        help="图像文件帧号是否从 1 开始（1 表示首帧是 000001.png；0 表示首帧是 000000.png）。不做自动回退，请显式设置。",
    )
    # MOT 输入来源：
    # - results_root: 读取 results_root/seq/expr_dir/predict.txt
    # - bytetrack: 读取 bytetrack_dir/{seq}.txt，并对该 seq 的所有表达式（expression/{seq}/*.json）都进行评估
    p.add_argument("--mot_input_type", type=str, default="results_root", choices=["results_root", "bytetrack"])
    p.add_argument("--bytetrack_dir", type=str, default="track_result/referdance", help="ByteTrack 输出目录（包含 {seq}.txt，frame 从 0 开始）")
    p.add_argument("--bytetrack_file_glob", type=str, default="*.txt", help="ByteTrack 文件 glob（默认 *.txt）")
    p.add_argument("--frame_offset", type=int, default=0, help="对输入 MOT/ByteTrack 文件的帧号做整体偏移：frame := frame + frame_offset；例如 -1 表示帧号减1，对两种模式都生效")
    p.add_argument("--skip_expression_check", action="store_true", help="跳过表达式与数据集的一致性检查（仅 results_root 有意义）")
    # 预处理加速：同帧只读一次图 + 裁剪/resize 并行
    p.add_argument("--preprocess_workers", type=int, default=4, help="评估时图像裁剪/resize 的线程数（0=单线程）")
    # 推理设置
    p.add_argument("--enable_video_mode", action="store_true", help="切换为全视频推理：同一 id 回溯多帧直接推理，不走单帧主分支")
    p.add_argument("--video_n_frames", type=int, default=4)
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--margin_ratio", type=float, default=0.2)
    p.add_argument("--margin_px", type=int, default=None)
    p.add_argument("--min_side", type=int, default=8)
    # Prompt
    p.add_argument("--prompt_single_tpl", type=str, default="The normalized position of the car or person in the picture is <{coord}>.Determine whether this description matches this image: {sentence}. Answer Yes or No.", help="单帧模板，必须提供，支持 {coord}, {sentence}")
    p.add_argument("--prompt_video_tpl", type=str, default="This is a short video clip of a car or person at <{coord}> across frames. The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. Answer Yes or No.", help="视频模板，启用视频/二阶段时必须提供，支持 {sentence}, {coord}")
    # 二阶段精炼（与 llm_eval.py 对齐）
    p.add_argument("--re_refer_thresh", type=float, default=0.8, help="二阶段精炼阈值上限（触发区间 [re_refer_lower, re_refer_thresh)）")
    p.add_argument("--re_refer_lower", type=float, default=0.2, help="二阶段精炼触发下限（需显式提供）")
    p.add_argument("--disable_refine", action="store_true", help="不使用二阶段精炼（忽略 re_refer_* 与视频模板）")
    # 在线稳定性加分（最近 N 帧都 ≥ 阈值，则当前帧 +boost；仅使用历史帧）
    p.add_argument("--stability_enable", action="store_true", help="开启在线稳定性加分")
    p.add_argument("--stability_window", type=int, default=6, help="稳定性窗口（最近 N 帧）")
    p.add_argument("--stability_thresh", type=float, default=0.4, help="稳定性阈值（最近 N 帧均需 ≥ 该值）")
    p.add_argument("--stability_boost", type=float, default=0.3, help="满足稳定性条件时的固定加分（裁剪到 ≤ 1.0）")
    # 跳帧推理：每隔 N 帧做一次“全量”推理；中间帧复用上一次 confidence，新 id 当帧补推理
    p.add_argument("--infer_every_n_frames", type=int, default=4, help="每隔 N 帧做一次完整推理；中间帧复用同 id 上次置信度，新 id 立即补推理（1=每帧推理）")
    # 断点续跑（任务粒度）：若某个 seq/expr 的输出文件已完整存在，则跳过该任务；否则重跑该任务
    p.add_argument("--resume", action="store_true", help="启用断点续跑（按任务跳过已完成输出）")
    # 输出
    p.add_argument("--output_root", type=str, default='track_result/trpandtcp64', help="输出根目录；若为空，将在 lora_path 的上级目录创建 eval_with_mot_result")
    # 可视化
    p.add_argument("--vis_every", type=int, default=500, help="每处理多少帧保存一次可视化拼图（按当前进程计数）")
    p.add_argument("--max_vis_items", type=int, default=8, help="每个拼图的最大样本数")
    args = p.parse_args()

    # 分布式
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model, processor, _ = build_model_and_processor_for_eval(
        model_name=args.model_name,
        use_4bit=bool(args.use_4bit),
        bf16=not bool(args.fp16),
        lora_path=args.lora_path,
    )
    model.to(device)
    yes_id, no_id = get_yes_no_token_ids(processor)

    # 输出目录
    if args.output_root is None or len(str(args.output_root).strip()) == 0:
        args.output_root = os.path.join(os.path.dirname(os.path.abspath(args.lora_path)), "eval_with_mot_result")
    if is_main_process():
        os.makedirs(args.output_root, exist_ok=True)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    # 任务收集 + 表达式检查
    tasks: List[Tuple[str, str, str]] = []  # (seq, expr_dirname, mot_file)
    if str(args.mot_input_type) == "results_root":
        if not bool(getattr(args, "skip_expression_check", False)):
            # 先一致性检查（所有进程都执行，遇错直接抛出）
            _check_expressions_match_dataset(data_root=args.data_root, results_root=args.results_root)
            if world_size > 1 and dist.is_initialized():
                dist.barrier()
        # 从 results_root/seq/expr_dir 读取
        for seq in sorted([d for d in os.listdir(args.results_root) if os.path.isdir(os.path.join(args.results_root, d))]):
            seq_dir = os.path.join(args.results_root, seq)
            for expr_dirname in sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]):
                mot_file = os.path.join(seq_dir, expr_dirname, args.mot_filename)
                tasks.append((seq, expr_dirname, mot_file))
    else:
        if args.bytetrack_dir is None or len(str(args.bytetrack_dir).strip()) == 0:
            raise ValueError("mot_input_type=bytetrack 时必须提供 --bytetrack_dir")
        # ByteTrack: 对每个 seq 的所有 expression 都评估同一份 {seq}.txt
        ds_all = ReferKittiMOT(
            data_root=args.data_root,
            split="all",
            load_annotation=False,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
        )
        seq_files = _iter_seq_files_from_bytetrack_dir(
            bytetrack_dir=str(args.bytetrack_dir),
            file_glob=str(getattr(args, "bytetrack_file_glob", "*.txt")),
        )
        for seq, mot_path in seq_files:
            exprs_ds = ds_all._load_expressions_for_sequence(seq)
            if len(exprs_ds) == 0:
                raise RuntimeError(f"数据集中序列 {seq} 未找到 expressions（expression/{seq} 为空）")
            # 生成稳定且唯一的 expr_dirname
            used: Dict[str, int] = {}
            for e in exprs_ds:
                sent = str(e.get("sentence", "") or "").strip()
                dirname = _dirname_from_sentence(sent)
                if dirname in used:
                    used[dirname] += 1
                    dirname = f"{dirname}__{used[dirname]}"
                else:
                    used[dirname] = 0
                tasks.append((seq, dirname, mot_path))

    # -----------------------------
    # 整体推理计时（wall-clock）
    # 统计从开始处理任务到全部结束的总耗时。
    # 分布式下以 max(rank_elapsed) 作为整体耗时（由最慢 rank 决定）。
    # -----------------------------
    timing_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timing_start_wall = time.time()
    timing_start_perf = time.perf_counter()
    attempted_tasks = 0
    ok_tasks = 0
    err_tasks = 0

    # 二阶段参数与开关（与 llm_eval.py 一致的校验）
    enable_refine = (not bool(getattr(args, "disable_refine", False)))
    lower_bound = None
    if enable_refine:
        if getattr(args, "re_refer_lower", None) is None:
            enable_refine = False
        else:
            lower_bound = float(args.re_refer_lower)
            if not (float(args.re_refer_thresh) > float(lower_bound)):
                raise ValueError(f"re_refer_thresh({args.re_refer_thresh}) 必须大于 re_refer_lower({args.re_refer_lower})")
            if args.prompt_video_tpl is None or len(str(args.prompt_video_tpl).strip()) == 0:
                raise ValueError("二阶段精炼需要提供 prompt_video_tpl（无回退）")

    # 任务分配
    is_bytetrack = (str(args.mot_input_type) == "bytetrack")
    for t_idx, (seq, expr_dirname, mot_file) in enumerate(tasks):
        if world_size > 1 and (t_idx % world_size != rank):
            continue
        attempted_tasks += 1
        if not os.path.isfile(mot_file):
            if str(args.mot_input_type) == "results_root":
                raise FileNotFoundError(f"{seq}/{expr_dirname} 缺少 {args.mot_filename}，请检查输入；严禁回退 gt.txt")
            else:
                raise FileNotFoundError(f"{seq}/{expr_dirname} 缺少 tracker 文件: {mot_file}")

        try:
            frame_to_dets = _load_mot_file(mot_file, frame_offset=int(getattr(args, "frame_offset", 0) or 0))
            # 图像路径模式
            img_seq_dir = os.path.join(args.data_root, "KITTI", "training", "image_02", seq)
            # img_seq_dir = os.path.join(args.data_root, "DanceTrack", "training", "image_02", seq)
            if not os.path.isdir(img_seq_dir):
                # 兼容 data_root 布局为 .../KITTI/training/...
                img_seq_dir_alt = os.path.join(args.data_root, "training", "image_02", seq)
                img_seq_dir = img_seq_dir_alt if os.path.isdir(img_seq_dir_alt) else img_seq_dir

            # 输出文件
            out_seq_expr_dir = os.path.join(args.output_root, seq, expr_dirname)
            os.makedirs(out_seq_expr_dir, exist_ok=True)
            out_path = os.path.join(out_seq_expr_dir, "predict_with_conf.txt")
            # resume：若输出文件存在且已覆盖到最后一帧，则认为该任务已完成，直接跳过
            if bool(getattr(args, "resume", False)) and os.path.isfile(out_path) and len(frame_to_dets) > 0:
                last_out = _max_frame_in_mot_output(out_path)
                if last_out is not None:
                    last_in = max(frame_to_dets.keys())
                    expected_last_out = (int(last_in) + 1) if is_bytetrack else int(last_in)
                    if int(last_out) >= int(expected_last_out):
                        if is_main_process():
                            print(f"[SKIP] {seq}/{expr_dirname} (resume): {out_path}")
                        ok_tasks += 1
                        continue
            vis_dir = os.path.join(out_seq_expr_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)

            # 构建 id -> [(frame, (x,y,w,h))]
            id_to_traj: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
            for f, dets in frame_to_dets.items():
                for tid, x, y, w, h, _ in dets:
                    id_to_traj.setdefault(tid, []).append((f, (x, y, w, h)))
            for tid in id_to_traj:
                id_to_traj[tid].sort(key=lambda z: z[0])

            # 逐帧推理
            lines_out: List[str] = []
            vis_counter = 0
            # 在线稳定性缓存：tid -> 最近若干帧的置信度（仅历史，不含未来）
            stability_cache: Dict[int, deque] = {}
            # 跳帧推理缓存：tid -> 最近一次推理得到的置信度
            last_conf_by_tid: Dict[int, float] = {}
            infer_stride = max(1, int(getattr(args, "infer_every_n_frames", 1) or 1))
            frame_counter = 0  # 当前任务内处理帧计数（从 0 开始）
            for frame in sorted(frame_to_dets.keys()):
                dets_one = frame_to_dets[frame]
                if len(dets_one) == 0:
                    continue
                do_full_infer = (infer_stride <= 1) or (frame_counter % infer_stride == 0)
                # 将 tracker/MOT 帧号转换为“图像帧索引”（再根据 image_frame_start_one 映射到实际文件名）
                # - bytetrack: 约定 frame 从 0 开始
                # - results_root: 由 mot_frame_start_one 决定是否需要 -1
                base_img_idx = frame if is_bytetrack else (frame - 1 if int(args.mot_frame_start_one) == 1 else frame)
                if int(getattr(args, "image_frame_start_one", 0)) == 1:
                    base_img_idx = int(base_img_idx) + 1
                fidx_img = int(base_img_idx)
                cur_img_path = os.path.join(img_seq_dir, f"{fidx_img:06d}.png")

                sentence = _natural_sentence_from_dirname(expr_dirname)

                if not args.enable_video_mode:
                    # 单帧：支持跳帧推理（中间帧复用已有 tid；新 tid 当帧补推理）
                    did_infer_any = False
                    vis_boxes: List[Tuple[float, float, float, float]] = []
                    vis_prompts: List[str] = []
                    refine_candidate_indices: List[int] = []

                    if do_full_infer:
                        did_infer_any = True
                        # 单帧：每个检测单独一次（可以批起来）
                        dets_boxes = [(x, y, w, h) for (tid, x, y, w, h, _) in dets_one]
                        # 同帧只读一次图并裁剪 patch（可选多线程）
                        patches, W, H = _prepare_patches_for_frame(
                            img_path=cur_img_path,
                            dets_xywh=dets_boxes,
                            image_size=args.image_size,
                            margin_ratio=args.margin_ratio,
                            margin_px=args.margin_px,
                            min_side=args.min_side,
                            preprocess_workers=int(getattr(args, "preprocess_workers", 0) or 0),
                            context=f"seq={seq} expr={expr_dirname} frame={frame}",
                        )
                        # 归一化坐标用于 prompt（复用 W,H）
                        prompts = []
                        for (_, x, y, w, h, _) in dets_one:
                            cx = x + 0.5 * w; cy = y + 0.5 * h
                            nx = max(0.0, min(1.0, cx / float(W)))
                            ny = max(0.0, min(1.0, cy / float(H)))
                            coord_str = f"{nx:.3f} {ny:.3f}"
                            pr = args.prompt_single_tpl.format(coord=coord_str, sentence=sentence)
                            prompts.append(pr)
                        probs = _forward_probs_for_patches(
                            model=model,
                            processor=processor,
                            device=device,
                            patches=patches,
                            prompts=prompts,
                            yes_id=yes_id,
                            no_id=no_id,
                            context=f"seq={seq} expr={expr_dirname} frame={frame} img={cur_img_path}",
                        )
                        # 更新缓存
                        for (tid, _, _, _, _, _), p0 in zip(dets_one, probs):
                            last_conf_by_tid[int(tid)] = float(p0)
                        vis_boxes = dets_boxes
                        vis_prompts = prompts
                        refine_candidate_indices = list(range(len(dets_one)))
                    else:
                        # 中间帧：先复用已有 tid，并找出新 tid
                        probs = []
                        new_indices: List[int] = []
                        for i, (tid, _, _, _, _, _) in enumerate(dets_one):
                            tid_i = int(tid)
                            if tid_i in last_conf_by_tid:
                                probs.append(float(last_conf_by_tid[tid_i]))
                            else:
                                probs.append(0.0)
                                new_indices.append(i)

                        # 仅对新 tid 做推理
                        if len(new_indices) > 0:
                            did_infer_any = True
                            dets_boxes_new = [(dets_one[i][1], dets_one[i][2], dets_one[i][3], dets_one[i][4]) for i in new_indices]
                            patches, W, H = _prepare_patches_for_frame(
                                img_path=cur_img_path,
                                dets_xywh=dets_boxes_new,
                                image_size=args.image_size,
                                margin_ratio=args.margin_ratio,
                                margin_px=args.margin_px,
                                min_side=args.min_side,
                                preprocess_workers=int(getattr(args, "preprocess_workers", 0) or 0),
                                context=f"seq={seq} expr={expr_dirname} frame={frame} (new_ids_only)",
                            )
                            prompts_new: List[str] = []
                            for i in new_indices:
                                _, x, y, w, h, _ = dets_one[i]
                                cx = x + 0.5 * w; cy = y + 0.5 * h
                                nx = max(0.0, min(1.0, cx / float(W)))
                                ny = max(0.0, min(1.0, cy / float(H)))
                                coord_str = f"{nx:.3f} {ny:.3f}"
                                pr = args.prompt_single_tpl.format(coord=coord_str, sentence=sentence)
                                prompts_new.append(pr)
                            probs_new = _forward_probs_for_patches(
                                model=model,
                                processor=processor,
                                device=device,
                                patches=patches,
                                prompts=prompts_new,
                                yes_id=yes_id,
                                no_id=no_id,
                                context=f"seq={seq} expr={expr_dirname} frame={frame} img={cur_img_path} (new_ids_only)",
                            )
                            for local_k, det_idx in enumerate(new_indices):
                                p0 = float(probs_new[local_k]) if local_k < len(probs_new) else 0.0
                                probs[det_idx] = p0
                                last_conf_by_tid[int(dets_one[det_idx][0])] = p0
                            vis_boxes = dets_boxes_new
                            vis_prompts = prompts_new
                        refine_candidate_indices = new_indices
                    # 二阶段精炼：对 [lower, upper) 的样本使用多帧视频再判
                    if enable_refine and (probs is not None and len(probs) > 0) and did_infer_any:
                        upper = float(args.re_refer_thresh)
                        lower = float(lower_bound)
                        refine_indices = [i for i in refine_candidate_indices if (probs[i] >= lower and probs[i] < upper)]
                        if len(refine_indices) > 0:
                            img_paths_per_sample: List[List[str]] = []
                            dets_per_sample: List[List[Tuple[float, float, float, float]]] = []
                            prompts_video: List[str] = []
                            for i in refine_indices:
                                tid_i, x_i, y_i, w_i, h_i, _ = dets_one[i]
                                traj = id_to_traj.get(tid_i, [])
                                # 找到当前帧在轨迹中的索引
                                idx_in_traj = None
                                for k, (f_tr, _) in enumerate(traj):
                                    if f_tr == frame:
                                        idx_in_traj = k; break
                                paths_one: List[str] = []
                                boxes_one: List[Tuple[float, float, float, float]] = []
                                if idx_in_traj is not None:
                                    start_k = max(0, idx_in_traj - int(args.video_n_frames) + 1)
                                    for kk in range(start_k, idx_in_traj + 1):
                                        f_tr, (xk, yk, wk, hk) = traj[kk]
                                        # bytetrack: f_tr 本身从 0 开始，直接对应 000000.png
                                        # results_root: 仍按 mot_frame_start_one 控制
                                        fidx_img_k = f_tr if is_bytetrack else (f_tr - 1 if int(args.mot_frame_start_one) == 1 else f_tr)
                                        if int(getattr(args, "image_frame_start_one", 0)) == 1:
                                            fidx_img_k = int(fidx_img_k) + 1
                                        img_path_k = os.path.join(img_seq_dir, f"{fidx_img_k:06d}.png")
                                        paths_one.append(img_path_k)
                                        boxes_one.append((xk, yk, wk, hk))
                                if len(paths_one) == 0:
                                    paths_one = [cur_img_path]
                                    boxes_one = [(x_i, y_i, w_i, h_i)]
                                img_paths_per_sample.append(paths_one)
                                dets_per_sample.append(boxes_one)
                                # 当前帧坐标生成视频 prompt
                                cx_i = x_i + 0.5 * w_i; cy_i = y_i + 0.5 * h_i
                                nx_i = max(0.0, min(1.0, cx_i / float(W)))
                                ny_i = max(0.0, min(1.0, cy_i / float(H)))
                                coord_str_i = f"{nx_i:.3f} {ny_i:.3f}"
                                pr_v = args.prompt_video_tpl.format(coord=coord_str_i, sentence=sentence)
                                prompts_video.append(pr_v)
                            # 前向视频 refine
                            probs_refined = _forward_probs_for_detections_video(
                                model=model, processor=processor, device=device,
                                image_paths_per_sample=img_paths_per_sample,
                                dets_per_sample=dets_per_sample,
                                prompts=prompts_video, yes_id=yes_id, no_id=no_id,
                                image_size=args.image_size, margin_ratio=args.margin_ratio, margin_px=args.margin_px, min_side=args.min_side,
                                preprocess_workers=int(getattr(args, "preprocess_workers", 0) or 0),
                                context=f"seq={seq} expr={expr_dirname} frame={frame} (refine)",
                            )
                            # 回填更新
                            for local_idx, det_idx in enumerate(refine_indices):
                                p_new = probs_refined[local_idx] if local_idx < len(probs_refined) else probs[det_idx]
                                # 以前用 0 作为失败哨兵；现在严格模式下异常会直接抛出，不再需要哨兵逻辑
                                probs[det_idx] = float(p_new)
                                last_conf_by_tid[int(dets_one[det_idx][0])] = float(probs[det_idx])
                    # 在线稳定性加分（仅依赖历史）
                    if bool(getattr(args, "stability_enable", False)) and len(dets_one) > 0 and len(probs) == len(dets_one):
                        win = max(1, int(getattr(args, "stability_window", 3)))
                        thresh = float(getattr(args, "stability_thresh", 0.8))
                        boost = float(getattr(args, "stability_boost", 0.1))
                        for i, ((tid, x, y, w, h, _), p) in enumerate(zip(dets_one, probs)):
                            dq = stability_cache.get(tid)
                            if dq is None:
                                dq = deque(maxlen=win)
                                stability_cache[tid] = dq
                            if len(dq) >= win and all(v >= thresh for v in dq):
                                p = min(1.0, float(p) + boost)
                                probs[i] = p
                            dq.append(float(probs[i]))
                    frame_out = (int(frame) + 1) if is_bytetrack else int(frame)
                    for (tid, x, y, w, h, _), p in zip(dets_one, probs):
                        # 更新缓存：确保跳帧时复用的是“最终输出”的 confidence（含 refine/stability）
                        last_conf_by_tid[int(tid)] = float(p)
                        # 输出 frame：results_root 通常为 1-based；bytetrack 输入为 0-based，这里统一写成 1-based
                        lines_out.append(f"{frame_out},{tid},{int(x)},{int(y)},{int(w)},{int(h)},{p:.6f}")
                    # 可视化（按照频率）
                    vis_counter += 1
                    if did_infer_any and int(args.vis_every) > 0 and (vis_counter % int(args.vis_every) == 0):
                        try:
                            dbg_imgs = []
                            dbg_prs = []
                            for (x, y, w, h), pr in list(zip(vis_boxes, vis_prompts))[: int(args.max_vis_items)]:
                                with Image.open(cur_img_path).convert("RGB") as _im_:
                                    patch = crop_with_margin(
                                        image=_im_,
                                        bbox_xywh=(x, y, w, h),
                                        margin_ratio=args.margin_ratio,
                                        margin_px=args.margin_px,
                                        min_side=args.min_side,
                                    )
                                    if patch is None:
                                        patch = _im_.crop((int(x), int(y), int(x + w), int(y + h)))
                                    if args.image_size is not None:
                                        patch = patch.resize((int(args.image_size), int(args.image_size)))
                                    dbg_imgs.append(patch.copy())
                                dbg_prs.append(pr)
                            _save_batch_visualization(dbg_imgs, dbg_prs, vis_dir, global_step=frame, max_items=int(args.max_vis_items))
                        except Exception:
                            pass
                else:
                    # 视频：支持跳帧推理（中间帧复用已有 tid；新 tid 当帧补推理）
                    did_infer_any = False
                    img_paths_per_sample: List[List[str]] = []
                    dets_per_sample: List[List[Tuple[float, float, float, float]]] = []
                    prompts: List[str] = []

                    if do_full_infer:
                        infer_indices = list(range(len(dets_one)))
                        probs = [0.0 for _ in dets_one]
                    else:
                        infer_indices: List[int] = []
                        probs = []
                        for i, (tid, _, _, _, _, _) in enumerate(dets_one):
                            tid_i = int(tid)
                            if tid_i in last_conf_by_tid:
                                probs.append(float(last_conf_by_tid[tid_i]))
                            else:
                                probs.append(0.0)
                                infer_indices.append(i)

                    if len(infer_indices) > 0:
                        did_infer_any = True
                        # 获取当前图尺寸（用于坐标归一化；仅在需要生成 prompt 时才读图）
                        try:
                            with Image.open(cur_img_path) as _im_:
                                W, H = _im_.size
                        except Exception:
                            W, H = 1242, 375

                        for det_idx in infer_indices:
                            tid, x, y, w, h, _ = dets_one[det_idx]
                            traj = id_to_traj.get(tid, [])
                            # 找到当前帧在轨迹中的索引
                            idx_in_traj = None
                            for k, (f_tr, _) in enumerate(traj):
                                if f_tr == frame:
                                    idx_in_traj = k; break
                            # 回溯取最多 N 帧
                            paths_one: List[str] = []
                            boxes_one: List[Tuple[float, float, float, float]] = []
                            if idx_in_traj is not None:
                                start_k = max(0, idx_in_traj - int(args.video_n_frames) + 1)
                                for kk in range(start_k, idx_in_traj + 1):
                                    f_tr, (xk, yk, wk, hk) = traj[kk]
                                    # bytetrack: f_tr 本身从 0 开始，直接对应 000000.png
                                    # results_root: 仍按 mot_frame_start_one 控制
                                    fidx_img_k = f_tr if is_bytetrack else (f_tr - 1 if int(args.mot_frame_start_one) == 1 else f_tr)
                                    if int(getattr(args, "image_frame_start_one", 0)) == 1:
                                        fidx_img_k = int(fidx_img_k) + 1
                                    img_path_k = os.path.join(img_seq_dir, f"{fidx_img_k:06d}.png")
                                    paths_one.append(img_path_k)
                                    boxes_one.append((xk, yk, wk, hk))
                            if len(paths_one) == 0:
                                paths_one = [cur_img_path]
                                boxes_one = [(x, y, w, h)]
                            img_paths_per_sample.append(paths_one)
                            dets_per_sample.append(boxes_one)
                            # 使用当前帧坐标生成 prompt
                            cx = x + 0.5 * w; cy = y + 0.5 * h
                            nx = max(0.0, min(1.0, cx / float(W)))
                            ny = max(0.0, min(1.0, cy / float(H)))
                            coord_str = f"{nx:.3f} {ny:.3f}"
                            pr = args.prompt_video_tpl.format(coord=coord_str, sentence=sentence)
                            prompts.append(pr)

                        probs_infer = _forward_probs_for_detections_video(
                            model=model, processor=processor, device=device,
                            image_paths_per_sample=img_paths_per_sample,
                            dets_per_sample=dets_per_sample,
                            prompts=prompts, yes_id=yes_id, no_id=no_id,
                            image_size=args.image_size, margin_ratio=args.margin_ratio, margin_px=args.margin_px, min_side=args.min_side,
                            preprocess_workers=int(getattr(args, "preprocess_workers", 0) or 0),
                            context=f"seq={seq} expr={expr_dirname} frame={frame} (video_mode{' full' if do_full_infer else ' new_ids_only'})",
                        )
                        for local_k, det_idx in enumerate(infer_indices):
                            p0 = float(probs_infer[local_k]) if local_k < len(probs_infer) else 0.0
                            probs[det_idx] = p0
                            last_conf_by_tid[int(dets_one[det_idx][0])] = p0

                    # 在线稳定性加分（仅依赖历史）
                    if bool(getattr(args, "stability_enable", False)) and len(dets_one) > 0 and len(probs) == len(dets_one):
                        win = max(1, int(getattr(args, "stability_window", 3)))
                        thresh = float(getattr(args, "stability_thresh", 0.8))
                        boost = float(getattr(args, "stability_boost", 0.1))
                        for i, ((tid, x, y, w, h, _), p) in enumerate(zip(dets_one, probs)):
                            dq = stability_cache.get(tid)
                            if dq is None:
                                dq = deque(maxlen=win)
                                stability_cache[tid] = dq
                            if len(dq) >= win and all(v >= thresh for v in dq):
                                p = min(1.0, float(p) + boost)
                                probs[i] = p
                            dq.append(float(probs[i]))

                    frame_out = (int(frame) + 1) if is_bytetrack else int(frame)
                    for (tid, x, y, w, h, _), p in zip(dets_one, probs):
                        # 更新缓存：确保跳帧时复用的是“最终输出”的 confidence（含 stability）
                        last_conf_by_tid[int(tid)] = float(p)
                        # 输出 frame：results_root 通常为 1-based；bytetrack 输入为 0-based，这里统一写成 1-based
                        lines_out.append(f"{frame_out},{tid},{int(x)},{int(y)},{int(w)},{int(h)},{p:.6f}")

                    # 可视化（视频：使用推理过的样本的第一帧裁剪预览）
                    vis_counter += 1
                    if did_infer_any and int(args.vis_every) > 0 and (vis_counter % int(args.vis_every) == 0):
                        try:
                            dbg_imgs = []
                            dbg_prs = []
                            for paths_one, boxes_one, pr in list(zip(img_paths_per_sample, dets_per_sample, prompts))[: int(args.max_vis_items)]:
                                if len(paths_one) == 0:
                                    continue
                                first_path = paths_one[0]
                                x0, y0, w0, h0 = boxes_one[0]
                                with Image.open(first_path).convert("RGB") as _im_:
                                    patch = crop_with_margin(
                                        image=_im_,
                                        bbox_xywh=(x0, y0, w0, h0),
                                        margin_ratio=args.margin_ratio,
                                        margin_px=args.margin_px,
                                        min_side=args.min_side,
                                    )
                                    if patch is None:
                                        patch = _im_.crop((int(x0), int(y0), int(x0 + w0), int(y0 + h0)))
                                    if args.image_size is not None:
                                        patch = patch.resize((int(args.image_size), int(args.image_size)))
                                    dbg_imgs.append(patch.copy())
                                dbg_prs.append(pr)
                            _save_batch_visualization(dbg_imgs, dbg_prs, vis_dir, global_step=frame, max_items=int(args.max_vis_items))
                        except Exception:
                            pass

                frame_counter += 1

            # 写出
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines_out))
            if is_main_process():
                print(f"[OK] {seq}/{expr_dirname} -> {out_path}")
            ok_tasks += 1
        except Exception as e:
            print(f"[ERR] {seq}/{expr_dirname} failed: {e}")
            err_tasks += 1

    # 等待所有 rank 完成，再汇总“整体推理时间”
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    elapsed_rank = float(time.perf_counter() - timing_start_perf)
    timing_end_wall = time.time()

    elapsed_max = elapsed_rank
    elapsed_min = elapsed_rank
    elapsed_avg = elapsed_rank
    attempted_sum = attempted_tasks
    ok_sum = ok_tasks
    err_sum = err_tasks

    # 幻觉统计全局计数（先使用本 rank 的局部值）
    global HALLUCINATION_TOTAL_SAMPLES, HALLUCINATION_TOTAL_HITS
    hallu_total = int(HALLUCINATION_TOTAL_SAMPLES)
    hallu_hits = int(HALLUCINATION_TOTAL_HITS)

    if world_size > 1 and dist.is_initialized():
        # 用 all_reduce 在各 rank 间聚合时间与任务数
        t_max = torch.tensor([elapsed_rank], device=device, dtype=torch.float64)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
        elapsed_max = float(t_max.item())

        t_min = torch.tensor([elapsed_rank], device=device, dtype=torch.float64)
        dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
        elapsed_min = float(t_min.item())

        t_sum = torch.tensor([elapsed_rank], device=device, dtype=torch.float64)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        elapsed_avg = float((t_sum / float(world_size)).item())

        c_attempted = torch.tensor([attempted_tasks], device=device, dtype=torch.int64)
        c_ok = torch.tensor([ok_tasks], device=device, dtype=torch.int64)
        c_err = torch.tensor([err_tasks], device=device, dtype=torch.int64)
        dist.all_reduce(c_attempted, op=dist.ReduceOp.SUM)
        dist.all_reduce(c_ok, op=dist.ReduceOp.SUM)
        dist.all_reduce(c_err, op=dist.ReduceOp.SUM)
        attempted_sum = int(c_attempted.item())
        ok_sum = int(c_ok.item())
        err_sum = int(c_err.item())

        # 聚合幻觉统计
        c_hallu_total = torch.tensor([hallu_total], device=device, dtype=torch.int64)
        c_hallu_hits = torch.tensor([hallu_hits], device=device, dtype=torch.int64)
        dist.all_reduce(c_hallu_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(c_hallu_hits, op=dist.ReduceOp.SUM)
        hallu_total = int(c_hallu_total.item())
        hallu_hits = int(c_hallu_hits.item())

    if is_main_process():
        timing = {
            "run_id": timing_run_id,
            "start_time": datetime.datetime.fromtimestamp(timing_start_wall).isoformat(timespec="seconds"),
            "end_time": datetime.datetime.fromtimestamp(timing_end_wall).isoformat(timespec="seconds"),
            "elapsed_seconds_max": elapsed_max,
            "elapsed_seconds_min": elapsed_min,
            "elapsed_seconds_avg": elapsed_avg,
            "world_size": int(world_size),
            "tasks_total": int(len(tasks)),
            "attempted_tasks_sum": int(attempted_sum),
            "ok_tasks_sum": int(ok_sum),
            "err_tasks_sum": int(err_sum),
            "notes": "elapsed_seconds_max 是整体推理时间（分布式下由最慢 rank 决定）。",
        }
        out_json = os.path.join(args.output_root, "overall_infer_time.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(timing, f, ensure_ascii=False, indent=2)
        print(f"[TIME] Saved overall inference time: {out_json}")

        # 写出幻觉统计结果
        hallu_rate = float(hallu_hits) / float(hallu_total) if hallu_total > 0 else 0.0
        hallu_stats = {
            "total_samples": int(hallu_total),
            "hallucination_samples": int(hallu_hits),
            "hallucination_rate": hallu_rate,
        }
        hallu_json = os.path.join(args.output_root, "hallucination_stats.json")
        with open(hallu_json, "w", encoding="utf-8") as f:
            json.dump(hallu_stats, f, ensure_ascii=False, indent=2)
        print(f"[HALLU] Saved hallucination stats: {hallu_json}")

    cleanup_distributed()

if __name__ == "__main__":
    main()

