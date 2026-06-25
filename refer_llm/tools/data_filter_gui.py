from __future__ import annotations

import os
import sys
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

# 确保可从项目根目录导入
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.refer_kitti_mot import ReferKittiMOT
from refer_llm.crop_utils import crop_with_margin


DEFAULT_PROMPT_SINGLE = (
    "The normalized position of the car or person in the picture is <{coord}>."
    "Determine whether this description matches this image: {sentence}. Answer Yes or No."
)
DEFAULT_PROMPT_VIDEO = (
    "This is a short video clip of a car or person at <{coord}> across frames. "
    "The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. "
    "Answer Yes or No."
)


@dataclass
class SingleSample:
    sequence: str
    frame_idx: int
    obj_id: int
    bbox_xywh: Tuple[int, int, int, int]
    sentence_idx: int
    sentence: str
    y_true: int
    coord_str: str
    image_path: str
    crop_image: Image.Image


@dataclass
class VideoSample:
    sequence: str
    sentence_idx: int
    sentence: str
    obj_id: int
    frames: List[int]
    bboxes_xywh: List[Tuple[int, int, int, int]]
    coord_str: str  # 使用当前帧的中心点归一化坐标
    image_paths: List[str]
    crops: List[Image.Image]


class DataEnumerator:
    """
    基于 ReferKittiMOT 的采样器。
    - 单帧模式：按表达式(sentence)维度遍历，随机采样每个表达式下的不同行和目标，生成 SingleSample。
    - 视频模式：在当前帧对象基础上，回溯获取至多 N 帧生成 VideoSample。
    """
    def __init__(
        self,
        ds: ReferKittiMOT,
        prompt_single_tpl: str,
        prompt_video_tpl: str,
        image_size: int = 320,
        margin_ratio: float = 0.1,
        margin_px: Optional[int] = None,
        min_side: int = 8,
        coord_mode: str = "xy",
        coord_decimals: int = 3,
        video_n_frames: int = 4,
        seed: int = 42,
    ):
        self.ds = ds
        self.prompt_single_tpl = prompt_single_tpl
        self.prompt_video_tpl = prompt_video_tpl
        self.image_size = int(image_size) if image_size is not None else None
        self.margin_ratio = margin_ratio
        self.margin_px = margin_px
        self.min_side = min_side
        self.coord_mode = coord_mode
        self.coord_decimals = max(0, int(coord_decimals))
        self.video_n_frames = max(1, int(video_n_frames))
        self.rng = random.Random(int(seed))

        # 预构建 (sequence, sentence_idx) 列表，并随机打乱
        self.seq_to_exprs: Dict[str, List[Dict[str, Any]]] = {}
        self.seq_expr_index: List[Tuple[str, int]] = []
        for seq in self.ds.sequence_names:
            exprs = self.ds._load_expressions_for_sequence(seq)
            self.seq_to_exprs[seq] = exprs
            for sidx in range(len(exprs)):
                self.seq_expr_index.append((seq, sidx))
        self.rng.shuffle(self.seq_expr_index)
        self.cursor = 0

        # 每个序列预构建 id -> [(frame, bbox)] 轨迹，便于视频模式快速取帧
        self.seq_id_to_traj: Dict[str, Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]]] = {}
        for seq in self.ds.sequence_names:
            traj: Dict[int, List[Tuple[int, Tuple[int, int, int, int]]]] = {}
            num_frames = self.ds.sequence_infos[seq]["length"]
            for fidx in range(num_frames):
                ann = self.ds.annotations[seq][fidx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                for aidx in range(M):
                    x, y, w, h = ann["bbox"][aidx].tolist()
                    oid = int(ann["id"][aidx].item())
                    traj.setdefault(oid, []).append((fidx, (int(x), int(y), int(w), int(h))))
            for oid in traj:
                traj[oid].sort(key=lambda z: z[0])
            self.seq_id_to_traj[seq] = traj

    def set_seed(self, seed: int):
        self.rng = random.Random(int(seed))
        self.cursor = 0
        self.rng.shuffle(self.seq_expr_index)

    def _fmt_coord(self, vals: List[float]) -> str:
        fmt = "{:." + str(self.coord_decimals) + "f}"
        return " ".join(fmt.format(v) for v in vals)

    def _compute_prompt_coord_str(self, W: int, H: int, bbox: Tuple[int, int, int, int]) -> str:
        x, y, w, h = bbox
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        nx = max(0.0, min(1.0, cx / float(W)))
        ny = max(0.0, min(1.0, cy / float(H)))
        if self.coord_mode == "xywh":
            nw = max(0.0, min(1.0, w / float(W)))
            nh = max(0.0, min(1.0, h / float(H)))
            return self._fmt_coord([nx, ny, nw, nh])
        return self._fmt_coord([nx, ny])

    def _make_crop(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        patch = crop_with_margin(
            image=image,
            bbox_xywh=bbox,
            margin_ratio=self.margin_ratio,
            margin_px=self.margin_px,
            min_side=self.min_side,
        )
        if patch is None:
            x, y, w, h = bbox
            patch = image.crop((int(x), int(y), int(x + w), int(y + h)))
        if self.image_size is not None:
            patch = patch.resize((self.image_size, self.image_size))
        return patch

    def next_single_page(self, page_size: int = 8) -> List[SingleSample]:
        """返回最多 page_size 条单帧样本。按表达式维度推进游标，保证随机但可复现。"""
        out: List[SingleSample] = []
        guard = 0
        total_slots = len(self.seq_expr_index)
        while len(out) < page_size and guard < total_slots:
            if self.cursor >= len(self.seq_expr_index):
                break
            seq, sidx = self.seq_expr_index[self.cursor]
            self.cursor += 1
            guard += 1

            exprs = self.seq_to_exprs.get(seq, [])
            if sidx < 0 or sidx >= len(exprs):
                continue
            expr = exprs[sidx]
            sentence = str(expr.get("sentence", "") or "")
            label_map = expr.get("label", {})

            num_frames = self.ds.sequence_infos[seq]["length"]
            frames = list(range(num_frames))
            self.rng.shuffle(frames)

            for fidx in frames:
                ann = self.ds.annotations[seq][fidx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                ids_frame = label_map.get(str(fidx), [])
                try:
                    pos_ids = set(int(i) for i in ids_frame)
                except Exception:
                    pos_ids = set()
                # 随机对象顺序
                obj_indices = list(range(M))
                self.rng.shuffle(obj_indices)

                img_path = self.ds.image_paths[seq][fidx]
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                W, H = image.size

                for aidx in obj_indices:
                    x, y, w, h = ann["bbox"][aidx].tolist()
                    oid = int(ann["id"][aidx].item())
                    y_true = 1 if oid in pos_ids else 0
                    bbox_xywh = (int(x), int(y), int(w), int(h))
                    crop = self._make_crop(image, bbox_xywh)
                    coord_str = self._compute_prompt_coord_str(W, H, bbox_xywh)
                    # 组装 SingleSample（prompt 在GUI里显示，不参与采样）
                    out.append(SingleSample(
                        sequence=seq,
                        frame_idx=fidx,
                        obj_id=oid,
                        bbox_xywh=bbox_xywh,
                        sentence_idx=sidx,
                        sentence=sentence,
                        y_true=y_true,
                        coord_str=coord_str,
                        image_path=img_path,
                        crop_image=crop,
                    ))
                    if len(out) >= page_size:
                        break
                if len(out) >= page_size:
                    break
        return out

    def next_video_page(self, page_size: int = 2) -> List[VideoSample]:
        """
        返回最多 page_size 条视频样本。
        每条样本使用表达式上下文、当前帧对象，向前回溯至多 N-1 帧生成多帧裁剪。
        """
        out: List[VideoSample] = []
        guard = 0
        total_slots = len(self.seq_expr_index)
        while len(out) < page_size and guard < total_slots:
            if self.cursor >= len(self.seq_expr_index):
                break
            seq, sidx = self.seq_expr_index[self.cursor]
            self.cursor += 1
            guard += 1

            exprs = self.seq_to_exprs.get(seq, [])
            if sidx < 0 or sidx >= len(exprs):
                continue
            expr = exprs[sidx]
            sentence = str(expr.get("sentence", "") or "")
            label_map = expr.get("label", {})

            num_frames = self.ds.sequence_infos[seq]["length"]
            frames = list(range(num_frames))
            self.rng.shuffle(frames)

            traj_map = self.seq_id_to_traj.get(seq, {})
            for fidx in frames:
                ann = self.ds.annotations[seq][fidx]
                M = ann["bbox"].shape[0]
                if M == 0:
                    continue
                obj_indices = list(range(M))
                self.rng.shuffle(obj_indices)

                img_path_cur = self.ds.image_paths[seq][fidx]
                try:
                    image_cur = Image.open(img_path_cur).convert("RGB")
                except Exception:
                    continue
                W, H = image_cur.size

                for aidx in obj_indices:
                    x, y, w, h = ann["bbox"][aidx].tolist()
                    oid = int(ann["id"][aidx].item())
                    # 构造多帧轨迹（向前回溯）
                    traj = traj_map.get(oid, [])
                    # 找当前帧在轨迹中的位置
                    cur_pos = None
                    for k, (f_tr, _) in enumerate(traj):
                        if f_tr == fidx:
                            cur_pos = k
                            break
                    if cur_pos is None:
                        continue
                    start_k = max(0, cur_pos - (self.video_n_frames - 1))
                    frames_sel = [traj[k][0] for k in range(start_k, cur_pos + 1)]
                    boxes_sel = [traj[k][1] for k in range(start_k, cur_pos + 1)]
                    img_paths_sel = [self.ds.image_paths[seq][fi] for fi in frames_sel]
                    crops: List[Image.Image] = []
                    for ipath, bbox in zip(img_paths_sel, boxes_sel):
                        try:
                            im = Image.open(ipath).convert("RGB")
                        except Exception:
                            continue
                        crops.append(self._make_crop(im, bbox))
                    if len(crops) == 0:
                        continue
                    coord_str = self._compute_prompt_coord_str(W, H, (int(x), int(y), int(w), int(h)))
                    out.append(VideoSample(
                        sequence=seq,
                        sentence_idx=sidx,
                        sentence=sentence,
                        obj_id=oid,
                        frames=frames_sel,
                        bboxes_xywh=boxes_sel,
                        coord_str=coord_str,
                        image_paths=img_paths_sel,
                        crops=crops,
                    ))
                    if len(out) >= page_size:
                        break
                if len(out) >= page_size:
                    break
        return out


class DataFilterGUI:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Refer Data Filter")
        self.resume_enabled = bool(int(getattr(args, "resume", 1)))
        self.yes_ratio = max(0.0, min(1.0, float(getattr(args, "yes_ratio", 0.5))))

        # 数据集
        train_ids_override, val_ids_override = self._split_overrides_by_version(args.dataset_version)
        ds = ReferKittiMOT(
            data_root=args.data_root,
            split=args.split,
            load_annotation=True,
            expression_sub_dir="expression",
            labels_with_ids_sub_dir="labels_with_ids/image_02",
            train_ids_override=train_ids_override,
            val_ids_override=val_ids_override,
        )
        self.enumerator = DataEnumerator(
            ds=ds,
            prompt_single_tpl=args.prompt_single_tpl or DEFAULT_PROMPT_SINGLE,
            prompt_video_tpl=args.prompt_video_tpl or DEFAULT_PROMPT_VIDEO,
            image_size=args.image_size,
            margin_ratio=args.margin_ratio,
            margin_px=args.margin_px,
            min_side=args.min_side,
            coord_mode=args.coord_mode,
            coord_decimals=args.coord_decimals,
            video_n_frames=args.video_n_frames,
            seed=args.seed,
        )

        # 会话统计
        self.saved_single = 0
        self.saved_single_pos = 0
        self.saved_single_neg = 0
        self.saved_video = 0
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "single"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "video"), exist_ok=True)

        # 当前页缓存（用于一键保存）
        self.current_single_samples: List[SingleSample] = []
        self.current_video_samples: List[VideoSample] = []

        # UI 布局：左侧主区 + 右侧统计/控制
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.sidebar = ttk.Frame(self.root, width=280)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=8, pady=8)

        # 控制区
        self.mode_var = tk.StringVar(value="single")
        mode_frame = ttk.LabelFrame(self.sidebar, text="模式")
        mode_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(mode_frame, text="单帧 (8项/页)", variable=self.mode_var, value="single", command=self._on_mode_change).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="视频 (2项/页)", variable=self.mode_var, value="video", command=self._on_mode_change).pack(anchor="w")

        seed_frame = ttk.LabelFrame(self.sidebar, text="采样")
        seed_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(seed_frame, text="随机种子:").pack(anchor="w")
        self.seed_entry = ttk.Entry(seed_frame)
        self.seed_entry.insert(0, str(args.seed))
        self.seed_entry.pack(fill=tk.X)
        ttk.Button(seed_frame, text="重采样", command=self._on_reshuffle).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(seed_frame, text="下一页", command=self._on_next_page).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(seed_frame, text="保存当前页所有", command=self._on_save_all).pack(fill=tk.X, pady=(4, 0))

        info_frame = ttk.LabelFrame(self.sidebar, text="信息")
        info_frame.pack(fill=tk.BOTH, expand=True)
        # 动态统计标签（不清空日志）
        self.stats_var = tk.StringVar(value="")
        self.stats_label = ttk.Label(info_frame, textvariable=self.stats_var, justify="left")
        self.stats_label.pack(fill=tk.X, pady=(0, 6))
        self.info_text = tk.Text(info_frame, height=18, wrap="word")
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self._write_info_initial()

        # 主展示区容器
        self.canvas = tk.Canvas(self.main_frame, bg="#f5f5f5")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.items_container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.items_container, anchor="nw")
        self.items_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # 滚动条（纵向）
        vbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=vbar.set)

        # 首次加载
        self.photo_refs: List[ImageTk.PhotoImage] = []
        self._render_page()

    def _split_overrides_by_version(self, version: str) -> Tuple[list, list]:
        v = (version or "v1").lower()
        if v == "v2":
            return (
                [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20],
                [5, 11, 13, 19],
            )
        return (
            [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20],
            [5, 11, 13],
        )

    def _write_info_initial(self):
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, f"数据根: {self.args.data_root}\n")
        self.info_text.insert(tk.END, f"版本: {self.args.dataset_version} | 划分: {self.args.split}\n")
        self.info_text.insert(tk.END, f"输出: {self.args.output_dir}\n")
        self._update_stats_text()

    def _update_stats_text(self):
        total_single_disk = self._count_items_on_disk(os.path.join(self.args.output_dir, "single"))
        total_video_disk = self._count_items_on_disk(os.path.join(self.args.output_dir, "video"))
        self.info_text.insert(tk.END, "\n—— 统计 ——\n")
        self.info_text.insert(tk.END, f"本次已保存（单帧/视频）: {self.saved_single} / {self.saved_video}\n")
        self.info_text.insert(tk.END, f"目录累计（单帧/视频）: {total_single_disk} / {total_video_disk}\n")

    @staticmethod
    def _count_items_on_disk(path: str) -> int:
        if not os.path.isdir(path):
            return 0
        cnt = 0
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".json"):
                    cnt += 1
        return cnt

    def _on_mode_change(self):
        # 切换模式时，刷新页面但不重置随机序列（保留当前推进）
        self._render_page()

    def _on_reshuffle(self):
        try:
            seed = int(self.seed_entry.get().strip())
        except Exception:
            messagebox.showerror("错误", "随机种子必须是整数")
            return
        self.enumerator.set_seed(seed)
        self._render_page()

    def _on_next_page(self):
        self._render_page()

    def _clear_items(self):
        for child in self.items_container.winfo_children():
            child.destroy()
        self.photo_refs.clear()

    def _render_page(self):
        self._clear_items()
        mode = self.mode_var.get()
        if mode == "single":
            samples = self._fetch_balanced_single_page(target=8, yes_ratio=self.yes_ratio)
            self.current_single_samples = samples[:]
            self.current_video_samples = []
            self._render_single_grid(samples)
            # 更新统计（当前页）
            pos = sum(1 for s in samples if int(s.y_true) == 1)
            neg = len(samples) - pos
            self._update_stats_label(current_single=(pos, neg))
        else:
            samples = self._fetch_unsaved_video_page(target=2)
            self.current_video_samples = samples[:]
            self.current_single_samples = []
            self._render_video_list(samples)
            self._update_stats_label(current_single=None)  # 视频页无 Yes/No 统计

    def _render_single_grid(self, samples: List[SingleSample]):
        # 2 行 x 4 列
        cols = 4
        tile_w = 280
        for idx, s in enumerate(samples):
            r = idx // cols
            c = idx % cols
            frame = ttk.Frame(self.items_container, padding=6, relief=tk.GROOVE)
            frame.grid(row=r, column=c, padx=6, pady=6, sticky="n")

            # 图像
            ph = ImageTk.PhotoImage(s.crop_image)
            self.photo_refs.append(ph)
            img_label = ttk.Label(frame, image=ph)
            img_label.pack()

            # 文本信息：序列、帧、ID、GT、prompt
            info = (
                f"seq: {s.sequence} | frame: {s.frame_idx} | id: {s.obj_id}\n"
                f"GT: {'Yes' if s.y_true == 1 else 'No'}\n"
                f"prompt: {self.args.prompt_single_tpl.format(coord=s.coord_str, sentence=s.sentence)}"
            )
            text = tk.Text(frame, width=36, height=7, wrap="word")
            text.insert(tk.END, info)
            text.configure(state="disabled")
            text.pack(pady=(4, 4))

            ttk.Button(
                frame,
                text="保留",
                command=lambda sample=s: self._save_single_sample(sample),
            ).pack(fill=tk.X)

    def _render_video_list(self, samples: List[VideoSample]):
        # 垂直两个样本；每个样本将多帧横向拼接显示
        for s in samples:
            frame = ttk.Frame(self.items_container, padding=6, relief=tk.GROOVE)
            frame.pack(fill=tk.X, pady=6)

            # 拼接横向图
            gap = 6
            if len(s.crops) > 0:
                w, h = s.crops[0].size
            else:
                w, h = 320, 320
            canvas_w = len(s.crops) * w + (len(s.crops) - 1) * gap if len(s.crops) > 0 else w
            canvas_h = h
            strip = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
            x0 = 0
            for im in s.crops:
                strip.paste(im, (x0, 0))
                x0 += w + gap
            ph = ImageTk.PhotoImage(strip)
            self.photo_refs.append(ph)
            img_label = ttk.Label(frame, image=ph)
            img_label.pack()

            info = (
                f"seq: {s.sequence} | id: {s.obj_id} | frames: {s.frames}\n"
                f"prompt(video): {self.args.prompt_video_tpl.format(coord=s.coord_str, sentence=s.sentence)}"
            )
            text = tk.Text(frame, height=5, wrap="word")
            text.insert(tk.END, info)
            text.configure(state="disabled")
            text.pack(fill=tk.X, pady=(4, 4))

            ttk.Button(
                frame,
                text="保留",
                command=lambda sample=s: self._save_video_sample(sample),
            ).pack(fill=tk.X)

    @staticmethod
    def _slugify(text: str, max_len: int = 80) -> str:
        import re as _re
        t = (text or "").strip().lower()
        t = _re.sub(r"\s+", "_", t)
        t = _re.sub(r"[^a-z0-9_\-]+", "", t)
        return t[:max_len] if len(t) > max_len else t

    def _save_single_sample(self, s: SingleSample):
        seq_dir = os.path.join(self.args.output_dir, "single", s.sequence)
        os.makedirs(seq_dir, exist_ok=True)
        base_name = f"f{int(s.frame_idx):06d}_id{s.obj_id}_{self._slugify(s.sentence, 40)}"
        img_path = os.path.join(seq_dir, base_name + ".jpg")
        meta_path = os.path.join(seq_dir, base_name + ".json")
        try:
            s.crop_image.save(img_path, quality=95)
            meta = {
                "mode": "single",
                "sequence": s.sequence,
                "frame_idx": int(s.frame_idx),
                "obj_id": int(s.obj_id),
                "bbox_xywh": list(map(int, s.bbox_xywh)),
                "sentence_idx": int(s.sentence_idx),
                "sentence": s.sentence,
                "y_true": int(s.y_true),
                "coord_str": s.coord_str,
                "prompt": self.args.prompt_single_tpl.format(coord=s.coord_str, sentence=s.sentence),
                "source_image": s.image_path,
                "saved_image": img_path,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.saved_single += 1
            if int(s.y_true) == 1:
                self.saved_single_pos += 1
            else:
                self.saved_single_neg += 1
            self.info_text.insert(tk.END, f"\n[保存] 单帧 -> {meta_path}\n")
            # 刷新统计
            self._update_stats_label()
        except Exception as e:
            messagebox.showerror("保存失败", f"保存单帧样本失败: {e}")

    def _save_video_sample(self, s: VideoSample):
        seq_dir = os.path.join(self.args.output_dir, "video", s.sequence)
        base_dir = os.path.join(seq_dir, f"id{s.obj_id}_{self._slugify(s.sentence, 40)}_f{int(s.frames[-1]):06d}")
        try:
            os.makedirs(base_dir, exist_ok=True)
            saved_paths = []
            for k, (im, ipath, bbox) in enumerate(zip(s.crops, s.image_paths, s.bboxes_xywh)):
                out_img = os.path.join(base_dir, f"frame_{k:02d}.jpg")
                im.save(out_img, quality=95)
                saved_paths.append({"frame_idx": int(s.frames[k]), "image": out_img, "source": ipath, "bbox_xywh": list(map(int, bbox))})
            meta = {
                "mode": "video",
                "sequence": s.sequence,
                "obj_id": int(s.obj_id),
                "frames": list(map(int, s.frames)),
                "sentence_idx": int(s.sentence_idx),
                "sentence": s.sentence,
                "coord_str": s.coord_str,
                "prompt": self.args.prompt_video_tpl.format(coord=s.coord_str, sentence=s.sentence),
                "items": saved_paths,
            }
            meta_path = os.path.join(base_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            self.saved_video += 1
            self.info_text.insert(tk.END, f"\n[保存] 视频 -> {meta_path}\n")
        except Exception as e:
            messagebox.showerror("保存失败", f"保存视频样本失败: {e}")

    def _on_save_all(self):
        mode = self.mode_var.get()
        if mode == "single":
            count = 0
            for s in list(self.current_single_samples):
                if self.resume_enabled and self._is_single_saved(s):
                    continue
                self._save_single_sample(s)
                count += 1
            self.info_text.insert(tk.END, f"\n[批量保存] 单帧本页共保存 {count} 项\n")
        else:
            count = 0
            for s in list(self.current_video_samples):
                if self.resume_enabled and self._is_video_saved(s):
                    continue
                self._save_video_sample(s)
                count += 1
            self.info_text.insert(tk.END, f"\n[批量保存] 视频本页共保存 {count} 项\n")

    # ========== 恢复与跳过已保存 ==========
    def _is_single_saved(self, s: SingleSample) -> bool:
        seq_dir = os.path.join(self.args.output_dir, "single", s.sequence)
        base_name = f"f{int(s.frame_idx):06d}_id{s.obj_id}_{self._slugify(s.sentence, 40)}"
        meta_path = os.path.join(seq_dir, base_name + ".json")
        return os.path.isfile(meta_path)

    def _is_video_saved(self, s: VideoSample) -> bool:
        seq_dir = os.path.join(self.args.output_dir, "video", s.sequence)
        base_dir = os.path.join(seq_dir, f"id{s.obj_id}_{self._slugify(s.sentence, 40)}_f{int(s.frames[-1]):06d}")
        meta_path = os.path.join(base_dir, "meta.json")
        return os.path.isfile(meta_path)

    def _fetch_balanced_single_page(self, target: int, yes_ratio: float) -> List[SingleSample]:
        """按 yes_ratio 目标比例获取单帧页；若某类不足，用另一类补齐。考虑 resume 跳过已保存。"""
        pos_target = int(round(target * float(yes_ratio)))
        neg_target = target - pos_target
        pos_list: List[SingleSample] = []
        neg_list: List[SingleSample] = []
        safety = 0
        # 尝试多批抓取直到满足目标或耗尽
        while (len(pos_list) < pos_target or len(neg_list) < neg_target) and safety < 64:
            batch = self.enumerator.next_single_page(page_size=target)
            if len(batch) == 0:
                break
            for s in batch:
                if self.resume_enabled and self._is_single_saved(s):
                    continue
                if int(s.y_true) == 1 and len(pos_list) < pos_target:
                    pos_list.append(s)
                elif int(s.y_true) == 0 and len(neg_list) < neg_target:
                    neg_list.append(s)
                # 提前结束
                if len(pos_list) >= pos_target and len(neg_list) >= neg_target:
                    break
            safety += 1
        # 若某类不足，用另一类补齐
        if len(pos_list) + len(neg_list) < target:
            safety2 = 0
            while len(pos_list) + len(neg_list) < target and safety2 < 64:
                batch = self.enumerator.next_single_page(page_size=target)
                if len(batch) == 0:
                    break
                for s in batch:
                    if self.resume_enabled and self._is_single_saved(s):
                        continue
                    if int(s.y_true) == 1 and s not in pos_list:
                        pos_list.append(s)
                    elif int(s.y_true) == 0 and s not in neg_list:
                        neg_list.append(s)
                    if len(pos_list) + len(neg_list) >= target:
                        break
                safety2 += 1
        # 组装并截断
        combined = (pos_list[:pos_target] + neg_list[:neg_target])
        # 如果仍未达标（比如 pos_target/neg_target极端且样本不足），从两类池子继续补齐
        i = 0
        pool = pos_list + neg_list
        while len(combined) < target and i < len(pool):
            if pool[i] not in combined:
                combined.append(pool[i])
            i += 1
        return combined[:target]

    def _fetch_unsaved_single_page(self, target: int = 8) -> List[SingleSample]:
        if not self.resume_enabled:
            return self.enumerator.next_single_page(page_size=target)
        collected: List[SingleSample] = []
        safety = 0
        while len(collected) < target and safety < 32:
            batch = self.enumerator.next_single_page(page_size=target)
            if len(batch) == 0:
                break
            for s in batch:
                if not self._is_single_saved(s):
                    collected.append(s)
                    if len(collected) >= target:
                        break
            safety += 1
        return collected

    def _fetch_unsaved_video_page(self, target: int = 2) -> List[VideoSample]:
        if not self.resume_enabled:
            return self.enumerator.next_video_page(page_size=target)
        collected: List[VideoSample] = []
        safety = 0
        while len(collected) < target and safety < 32:
            batch = self.enumerator.next_video_page(page_size=target)
            if len(batch) == 0:
                break
            for s in batch:
                if not self._is_video_saved(s):
                    collected.append(s)
                    if len(collected) >= target:
                        break
            safety += 1
        return collected

    def run(self):
        self.root.mainloop()

    # ========== 统计 ==========
    def _count_single_yes_no_on_disk(self) -> Tuple[int, int]:
        root = os.path.join(self.args.output_dir, "single")
        yes_cnt = 0
        no_cnt = 0
        if not os.path.isdir(root):
            return 0, 0
        for r, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(r, f), "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    if int(meta.get("y_true", 0)) == 1:
                        yes_cnt += 1
                    else:
                        no_cnt += 1
                except Exception:
                    continue
        return yes_cnt, no_cnt

    def _update_stats_label(self, current_single: Optional[Tuple[int, int]] = None):
        disk_yes, disk_no = self._count_single_yes_no_on_disk()
        lines = []
        lines.append(f"数据根: {self.args.data_root}")
        lines.append(f"版本/划分: {self.args.dataset_version}/{self.args.split}")
        lines.append(f"输出: {self.args.output_dir}")
        lines.append(f"—— 统计 ——")
        lines.append(f"目标Yes比例(单帧): {self.yes_ratio:.2f}")
        if current_single is not None:
            cy, cn = current_single
            lines.append(f"当前页(单帧) Yes/No: {cy}/{cn}")
        lines.append(f"本次保存(单帧) Yes/No: {self.saved_single_pos}/{self.saved_single_neg}")
        lines.append(f"目录累计(单帧) Yes/No: {disk_yes}/{disk_no}")
        self.stats_var.set("\n".join(lines))


def build_argparser():
    p = argparse.ArgumentParser(description="Refer 数据筛选器（GUI）")
    # 数据与版本
    p.add_argument("--data_root", type=str, default="/data/sq_2023/refer_kitti")
    p.add_argument("--dataset_version", type=str, default="v1", choices=["v1", "v2"])
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    # 图像/裁剪
    p.add_argument("--image_size", type=int, default=320)
    p.add_argument("--margin_ratio", type=float, default=0.1)
    p.add_argument("--margin_px", type=int, default=None)
    p.add_argument("--min_side", type=int, default=8)
    p.add_argument("--coord_mode", type=str, default="xy", choices=["xy", "xywh"])
    p.add_argument("--coord_decimals", type=int, default=3)
    # 视频
    p.add_argument("--video_n_frames", type=int, default=4)
    # Prompt
    p.add_argument("--prompt_single_tpl", type=str, default=DEFAULT_PROMPT_SINGLE)
    p.add_argument("--prompt_video_tpl", type=str, default=DEFAULT_PROMPT_VIDEO)
    # 采样
    p.add_argument("--seed", type=int, default=42)
    # 输出
    p.add_argument("--output_dir", type=str, default="./filter_outputs")
    # 恢复：默认启用（1），传 0 可关闭
    p.add_argument("--resume", type=int, default=1, help="是否跳过已保存样本并从输出目录恢复(1=启用,0=关闭)")
    # 单帧 Yes/No 比例控制
    p.add_argument("--yes_ratio", type=float, default=0.5, help="单帧页内Yes样本目标比例[0,1]")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    app = DataFilterGUI(args)
    app.run()


