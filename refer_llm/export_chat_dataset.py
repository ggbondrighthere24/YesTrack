from __future__ import annotations

"""
Export the training dataset used by `refer_llm/llm_train.py` into a chat-style JSON format:

[
  {
    "conversations": [
      {"from": "human", "value": "<image><image>用户指令"},
      {"from": "gpt", "value": "模型回答"}
    ],
    "images": ["图像路径", "图像路径"]
  }
]

Notes
- Single-frame and multi-frame are exported as TWO separate datasets (two JSON files).
- `<image>` tokens count will always match the number of image paths in `images`.
- Image processing (crop/resize/save) is done with multi-threading.
- JSON writing is single-threaded and deterministic (ordered by sample index).
- `--seed` controls negative downsample determinism (via torch RNG) + optional shuffle order.
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from refer_llm.crop_utils import crop_with_margin
from refer_llm.data.datasets import build_refer_dataset, QwenReferYesNoDataset, QwenReferVideoYesNoDataset

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_split_overrides(dataset_version: str) -> Tuple[List[int], List[int]]:
    """
    Keep this consistent with `refer_llm/llm_train.py`.
    """
    if str(dataset_version) == "v2":
        train_ids_override = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20]
        val_ids_override = [5, 11, 13, 19]
    else:
        train_ids_override = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        val_ids_override = [5, 11, 13]
    return train_ids_override, val_ids_override


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save_image(
    im: Image.Image,
    out_path: str,
    *,
    image_format: str = "jpg",
    jpg_quality: int = 90,
    png_compress_level: int = 6,
):
    _ensure_dir(os.path.dirname(out_path))
    fmt = str(image_format).lower().strip()
    if fmt in ("jpg", "jpeg"):
        # JPEG is lossy; quality balances size vs fidelity.
        im.save(out_path, format="JPEG", quality=int(jpg_quality))
        return
    if fmt == "png":
        # PNG is lossless; compress_level balances size vs CPU time.
        im.save(out_path, format="PNG", compress_level=int(png_compress_level))
        return
    raise ValueError(f"Unsupported image_format={image_format!r} (expected jpg/png)")


def _norm_coord_str(
    W: int,
    H: int,
    x: float,
    y: float,
    w: float,
    h: float,
    coord_mode: str,
    coord_decimals: int,
) -> str:
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    nx = max(0.0, min(1.0, cx / float(W)))
    ny = max(0.0, min(1.0, cy / float(H)))
    fmt = "{:." + str(max(0, int(coord_decimals))) + "f}"
    if str(coord_mode) == "xywh":
        nw = max(0.0, min(1.0, w / float(W)))
        nh = max(0.0, min(1.0, h / float(H)))
        return f"{fmt.format(nx)} {fmt.format(ny)} {fmt.format(nw)} {fmt.format(nh)}"
    return f"{fmt.format(nx)} {fmt.format(ny)}"


def _human_value(n_images: int, prompt: str) -> str:
    # No spaces: "<image><image>..."
    return ("<image>" * int(n_images)) + str(prompt)


@dataclass(frozen=True)
class ExportResult:
    idx: int
    record: Dict[str, Any]


class JsonListStreamWriter:
    """
    Single-threaded streaming JSON list writer with deterministic ordering.
    """

    def __init__(self, out_path: str):
        self.out_path = out_path
        _ensure_dir(os.path.dirname(out_path))
        self.f = open(out_path, "w", encoding="utf-8")
        self.first = True
        self.f.write("[\n")

    def write_one(self, obj: Dict[str, Any]):
        if not self.first:
            self.f.write(",\n")
        self.first = False
        self.f.write(json.dumps(obj, ensure_ascii=False))

    def close(self):
        self.f.write("\n]\n")
        self.f.close()


def _process_single_one(
    idx: int,
    sample: Tuple[str, int, int, int],
    *,
    base_ds,
    exprs_by_seq: Dict[str, List[Dict[str, Any]]],
    out_image_root: str,
    image_size: int,
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    coord_mode: str,
    coord_decimals: int,
    prompt_single_tpl: str,
    jpg_quality: int,
    image_format: str,
    png_compress_level: int,
    yes_text: str,
    no_text: str,
    store_absolute_paths: bool,
) -> ExportResult:
    seq, frame_idx, ann_idx, sentence_idx = sample
    img_path = base_ds.image_paths[seq][frame_idx]
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    ann = base_ds.annotations[seq][frame_idx]
    x, y, w, h = ann["bbox"][ann_idx].tolist()
    obj_id = int(ann["id"][ann_idx].item())

    patch = crop_with_margin(
        image=image,
        bbox_xywh=(x, y, w, h),
        margin_ratio=margin_ratio,
        margin_px=margin_px,
        min_side=min_side,
    )
    if patch is None:
        raise RuntimeError(f"crop_with_margin failed: idx={idx}, seq={seq}, frame={frame_idx}, ann_idx={ann_idx}")
    patch = patch.resize((int(image_size), int(image_size)))

    exprs = exprs_by_seq[seq]
    sentence = exprs[sentence_idx].get("sentence", "")
    label_map = exprs[sentence_idx].get("label", {})
    ids = label_map.get(str(frame_idx), [])
    try:
        pos_ids = set(int(i) for i in ids)
    except Exception:
        pos_ids = set()
    y_label = 1 if obj_id in pos_ids else 0
    answer = yes_text if y_label == 1 else no_text

    coord_str = _norm_coord_str(W, H, x, y, w, h, coord_mode=coord_mode, coord_decimals=coord_decimals)
    prompt = str(prompt_single_tpl).format(coord=coord_str, sentence=sentence)

    ext = "png" if str(image_format).lower().strip() == "png" else "jpg"
    out_path = os.path.join(out_image_root, "single", seq, f"{idx:09d}.{ext}")
    _save_image(
        patch,
        out_path,
        image_format=image_format,
        jpg_quality=jpg_quality,
        png_compress_level=png_compress_level,
    )
    img_json_path = os.path.abspath(out_path) if store_absolute_paths else out_path

    rec = {
        "conversations": [
            {"from": "human", "value": _human_value(1, prompt)},
            {"from": "gpt", "value": answer},
        ],
        "images": [img_json_path],
    }
    return ExportResult(idx=idx, record=rec)


def _process_video_one(
    idx: int,
    sample: Tuple[str, int, int, int],
    *,
    base_ds,
    exprs_by_seq: Dict[str, List[Dict[str, Any]]],
    out_image_root: str,
    image_size: int,
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    coord_mode: str,
    coord_decimals: int,
    prompt_video_tpl: str,
    video_n_frames: int,
    jpg_quality: int,
    image_format: str,
    png_compress_level: int,
    yes_text: str,
    no_text: str,
    store_absolute_paths: bool,
) -> ExportResult:
    seq, frame_idx, ann_idx, sentence_idx = sample
    img_path_cur = base_ds.image_paths[seq][frame_idx]
    image_cur = Image.open(img_path_cur).convert("RGB")

    ann = base_ds.annotations[seq][frame_idx]
    x, y, w, h = ann["bbox"][ann_idx].tolist()
    obj_id = int(ann["id"][ann_idx].item())

    # Build patch sequence (same logic as QwenReferVideoYesNoDataset)
    images_seq: List[Image.Image] = []
    frame_ids: List[int] = []
    start_f = max(0, int(frame_idx) - int(video_n_frames) + 1)
    for fidx in range(start_f, int(frame_idx) + 1):
        ann_f = base_ds.annotations[seq][fidx]
        ids_f = ann_f["id"].tolist()
        if obj_id not in ids_f:
            continue
        idx_f = ids_f.index(obj_id)
        x_f, y_f, w_f, h_f = ann_f["bbox"][idx_f].tolist()
        img_path_f = base_ds.image_paths[seq][fidx]
        image_f = Image.open(img_path_f).convert("RGB")
        patch_f = crop_with_margin(
            image=image_f,
            bbox_xywh=(x_f, y_f, w_f, h_f),
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
        )
        if patch_f is None:
            raise RuntimeError(f"crop_with_margin failed(video): idx={idx}, seq={seq}, fidx={fidx}, obj_id={obj_id}")
        patch_f = patch_f.resize((int(image_size), int(image_size)))
        images_seq.append(patch_f)
        frame_ids.append(int(fidx))

    if len(images_seq) == 0:
        patch_cur = crop_with_margin(
            image=image_cur,
            bbox_xywh=(x, y, w, h),
            margin_ratio=margin_ratio,
            margin_px=margin_px,
            min_side=min_side,
        )
        if patch_cur is None:
            raise RuntimeError(f"crop_with_margin failed(video fallback): idx={idx}, seq={seq}, frame={frame_idx}")
        patch_cur = patch_cur.resize((int(image_size), int(image_size)))
        images_seq = [patch_cur]
        frame_ids = [int(frame_idx)]

    # Label (based on current frame, consistent with training code)
    exprs = exprs_by_seq[seq]
    sentence = exprs[sentence_idx].get("sentence", "")
    label_map = exprs[sentence_idx].get("label", {})
    ids = label_map.get(str(frame_idx), [])
    try:
        pos_ids = set(int(i) for i in ids)
    except Exception:
        pos_ids = set()
    y_label = 1 if obj_id in pos_ids else 0
    answer = yes_text if y_label == 1 else no_text

    W, H = image_cur.size
    coord_str = _norm_coord_str(W, H, x, y, w, h, coord_mode=coord_mode, coord_decimals=coord_decimals)
    coords_str = f"<{coord_str}>"
    prompt = str(prompt_video_tpl).format(sentence=sentence, coord=coord_str, coords=coords_str)

    # Save sequence images
    out_paths: List[str] = []
    ext = "png" if str(image_format).lower().strip() == "png" else "jpg"
    for j, (imj, fidx) in enumerate(zip(images_seq, frame_ids)):
        out_path = os.path.join(out_image_root, "video", seq, f"{idx:09d}", f"f{int(fidx):06d}_{j}.{ext}")
        _save_image(
            imj,
            out_path,
            image_format=image_format,
            jpg_quality=jpg_quality,
            png_compress_level=png_compress_level,
        )
        out_paths.append(os.path.abspath(out_path) if store_absolute_paths else out_path)

    rec = {
        "conversations": [
            {"from": "human", "value": _human_value(len(out_paths), prompt)},
            {"from": "gpt", "value": answer},
        ],
        "images": out_paths,
    }
    return ExportResult(idx=idx, record=rec)


def _export_mode(
    *,
    mode: str,
    base_ds,
    samples: List[Tuple[str, int, int, int]],
    exprs_by_seq: Dict[str, List[Dict[str, Any]]],
    out_json_path: str,
    out_image_root: str,
    workers: int,
    image_size: int,
    margin_ratio: float,
    margin_px: Optional[int],
    min_side: int,
    coord_mode: str,
    coord_decimals: int,
    prompt_single_tpl: str,
    prompt_video_tpl: str,
    video_n_frames: int,
    jpg_quality: int,
    image_format: str,
    png_compress_level: int,
    yes_text: str,
    no_text: str,
    max_samples: int,
    shuffle: bool,
    seed: int,
    store_absolute_paths: bool,
    show_tqdm: bool,
):
    if max_samples > 0:
        samples = samples[: int(max_samples)]

    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(samples)

    # Stream write while processing:
    # - Image processing is multi-threaded
    # - JSON writing is single-threaded and ordered by sample index (deterministic)
    writer = JsonListStreamWriter(out_json_path)
    try:
        def _work(item: Tuple[int, Tuple[str, int, int, int]]) -> ExportResult:
            i, s = item
            if str(mode) == "single":
                return _process_single_one(
                    i,
                    s,
                    base_ds=base_ds,
                    exprs_by_seq=exprs_by_seq,
                    out_image_root=out_image_root,
                    image_size=image_size,
                    margin_ratio=margin_ratio,
                    margin_px=margin_px,
                    min_side=min_side,
                    coord_mode=coord_mode,
                    coord_decimals=coord_decimals,
                    prompt_single_tpl=prompt_single_tpl,
                    jpg_quality=jpg_quality,
                    image_format=image_format,
                    png_compress_level=png_compress_level,
                    yes_text=yes_text,
                    no_text=no_text,
                    store_absolute_paths=store_absolute_paths,
                )
            return _process_video_one(
                i,
                s,
                base_ds=base_ds,
                exprs_by_seq=exprs_by_seq,
                out_image_root=out_image_root,
                image_size=image_size,
                margin_ratio=margin_ratio,
                margin_px=margin_px,
                min_side=min_side,
                coord_mode=coord_mode,
                coord_decimals=coord_decimals,
                prompt_video_tpl=prompt_video_tpl,
                video_n_frames=video_n_frames,
                jpg_quality=jpg_quality,
                image_format=image_format,
                png_compress_level=png_compress_level,
                yes_text=yes_text,
                no_text=no_text,
                store_absolute_paths=store_absolute_paths,
            )

        total = len(samples)
        done_n = 0
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            # executor.map yields results in input order => we can write immediately (streaming + deterministic)
            it = ex.map(_work, enumerate(samples))
            if show_tqdm and tqdm is not None:
                it = tqdm(it, total=total, desc=f"export:{mode}", unit="sample", dynamic_ncols=True)
            for res in it:
                writer.write_one(res.record)
                done_n += 1
                if (not show_tqdm) or (tqdm is None):
                    if done_n % 5000 == 0 or done_n == total:
                        print(f"[{mode}] processed {done_n}/{total} samples...")
    finally:
        writer.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export Refer-KITTI-MOT training samples to chat JSON format.")
    p.add_argument("--data_root", type=str, default="/data/sq_2023/refer_kitti")
    p.add_argument("--dataset_version", type=str, default="v1", choices=["v1", "v2"])
    p.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p.add_argument("--out_dir", type=str, default="v1ds", help="Output directory (will create images/ and json files).")

    # Modes / outputs
    p.add_argument("--export_single", type=int, default=1, help="1=export single-frame dataset, 0=skip")
    p.add_argument("--export_video", type=int, default=1, help="1=export multi-frame dataset, 0=skip")
    p.add_argument("--single_json_name", type=str, default="single.json")
    p.add_argument("--video_json_name", type=str, default="video.json")

    # Processing / determinism
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=12, help="Thread workers for image processing.")
    p.add_argument("--shuffle", type=int, default=0, help="1=shuffle sample order before exporting (seed controlled).")
    p.add_argument("--max_samples", type=int, default=0, help="0=all, else export only first N samples.")
    p.add_argument("--show_tqdm", type=int, default=1, help="1=show progress bar; 0=disable.")

    # Image crop/resize params (match llm_train defaults)
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--margin_ratio", type=float, default=0.2)
    p.add_argument("--margin_px", type=int, default=None)
    p.add_argument("--min_side", type=int, default=8)
    p.add_argument("--coord_mode", type=str, default="xy", choices=["xy", "xywh"])
    p.add_argument("--coord_decimals", type=int, default=3)
    p.add_argument("--video_n_frames", type=int, default=4)
    p.add_argument("--jpg_quality", type=int, default=90)
    p.add_argument("--image_format", type=str, default="png", choices=["jpg", "png"], help="Export cropped patches as jpg (lossy) or png (lossless).")
    p.add_argument("--png_compress_level", type=int, default=9, help="PNG compress_level (0-9). Higher=smaller but slower.")

    # Dataset sampling params (match llm_train defaults)
    p.add_argument("--negative_downsample", type=float, default=1.0)
    p.add_argument("--oversample_pedestrian", action="store_true")
    p.add_argument("--oversample_factor", type=int, default=4)

    # Prompt templates
    p.add_argument(
        "--prompt_single_tpl",
        type=str,
        default="The normalized position of the car or person in the picture is <{coord}>.Determine whether this description matches this image: {sentence}. Answer Yes or No.",
    )
    p.add_argument(
        "--prompt_video_tpl",
        type=str,
        default="This is a short video clip of a car or person at <{coord}> across frames. The target may include motion cues; consider background and temporal context when deciding if the description matches this target: {sentence}. Answer Yes or No.",
    )

    # Answer tokens
    p.add_argument("--yes_text", type=str, default="Yes")
    p.add_argument("--no_text", type=str, default="No")

    # Paths
    p.add_argument("--absolute_paths", type=int, default=1, help="1=store absolute image paths in json; 0=relative paths.")

    return p


def main(argv: Optional[List[str]] = None):
    args = build_argparser().parse_args(argv)
    _set_seed(int(args.seed))

    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)
    out_images_root = os.path.join(out_dir, "images")
    _ensure_dir(out_images_root)

    train_ids_override, val_ids_override = _get_split_overrides(args.dataset_version)
    base_ds = build_refer_dataset(
        data_root=args.data_root,
        split=args.split,
        train_ids_override=train_ids_override,
        val_ids_override=val_ids_override,
    )

    # Preload expressions to avoid repeated disk IO in threads
    exprs_by_seq: Dict[str, List[Dict[str, Any]]] = {}
    for seq in base_ds.sequence_names:
        exprs_by_seq[seq] = base_ds._load_expressions_for_sequence(seq)
        if len(exprs_by_seq[seq]) == 0:
            raise RuntimeError(f"Missing expressions for sequence={seq} (split={args.split})")

    store_absolute_paths = bool(int(args.absolute_paths))

    # Build sample lists using the SAME selection logic as training datasets (uses torch RNG -> seed controls it).
    oversample_seq = "0016" if bool(args.oversample_pedestrian) else None
    oversample_factor = int(args.oversample_factor)

    if int(args.export_single) == 1:
        ds_single = QwenReferYesNoDataset(
            refer_dataset=base_ds,
            image_size=int(args.image_size),
            margin_ratio=float(args.margin_ratio),
            margin_px=args.margin_px,
            min_side=int(args.min_side),
            max_text_len=999999,  # unused here
            negative_downsample=float(args.negative_downsample),
            coord_mode=str(args.coord_mode),
            coord_decimals=int(args.coord_decimals),
            prompt_single_tpl=str(args.prompt_single_tpl),
            oversample_seq=oversample_seq,
            oversample_factor=oversample_factor,
        )
        single_samples = list(ds_single.samples)
        print(f"[single] samples={len(single_samples)}")
        _export_mode(
            mode="single",
            base_ds=base_ds,
            samples=single_samples,
            exprs_by_seq=exprs_by_seq,
            out_json_path=os.path.join(out_dir, str(args.single_json_name)),
            out_image_root=out_images_root,
            workers=int(args.workers),
            image_size=int(args.image_size),
            margin_ratio=float(args.margin_ratio),
            margin_px=args.margin_px,
            min_side=int(args.min_side),
            coord_mode=str(args.coord_mode),
            coord_decimals=int(args.coord_decimals),
            prompt_single_tpl=str(args.prompt_single_tpl),
            prompt_video_tpl=str(args.prompt_video_tpl),
            video_n_frames=int(args.video_n_frames),
            jpg_quality=int(args.jpg_quality),
            image_format=str(args.image_format),
            png_compress_level=int(args.png_compress_level),
            yes_text=str(args.yes_text),
            no_text=str(args.no_text),
            max_samples=int(args.max_samples),
            shuffle=bool(int(args.shuffle)),
            seed=int(args.seed),
            store_absolute_paths=store_absolute_paths,
            show_tqdm=bool(int(args.show_tqdm)),
        )
        print(f"[single] wrote: {os.path.join(out_dir, str(args.single_json_name))}")

    if int(args.export_video) == 1:
        ds_video = QwenReferVideoYesNoDataset(
            refer_dataset=base_ds,
            image_size=int(args.image_size),
            margin_ratio=float(args.margin_ratio),
            margin_px=args.margin_px,
            min_side=int(args.min_side),
            max_text_len=999999,  # unused here
            negative_downsample=float(args.negative_downsample),
            video_n_frames=int(args.video_n_frames),
            prompt_video_tpl=str(args.prompt_video_tpl),
            oversample_seq=oversample_seq,
            oversample_factor=oversample_factor,
            coord_mode=str(args.coord_mode),
            coord_decimals=int(args.coord_decimals),
        )
        video_samples = list(ds_video.samples)
        print(f"[video] samples={len(video_samples)}")
        _export_mode(
            mode="video",
            base_ds=base_ds,
            samples=video_samples,
            exprs_by_seq=exprs_by_seq,
            out_json_path=os.path.join(out_dir, str(args.video_json_name)),
            out_image_root=out_images_root,
            workers=int(args.workers),
            image_size=int(args.image_size),
            margin_ratio=float(args.margin_ratio),
            margin_px=args.margin_px,
            min_side=int(args.min_side),
            coord_mode=str(args.coord_mode),
            coord_decimals=int(args.coord_decimals),
            prompt_single_tpl=str(args.prompt_single_tpl),
            prompt_video_tpl=str(args.prompt_video_tpl),
            video_n_frames=int(args.video_n_frames),
            jpg_quality=int(args.jpg_quality),
            image_format=str(args.image_format),
            png_compress_level=int(args.png_compress_level),
            yes_text=str(args.yes_text),
            no_text=str(args.no_text),
            max_samples=int(args.max_samples),
            shuffle=bool(int(args.shuffle)),
            seed=int(args.seed),
            store_absolute_paths=store_absolute_paths,
            show_tqdm=bool(int(args.show_tqdm)),
        )
        print(f"[video] wrote: {os.path.join(out_dir, str(args.video_json_name))}")


if __name__ == "__main__":
    main()


