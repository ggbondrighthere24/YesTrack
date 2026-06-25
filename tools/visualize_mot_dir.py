#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-visualize MOT candidate/pred files under runs/**/probs/mot.

Input format per line (from refer_llm/llm_eval pipeline):
  frame_idx, obj_id, x, y, w, h, score

Typical properties:
- frame_idx is 0-based (KITTI image naming is also 0-based: 000000.png)
- file name is the KITTI seq id, e.g. 0005.txt

This script will:
- read every *.txt under --mot-dir
- overlay boxes on KITTI frames
- write per-frame JPGs and an MP4 into the SAME directory (under a "vis" subdir)

Design constraints:
- avoid silent fallbacks: if image roots / frames can't be found, raise explicit errors.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Det = Tuple[int, float, float, float, float, float]  # (obj_id, x, y, w, h, score)


def _safe_int(x: str) -> int:
    return int(float(x))


def _safe_float(x: str) -> float:
    return float(x)


def _read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except UnicodeDecodeError:
        return [ln.strip() for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]


def parse_mot_file(mot_path: Path) -> Dict[int, List[Det]]:
    per_frame: Dict[int, List[Det]] = {}
    for ln in _read_lines(mot_path):
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 7:
            continue
        try:
            fidx = _safe_int(parts[0])
            oid = _safe_int(parts[1])
            x = _safe_float(parts[2])
            y = _safe_float(parts[3])
            w = _safe_float(parts[4])
            h = _safe_float(parts[5])
            score = _safe_float(parts[6])
        except Exception:
            continue
        per_frame.setdefault(fidx, []).append((oid, x, y, w, h, score))
    return per_frame


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    # Deterministic "random" BGR color per id.
    # Keep it bright-ish for readability.
    r = (track_id * 37) % 200 + 30
    g = (track_id * 17) % 200 + 30
    b = (track_id * 97) % 200 + 30
    return (int(b), int(g), int(r))


def draw_dets(image: np.ndarray, dets: List[Det], show_score: bool, thickness: int) -> np.ndarray:
    out = image.copy()
    for oid, x, y, w, h, score in dets:
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        color = _color_for_id(oid)
        cv2.rectangle(out, p1, p2, color, int(thickness))
        if show_score:
            label = f"{oid} {score:.2f}"
        else:
            label = f"{oid}"
        y_text = max(10, p1[1] - 4)
        cv2.putText(out, label, (p1[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _infer_image_root_from_run_config(mot_dir: Path) -> Optional[Path]:
    """
    Walk up ancestors to find a config.json containing "data_root", then build candidate image roots.
    We do NOT silently fall back; we only return a path if it exists.
    """
    cur = mot_dir.resolve()
    for parent in [cur] + list(cur.parents):
        cfg = parent / "config.json"
        if not cfg.is_file():
            continue
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        data_root = obj.get("data_root", None)
        if not isinstance(data_root, str) or not data_root:
            continue

        cand1 = Path(data_root) / "KITTI" / "training" / "image_02"
        cand2 = Path(data_root) / "training" / "image_02"
        if cand1.is_dir():
            return cand1
        if cand2.is_dir():
            return cand2
        return None
    return None


@dataclass
class FrameBaseResolution:
    mot_frame_base: int  # 0 or 1
    img_index_offset: int  # add to mot frame to obtain image index


def _resolve_frame_base(
    seq_dir: Path,
    frame_indices: List[int],
    img_ext: str,
    frame_base: str,
) -> FrameBaseResolution:
    """
    Determine how to map MOT frame index -> KITTI image file index.
    - If frame_base is "0": image_idx = fidx
    - If frame_base is "1": image_idx = fidx - 1
    - If frame_base is "auto": probe a few frames and decide, otherwise raise.
    """
    if frame_base not in {"auto", "0", "1"}:
        raise ValueError(f"Invalid --frame-base={frame_base}, expect auto/0/1")

    if frame_base == "0":
        return FrameBaseResolution(mot_frame_base=0, img_index_offset=0)
    if frame_base == "1":
        return FrameBaseResolution(mot_frame_base=1, img_index_offset=-1)

    # auto: probe up to 10 indices
    probe = frame_indices[:10] if len(frame_indices) > 10 else frame_indices
    if not probe:
        raise RuntimeError("No frames to probe for frame base resolution.")

    def exists_for_offset(off: int) -> bool:
        for f in probe:
            idx = f + off
            if idx < 0:
                return False
            p = seq_dir / f"{idx:06d}{img_ext}"
            if not p.is_file():
                return False
        return True

    ok0 = exists_for_offset(0)
    ok1 = exists_for_offset(-1)
    if ok0 and not ok1:
        return FrameBaseResolution(mot_frame_base=0, img_index_offset=0)
    if ok1 and not ok0:
        return FrameBaseResolution(mot_frame_base=1, img_index_offset=-1)

    # Ambiguous or neither works -> raise explicit error
    msg = [
        "Failed to auto-resolve frame base.",
        f"- seq_dir: {seq_dir}",
        f"- probed frames: {probe[:10]}",
        f"- exists when image_idx=fidx (offset 0): {ok0}",
        f"- exists when image_idx=fidx-1 (offset -1): {ok1}",
        "Please pass --frame-base 0 or --frame-base 1 explicitly.",
    ]
    raise RuntimeError("\n".join(msg))


def _infer_image_size(seq_dir: Path, first_img: Path) -> Tuple[int, int]:
    img = cv2.imread(str(first_img))
    if img is None:
        raise RuntimeError(f"Failed to read image: {first_img}")
    h, w = img.shape[:2]
    return (w, h)


def visualize_one_seq(
    mot_file: Path,
    image_root: Path,
    out_root: Path,
    img_ext: str,
    fps: int,
    write_frames: bool,
    write_video: bool,
    show_score: bool,
    thickness: int,
    frame_base: str,
    limit_frames: Optional[int],
) -> None:
    seq = mot_file.stem
    if len(seq) != 4 or not seq.isdigit():
        raise ValueError(f"Unexpected MOT filename (expect 4-digit seq like 0005.txt): {mot_file.name}")

    seq_dir = image_root / seq
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    dets_by_frame = parse_mot_file(mot_file)
    if not dets_by_frame:
        raise RuntimeError(f"No detections parsed from: {mot_file}")

    frame_ids = sorted(dets_by_frame.keys())
    resolution = _resolve_frame_base(seq_dir=seq_dir, frame_indices=frame_ids, img_ext=img_ext, frame_base=frame_base)

    # Choose frame iteration: contiguous range gives nicer videos.
    f_min, f_max = int(frame_ids[0]), int(frame_ids[-1])
    all_frames = list(range(f_min, f_max + 1))
    if limit_frames is not None:
        all_frames = all_frames[: int(limit_frames)]

    # Determine output paths
    vis_dir = out_root / "vis" / seq
    vis_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = (out_root / "vis" / f"{seq}.mp4").as_posix()

    # Prepare video writer if requested
    writer: Optional[cv2.VideoWriter] = None
    if write_video:
        # pick first existing frame to infer size
        first = None
        for f in all_frames:
            img_idx = f + resolution.img_index_offset
            p = seq_dir / f"{img_idx:06d}{img_ext}"
            if p.is_file():
                first = p
                break
        if first is None:
            raise RuntimeError(f"Could not find any frame images under {seq_dir} for seq={seq}")
        w, h = _infer_image_size(seq_dir=seq_dir, first_img=first)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_path, fourcc, float(fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {mp4_path}")

    for f in all_frames:
        img_idx = f + resolution.img_index_offset
        if img_idx < 0:
            raise RuntimeError(f"Resolved negative image index: frame={f}, offset={resolution.img_index_offset}")
        img_path = seq_dir / f"{img_idx:06d}{img_ext}"
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path} (frame={f}, seq={seq})")

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        dets = dets_by_frame.get(f, [])
        vis = draw_dets(img, dets, show_score=show_score, thickness=thickness)
        cv2.putText(
            vis,
            f"seq={seq} frame={f} img={img_idx:06d}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if write_frames:
            out_img = vis_dir / f"{img_idx:06d}.jpg"
            ok = cv2.imwrite(str(out_img), vis)
            if not ok:
                raise RuntimeError(f"Failed to write image: {out_img}")

        if writer is not None:
            writer.write(vis)

    if writer is not None:
        writer.release()


def main() -> None:
    p = argparse.ArgumentParser("Batch visualize runs/**/probs/mot/*.txt on KITTI frames")
    p.add_argument(
        "--mot-dir",
        type=str,
        required=True,
        help="Directory containing <seq>.txt, e.g. runs/.../probs/mot",
    )
    p.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="KITTI image_02 root, contains <seq>/<frame>.png. If omitted, infer from nearest runs/**/config.json:data_root.",
    )
    p.add_argument("--img-ext", type=str, default=".png", help="Image extension under image-root, default .png")
    p.add_argument("--fps", type=int, default=10, help="Output mp4 FPS")
    p.add_argument("--write-frames", action="store_true", help="Write per-frame JPGs into <mot-dir>/vis/<seq>/")
    p.add_argument("--write-video", action="store_true", help="Write <mot-dir>/vis/<seq>.mp4")
    p.add_argument("--show-score", action="store_true", help="Render score next to id")
    p.add_argument("--thickness", type=int, default=2, help="Box line thickness")
    p.add_argument(
        "--frame-base",
        type=str,
        default="auto",
        choices=["auto", "0", "1"],
        help="Frame base of MOT file. 0 means frame_idx aligns with image index; 1 means image index = frame_idx-1.",
    )
    p.add_argument("--limit-frames", type=int, default=None, help="Optional max number of frames to render per seq")
    args = p.parse_args()

    mot_dir = Path(args.mot_dir).resolve()
    if not mot_dir.is_dir():
        raise FileNotFoundError(f"--mot-dir is not a directory: {mot_dir}")

    image_root: Optional[Path]
    if args.image_root is not None and str(args.image_root).strip():
        image_root = Path(args.image_root).resolve()
    else:
        image_root = _infer_image_root_from_run_config(mot_dir)

    if image_root is None:
        raise RuntimeError(
            "Could not infer --image-root from run config.json (data_root). "
            "Please pass --image-root explicitly, e.g. /path/to/KITTI/training/image_02"
        )
    if not image_root.is_dir():
        raise FileNotFoundError(f"--image-root not found or not a directory: {image_root}")

    mot_files = sorted([p for p in mot_dir.glob("*.txt") if p.is_file()])
    if not mot_files:
        raise RuntimeError(f"No *.txt found under: {mot_dir}")

    write_frames = bool(args.write_frames)
    write_video = bool(args.write_video)
    if not write_frames and not write_video:
        raise RuntimeError("Nothing to do: please pass --write-frames and/or --write-video")

    for mot_file in mot_files:
        visualize_one_seq(
            mot_file=mot_file,
            image_root=image_root,
            out_root=mot_dir,
            img_ext=str(args.img_ext),
            fps=int(args.fps),
            write_frames=write_frames,
            write_video=write_video,
            show_score=bool(args.show_score),
            thickness=int(args.thickness),
            frame_base=str(args.frame_base),
            limit_frames=args.limit_frames,
        )

    print(f"[OK] Saved visualizations under: {mot_dir / 'vis'}")


if __name__ == "__main__":
    main()

