#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple, Optional

import cv2


def _list_frame_files(seq_dir: str, img_ext: str) -> List[str]:
    files = []
    for f in os.listdir(seq_dir):
        if f.endswith(img_ext):
            files.append(f)
    # filenames are like 000000.png; lexicographic sort works
    files.sort()
    return files


def _pick_range(files: List[str], start_idx: Optional[int], end_idx: Optional[int]) -> List[str]:
    if not files:
        return []
    s = 0 if start_idx is None else max(0, int(start_idx))
    e = (len(files) - 1) if end_idx is None else min(len(files) - 1, int(end_idx))
    if e < s:
        return []
    return files[s : e + 1]


def _infer_size(first_img_path: str) -> Tuple[int, int]:
    img = cv2.imread(first_img_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {first_img_path}")
    h, w = img.shape[:2]
    return w, h


def export_seq_to_mp4(
    image_root: str,
    seq: str,
    out_path: str,
    fps: int,
    img_ext: str,
    start_idx: Optional[int],
    end_idx: Optional[int],
    quiet: bool,
) -> None:
    seq_dir = os.path.join(image_root, seq)
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    files = _list_frame_files(seq_dir, img_ext)
    files = _pick_range(files, start_idx, end_idx)
    if not files:
        raise RuntimeError(f"No frames found for seq={seq} under {seq_dir} (ext={img_ext})")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    first_img_path = os.path.join(seq_dir, files[0])
    w, h = _infer_size(first_img_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    for fname in files:
        img_path = os.path.join(seq_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            # skip unreadable frame
            if not quiet:
                print(f"[Warn] Skip unreadable frame: {img_path}")
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(img)

    writer.release()
    if not quiet:
        print(f"[OK] Wrote: {out_path}")


def main():
    parser = argparse.ArgumentParser("Export KITTI image_02 sequences to <seq>.mp4 (supports multiple seqs)")
    parser.add_argument(
        "--image-root",
        type=str,
        default="/data/sq_2023/refer_kitti/KITTI/training/image_02",
        help="KITTI image_02 root, contains <seq>/<frame>.png",
    )
    parser.add_argument("--seqs", type=str, nargs="+", default=['0005', '0011', '0013','0019'], help="Sequence IDs, e.g. 0005 0011 0013")
    parser.add_argument("--out-dir", type=str, default="v2_vis_seq_mp4", help="Output directory for mp4 files")
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS")
    parser.add_argument("--img-ext", type=str, default=".png", help="Image extension, default .png")
    parser.add_argument("--start-idx", type=int, default=None, help="Start frame index (0-based, inclusive)")
    parser.add_argument("--end-idx", type=int, default=None, help="End frame index (0-based, inclusive)")
    parser.add_argument("--quiet", action="store_true", help="Less printing")
    args = parser.parse_args()

    image_root = os.path.abspath(args.image_root)
    out_dir = os.path.abspath(args.out_dir)

    for seq in args.seqs:
        out_path = os.path.join(out_dir, f"{seq}.mp4")
        export_seq_to_mp4(
            image_root=image_root,
            seq=seq,
            out_path=out_path,
            fps=args.fps,
            img_ext=args.img_ext,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()


