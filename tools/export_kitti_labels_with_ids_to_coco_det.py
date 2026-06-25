#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export KITTI Tracking sequences (image_02) + labels_with_ids (normalized xywh) to COCO detection JSON.

Assumptions (match data/refer_kitti_mot.py):
- image_root contains: <seq>/<frame:06d>.png   (0-based)
- labels_root contains: <seq>/<frame:06d>.txt  lines: cls id x y w h
  where x,y,w,h are normalized to [0,1] and (x,y) is top-left.

This script:
- creates a COCO detection dataset with a SINGLE category (id=0)
- splits by seq id: val seqs are provided; all other seqs under image_root go to train
- optionally renders MP4 videos for val seqs with GT boxes for sanity checking

Design constraints:
- avoid silent fallbacks: missing directories / unreadable images / malformed labels -> raise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2


EPS = 1e-6
BOUND_TOL_PX = 2.0  # tolerate tiny rounding errors when converting normalized coords -> pixels


def _is_seq_dir(p: Path) -> bool:
    return p.is_dir() and len(p.name) == 4 and p.name.isdigit()


def list_seqs(image_root: Path) -> List[str]:
    if not image_root.is_dir():
        raise FileNotFoundError(f"image_root not found or not a directory: {image_root}")
    seqs = sorted([p.name for p in image_root.iterdir() if _is_seq_dir(p)])
    if not seqs:
        raise RuntimeError(f"No <seq>/ directories found under image_root: {image_root}")
    return seqs


def infer_labels_root(image_root: Path) -> Optional[Path]:
    """
    If image_root is .../KITTI/training/image_02, infer .../KITTI/labels_with_ids/image_02.
    Returns the inferred path ONLY if it exists; otherwise None.
    """
    parts = list(image_root.parts)
    if len(parts) < 3:
        return None
    # expected suffix: KITTI/training/image_02
    if image_root.name != "image_02":
        return None
    if image_root.parent.name != "training":
        return None
    if image_root.parent.parent.name != "KITTI":
        return None
    cand = image_root.parent.parent / "labels_with_ids" / "image_02"
    return cand if cand.is_dir() else None


def _read_lines(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except UnicodeDecodeError:
        return [ln.strip() for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]


def read_image_size(img_path: Path) -> Tuple[int, int]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]
    return (int(w), int(h))


@dataclass(frozen=True)
class DetBox:
    bbox_xywh: Tuple[float, float, float, float]
    track_id: int


def parse_labels_with_ids_txt(label_path: Path, width: int, height: int) -> List[DetBox]:
    """
    Parse one per-frame labels_with_ids txt.
    Each line: cls id x y w h (normalized, top-left xy).
    """
    dets: List[DetBox] = []
    for ln in _read_lines(label_path):
        parts = ln.split()
        if len(parts) < 6:
            raise ValueError(f"Invalid label line (need >=6 fields) in {label_path}: {ln}")

        try:
            track_id = int(float(parts[1]))
            nx = float(parts[2])
            ny = float(parts[3])
            nw = float(parts[4])
            nh = float(parts[5])
        except Exception as e:
            raise ValueError(f"Failed to parse label line in {label_path}: {ln}") from e

        # Validate normalized coords (strict-ish; allow tiny eps).
        if not (-EPS <= nx <= 1.0 + EPS and -EPS <= ny <= 1.0 + EPS and -EPS <= nw <= 1.0 + EPS and -EPS <= nh <= 1.0 + EPS):
            raise ValueError(f"Normalized bbox out of [0,1] in {label_path}: {ln}")
        if nw < 0.0 or nh < 0.0:
            raise ValueError(f"Negative bbox size in {label_path}: {ln}")

        x = nx * width
        y = ny * height
        w = nw * width
        h = nh * height

        # For COCO: bbox must be positive area; if zero, drop explicitly.
        if w <= 0.0 or h <= 0.0:
            continue

        # Clamp to image bounds (no silent fallbacks for gross errors; clamping only fixes eps-level).
        x2 = x + w
        y2 = y + h
        if x < -BOUND_TOL_PX or y < -BOUND_TOL_PX or x2 > width + BOUND_TOL_PX or y2 > height + BOUND_TOL_PX:
            raise ValueError(
                f"BBox exceeds image bounds by >{BOUND_TOL_PX}px; please check label normalization.\n"
                f"- label: {label_path}\n"
                f"- bbox(px): {(x, y, w, h)}\n"
                f"- image(w,h): {(width, height)}"
            )
        x = max(0.0, min(float(width - 1), x))
        y = max(0.0, min(float(height - 1), y))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        w = max(0.0, x2 - x)
        h = max(0.0, y2 - y)
        if w <= 0.0 or h <= 0.0:
            continue

        dets.append(DetBox(bbox_xywh=(float(x), float(y), float(w), float(h)), track_id=track_id))
    return dets


def iter_frames(seq_dir: Path, img_ext: str) -> List[Path]:
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    frames = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.name.endswith(img_ext)])
    if not frames:
        raise RuntimeError(f"No frames (*{img_ext}) found under: {seq_dir}")
    return frames


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    r = (track_id * 37) % 200 + 30
    g = (track_id * 17) % 200 + 30
    b = (track_id * 97) % 200 + 30
    return (int(b), int(g), int(r))  # BGR for cv2


def render_seq_to_mp4(
    *,
    seq: str,
    seq_dir: Path,
    labels_seq_dir: Path,
    out_path: Path,
    fps: int,
    img_ext: str,
    limit_frames: Optional[int],
) -> None:
    frames = iter_frames(seq_dir, img_ext=img_ext)
    if limit_frames is not None:
        frames = frames[: int(limit_frames)]

    w, h = read_image_size(frames[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    for img_path in frames:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if img.shape[1] != w or img.shape[0] != h:
            raise RuntimeError(f"Unexpected frame size change in seq={seq}: {img_path} got {img.shape[1]}x{img.shape[0]}, expect {w}x{h}")

        frame_id = int(img_path.stem)  # 000123 -> 123
        label_path = labels_seq_dir / f"{frame_id:06d}.txt"
        dets: List[DetBox] = []
        if label_path.is_file():
            dets = parse_labels_with_ids_txt(label_path, width=w, height=h)

        # Draw
        for d in dets:
            x, y, bw, bh = d.bbox_xywh
            p1 = (int(x), int(y))
            p2 = (int(x + bw), int(y + bh))
            cv2.rectangle(img, p1, p2, _color_for_id(d.track_id), 2)
            cv2.putText(
                img,
                f"{d.track_id}",
                (p1[0], max(15, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        cv2.putText(img, f"seq={seq} frame={frame_id:06d}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        writer.write(img)

    writer.release()


def build_coco_for_seqs(
    *,
    seqs: Sequence[str],
    image_root: Path,
    labels_root: Path,
    img_ext: str,
    starting_image_id: int = 1,
    starting_ann_id: int = 1,
) -> Tuple[Dict, int, int]:
    images: List[Dict] = []
    annotations: List[Dict] = []
    img_id = int(starting_image_id)
    ann_id = int(starting_ann_id)

    for seq in seqs:
        seq_dir = image_root / seq
        labels_seq_dir = labels_root / seq
        if not labels_seq_dir.is_dir():
            raise FileNotFoundError(f"labels seq directory not found: {labels_seq_dir}")

        frames = iter_frames(seq_dir, img_ext=img_ext)
        w, h = read_image_size(frames[0])

        for img_path in frames:
            frame_id = int(img_path.stem)
            # store file_name relative to image_root for portability
            file_name = f"{seq}/{img_path.name}"
            images.append(
                {
                    "id": img_id,
                    "file_name": file_name,
                    "width": w,
                    "height": h,
                }
            )

            label_path = labels_seq_dir / f"{frame_id:06d}.txt"
            if label_path.is_file():
                dets = parse_labels_with_ids_txt(label_path, width=w, height=h)
                for d in dets:
                    x, y, bw, bh = d.bbox_xywh
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 0,
                            "bbox": [x, y, bw, bh],
                            "area": float(bw * bh),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

            img_id += 1

    coco = {
        "info": {"description": "KITTI labels_with_ids exported to COCO detection (single class id=0)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "object", "supercategory": "object"}],
    }
    return coco, img_id, ann_id


def _write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    """
    Ensure directory exists and is empty.
    - If path exists and overwrite=False -> raise.
    - If path exists and overwrite=True -> delete it.
    """
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path} (pass --overwrite to delete)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_roboflow_coco_layout(
    *,
    out_root: Path,
    image_root: Path,
    labels_root: Path,
    train_seqs: Sequence[str],
    valid_seqs: Sequence[str],
    test_seqs: Sequence[str],
    src_img_ext: str,
    overwrite: bool,
    jpeg_quality: int,
) -> None:
    """
    Create:
      out_root/
        train/_annotations.coco.json + images (*.jpg)
        valid/_annotations.coco.json + images (*.jpg)
        test/_annotations.coco.json + images (*.jpg)

    Images are exported as JPG with unique names: <seq>_<frame:06d>.jpg
    The COCO JSON's image.file_name matches these basenames.
    """

    def export_split(split_name: str, seqs: Sequence[str], starting_image_id: int, starting_ann_id: int) -> Tuple[int, int]:
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=False)

        images: List[Dict] = []
        annotations: List[Dict] = []
        img_id = int(starting_image_id)
        ann_id = int(starting_ann_id)

        for seq in seqs:
            seq_dir = image_root / seq
            labels_seq_dir = labels_root / seq
            if not labels_seq_dir.is_dir():
                raise FileNotFoundError(f"labels seq directory not found: {labels_seq_dir}")

            frames = iter_frames(seq_dir, img_ext=src_img_ext)
            w, h = read_image_size(frames[0])

            for img_path in frames:
                frame_id = int(img_path.stem)
                dst_name = f"{seq}_{frame_id:06d}.jpg"
                dst_path = split_dir / dst_name
                if dst_path.exists():
                    raise FileExistsError(f"Image already exists (name collision?): {dst_path}")

                img = cv2.imread(str(img_path))
                if img is None:
                    raise RuntimeError(f"Failed to read image: {img_path}")
                if img.shape[1] != w or img.shape[0] != h:
                    raise RuntimeError(
                        f"Unexpected frame size change in seq={seq}: {img_path} got {img.shape[1]}x{img.shape[0]}, expect {w}x{h}"
                    )
                ok = cv2.imwrite(str(dst_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not ok:
                    raise RuntimeError(f"Failed to write JPG: {dst_path}")

                images.append({"id": img_id, "file_name": dst_name, "width": w, "height": h})

                label_path = labels_seq_dir / f"{frame_id:06d}.txt"
                if label_path.is_file():
                    dets = parse_labels_with_ids_txt(label_path, width=w, height=h)
                    for d in dets:
                        x, y, bw, bh = d.bbox_xywh
                        annotations.append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": 0,
                                "bbox": [x, y, bw, bh],
                                "area": float(bw * bh),
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

                img_id += 1

        coco = {
            "info": {"description": f"KITTI labels_with_ids exported to COCO detection ({split_name}, single class id=0)"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 0, "name": "object", "supercategory": "object"}],
        }
        _write_json(split_dir / "_annotations.coco.json", coco)
        print(f"[OK] Wrote {split_name}/_annotations.coco.json  images={len(images)}  anns={len(annotations)}")
        return img_id, ann_id

    # Ensure root is clean
    _ensure_empty_dir(out_root, overwrite=overwrite)

    # Export in order (IDs unique across splits)
    next_img_id, next_ann_id = export_split("train", train_seqs, starting_image_id=1, starting_ann_id=1)
    next_img_id, next_ann_id = export_split("valid", valid_seqs, starting_image_id=next_img_id, starting_ann_id=next_ann_id)
    export_split("test", test_seqs, starting_image_id=next_img_id, starting_ann_id=next_ann_id)


def main() -> None:
    p = argparse.ArgumentParser("Export KITTI image_02 + labels_with_ids to COCO detection JSON (single class id=0)")
    p.add_argument(
        "--image-root",
        type=str,
        default="/data/sq_2023/refer_kitti/KITTI/training/image_02",
        help="KITTI image_02 root, contains <seq>/<frame>.png",
    )
    p.add_argument(
        "--labels-root",
        type=str,
        default=None,
        help="labels_with_ids root, contains <seq>/<frame>.txt. If omitted, infer from --image-root when it matches .../KITTI/training/image_02",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs/coco_det_refer_kitti",
        help="Output directory. Will write train.json / val.json and optional mp4 under this dir.",
    )
    p.add_argument("--img-ext", type=str, default=".png", help="Image extension under image-root, default .png")
    p.add_argument(
        "--val-seqs",
        type=str,
        nargs="+",
        default=["0005", "0011", "0013", "0019"],
        help="Validation sequence IDs (4-digit), e.g. 0005 0011 0013 0019",
    )
    p.add_argument(
        "--test-seqs",
        type=str,
        nargs="*",
        default=[],
        help="Optional test sequence IDs (4-digit). Default: empty test split.",
    )
    p.add_argument(
        "--out-layout",
        type=str,
        default="flat",
        choices=["flat", "roboflow"],
        help="Output layout. 'flat' writes train.json/val.json. 'roboflow' writes dataset/{train,valid,test}/ with _annotations.coco.json + .jpg images.",
    )
    p.add_argument("--overwrite", action="store_true", help="When --out-layout roboflow: delete out-dir if it exists.")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality when exporting roboflow layout images (1-100).")
    p.add_argument("--export-val-mp4", action="store_true", help="Export bbox-overlay mp4 videos for val seqs")
    p.add_argument("--fps", type=int, default=10, help="MP4 FPS (when --export-val-mp4)")
    p.add_argument("--limit-frames", type=int, default=None, help="Optional max number of frames to render per val seq")
    args = p.parse_args()

    image_root = Path(args.image_root).resolve()
    labels_root: Optional[Path]
    if args.labels_root is not None and str(args.labels_root).strip():
        labels_root = Path(args.labels_root).resolve()
    else:
        labels_root = infer_labels_root(image_root)
    if labels_root is None:
        raise RuntimeError(
            "labels_root is not provided and could not be inferred from image_root.\n"
            "Please pass --labels-root explicitly, e.g. /data/.../KITTI/labels_with_ids/image_02"
        )
    if not labels_root.is_dir():
        raise FileNotFoundError(f"labels_root not found or not a directory: {labels_root}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seqs = list_seqs(image_root)
    val_seqs = [s.strip() for s in args.val_seqs]
    test_seqs = [s.strip() for s in args.test_seqs]

    if len(set(val_seqs) & set(test_seqs)) > 0:
        raise ValueError(f"val_seqs and test_seqs overlap: {sorted(set(val_seqs) & set(test_seqs))}")
    for s in val_seqs:
        if len(s) != 4 or not s.isdigit():
            raise ValueError(f"Invalid val seq id: {s} (expect 4-digit like 0005)")
        if s not in all_seqs:
            raise FileNotFoundError(f"Validation seq not found under image_root: {s} (image_root={image_root})")
        if not (labels_root / s).is_dir():
            raise FileNotFoundError(f"Validation seq labels dir not found: {labels_root / s}")

    for s in test_seqs:
        if len(s) != 4 or not s.isdigit():
            raise ValueError(f"Invalid test seq id: {s} (expect 4-digit like 0005)")
        if s not in all_seqs:
            raise FileNotFoundError(f"Test seq not found under image_root: {s} (image_root={image_root})")
        if not (labels_root / s).is_dir():
            raise FileNotFoundError(f"Test seq labels dir not found: {labels_root / s}")

    train_seqs = [s for s in all_seqs if s not in (set(val_seqs) | set(test_seqs))]
    if not train_seqs:
        raise RuntimeError("Train split would be empty. Check --val-seqs.")

    if str(args.out_layout) == "flat":
        train_json_path = out_dir / "train.json"
        val_json_path = out_dir / "val.json"

        # Build COCO JSONs (keep ids unique across splits; not strictly required, but convenient)
        coco_train, next_img_id, next_ann_id = build_coco_for_seqs(
            seqs=train_seqs,
            image_root=image_root,
            labels_root=labels_root,
            img_ext=str(args.img_ext),
            starting_image_id=1,
            starting_ann_id=1,
        )
        coco_val, _, _ = build_coco_for_seqs(
            seqs=val_seqs,
            image_root=image_root,
            labels_root=labels_root,
            img_ext=str(args.img_ext),
            starting_image_id=next_img_id,
            starting_ann_id=next_ann_id,
        )

        _write_json(train_json_path, coco_train)
        _write_json(val_json_path, coco_val)

        print("[OK] Wrote COCO JSONs:")
        print(f"- train: {train_json_path}  images={len(coco_train['images'])}  anns={len(coco_train['annotations'])}")
        print(f"- val:   {val_json_path}  images={len(coco_val['images'])}  anns={len(coco_val['annotations'])}")
        print("[Info] image_root:", image_root)
        print("[Info] labels_root:", labels_root)
    else:
        q = int(args.jpeg_quality)
        if not (1 <= q <= 100):
            raise ValueError(f"--jpeg-quality must be in [1,100], got {q}")
        export_roboflow_coco_layout(
            out_root=out_dir,
            image_root=image_root,
            labels_root=labels_root,
            train_seqs=train_seqs,
            valid_seqs=val_seqs,
            test_seqs=test_seqs,
            src_img_ext=str(args.img_ext),
            overwrite=bool(args.overwrite),
            jpeg_quality=q,
        )
        print("[OK] Wrote Roboflow-style dataset layout under:", out_dir)

    if bool(args.export_val_mp4):
        mp4_dir = out_dir / "val_mp4"
        for seq in val_seqs:
            out_path = mp4_dir / f"{seq}.mp4"
            render_seq_to_mp4(
                seq=seq,
                seq_dir=image_root / seq,
                labels_seq_dir=labels_root / seq,
                out_path=out_path,
                fps=int(args.fps),
                img_ext=str(args.img_ext),
                limit_frames=args.limit_frames,
            )
            print(f"[OK] Wrote mp4: {out_path}")


if __name__ == "__main__":
    main()

