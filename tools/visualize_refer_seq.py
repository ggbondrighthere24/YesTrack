import os
import argparse
from typing import Dict, List, Tuple

import cv2


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_gt_motc_file(gt_path: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """
    Parse Refer-KITTI MOTC gt file.
    Expected columns per line: frame(1-based), id, x, y, w, h, mark, class
    Return: {frame1: [(id, x, y, w, h), ...], ...}
    """
    per_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue
            try:
                frame1 = int(parts[0])
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except Exception:
                continue
            per_frame.setdefault(frame1, []).append((track_id, x, y, w, h))
    return per_frame


def parse_pred_file(pred_path: str) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    """
    Parse prediction file written by submit_and_evaluate for ReferKittiMOT.
    Expected columns per line: frame(1-based), id, x, y, w, h, score
    Return: {frame1: [(id, x, y, w, h, score), ...], ...}
    """
    per_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
    with open(pred_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                frame1 = int(parts[0])
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                score = float(parts[6]) if len(parts) >= 7 else 1.0
            except Exception:
                continue
            per_frame.setdefault(frame1, []).append((track_id, x, y, w, h, score))
    return per_frame


def draw_boxes(
    image,
    boxes_gt: List[Tuple[int, float, float, float, float]] | None,
    boxes_pred: List[Tuple[int, float, float, float, float, float]] | None,
) -> None:
    """
    Draw GT (green) and Pred (red) boxes on the image in-place.
    - boxes_gt: list of (id, x, y, w, h)
    - boxes_pred: list of (id, x, y, w, h, score)
    """
    if boxes_gt is not None:
        for track_id, x, y, w, h in boxes_gt:
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
            label = f"G{track_id}"
            cv2.putText(image, label, (p1[0], max(0, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if boxes_pred is not None:
        for track_id, x, y, w, h, score in boxes_pred:
            p1 = (int(x), int(y))
            p2 = (int(x + w), int(y + h))
            cv2.rectangle(image, p1, p2, (0, 0, 255), 2)
            label = f"P{track_id} {score:.2f}"
            cv2.putText(image, label, (p1[0], min(image.shape[0] - 2, p1[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def infer_seq_name(pred_path: str) -> str:
    base = os.path.basename(pred_path)
    name, _ = os.path.splitext(base)
    # Try to infer seq id from parent directory, e.g., .../refer/0005/black_cars_in_the_left.txt
    parent = os.path.basename(os.path.dirname(pred_path))
    seq_is_4digit = len(parent) == 4 and parent.isdigit()

    if seq_is_4digit:
        seq = parent
        # If filename starts with seq + delimiter, strip it; then normalize underscores to dashes
        prefixes = [f"{seq}_", f"{seq}-", f"{seq}__"]
        text = name
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):]
                break
        text = text.replace('_', '-')
        return f"{seq}__{text}"

    # Fallback: just use the filename stem as is
    return name


def main():
    parser = argparse.ArgumentParser("Visualize Refer-KITTI sequence with GT and predictions")
    parser.add_argument("--gt", type=str, default="outputs/refer_kitti_motc_gt/0005__cars-in-left/gt/gt.txt", help="Path to gt.txt (MOTC format)")
    parser.add_argument("--pred", type=str, default='track_result/v1best_track_result/0005/cars-in-left/predict.txt', help="Path to prediction txt file")
    parser.add_argument("--image-root", type=str, default="/data/sq_2023/refer_kitti/KITTI/training/image_02", help="Path to KITTI image_02 root (contains <seq>/<frame>.png)")
    parser.add_argument("--output-root", type=str, default="tempvis", help="Path to output root where 'vis/<seq_expr>/' will be created")
    parser.add_argument("--seq-name", type=str, default=None, help="Explicit seq name like '0005__black-cars-in-the-left' (optional)")
    parser.add_argument("--img-ext", type=str, default=".png", help="Image extension, default .png")
    parser.add_argument("--start-frame", type=int, default=None, help="1-based start frame (inclusive)")
    parser.add_argument("--end-frame", type=int, default=None, help="1-based end frame (inclusive)")
    parser.add_argument("--pred-frame-offset", type=int, default=0, help="Optional offset to add to prediction frame indices (e.g., +1 if preds start at 0)")
    args = parser.parse_args()

    seq_expr = args.seq_name or infer_seq_name(args.pred)
    base_seq = seq_expr.split("__")[0]

    gt_by_frame = parse_gt_motc_file(args.gt)
    pred_by_frame = parse_pred_file(args.pred)

    # Align prediction frames to 1-based if needed
    pred_offset = 0
    if args.pred_frame_offset is not None:
        pred_offset = args.pred_frame_offset
    else:
        if len(pred_by_frame) > 0:
            min_pred_frame = min(pred_by_frame.keys())
            # If predictions start at 0, shift to 1-based by default
            if min_pred_frame == 0:
                pred_offset = 1
    if pred_offset != 0:
        aligned_pred_by_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
        for f_idx, items in pred_by_frame.items():
            aligned_pred_by_frame[f_idx + pred_offset] = items
        pred_by_frame = aligned_pred_by_frame

    frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    # print(frames)
    if args.start_frame is not None:
        frames = [f for f in frames if f >= args.start_frame]
    if args.end_frame is not None:
        frames = [f for f in frames if f <= args.end_frame]

    out_dir = os.path.join(args.output_root, "vis", seq_expr)
    ensure_dir(out_dir)

    for frame1 in frames:
        # Convert to 0-based image index
        frame0 = frame1 - 1
        img_path = os.path.join(args.image_root,'0005', f"{frame0:06d}{args.img_ext}")
        if not os.path.isfile(img_path):
            print(f"Image not found: {img_path}")
            # Skip silently if image missing
            continue
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Image is None: {img_path}")
            continue
        
        boxes_gt = gt_by_frame.get(frame1, None)
        boxes_pred = pred_by_frame.get(frame1, None)
        draw_boxes(image, boxes_gt, boxes_pred)

        out_path = os.path.join(out_dir, f"{frame0:06d}.jpg")
        cv2.imwrite(out_path, image)

    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()


