import os
import argparse
import glob
import json


TRAIN_SEQS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 17, 18, 20]
VAL_SEQS = [5, 11, 13, 19]


def list_seq_names(kitti_root: str):
    image_root = os.path.join(kitti_root, "image_02")
    return sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])


def count_frames(kitti_root: str, seq: str) -> int:
    seq_dir = os.path.join(kitti_root, "image_02", seq)
    return len([f for f in os.listdir(seq_dir) if f.endswith(".png")])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_seqinfo_ini(out_seq_dir: str, seq_name: str, width: int, height: int, length: int):
    content = [
        "[Sequence]",
        f"name={seq_name}",
        f"imDir=img1",
        f"frameRate=30",
        f"seqLength={length}",
        f"imWidth={width}",
        f"imHeight={height}",
        f"imExt=.png",
        "",
    ]
    with open(os.path.join(out_seq_dir, "seqinfo.ini"), "w") as f:
        f.write("\n".join(content))


def read_labels_with_ids_boxes(labels_with_ids_root: str, seq: str, width: int, height: int):
    """
    Read per-frame labels from labels_with_ids: each frame file has lines: cls id x y w h (normalized [0,1])
    Returns: dict[int frame0] -> list of (track_id, x, y, w, h) in pixel xywh
    """
    seq_dir = os.path.join(labels_with_ids_root, seq)
    per_frame = dict()
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"labels_with_ids seq dir not found: {seq_dir}")
    # enumerate by existing frame files
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.endswith('.txt'):
            continue
        frame0 = int(os.path.splitext(fname)[0])
        file_path = os.path.join(seq_dir, fname)
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                try:
                    track_id = int(float(parts[1]))
                    nx = float(parts[2]); ny = float(parts[3]); nw = float(parts[4]); nh = float(parts[5])
                except Exception:
                    continue
                x = nx * width
                y = ny * height
                w = max(0.0, nw * width)
                h = max(0.0, nh * height)
                if frame0 not in per_frame:
                    per_frame[frame0] = []
                per_frame[frame0].append((track_id, x, y, w, h))
    return per_frame


def load_expressions(expression_root: str, seq: str):
    seq_dir = os.path.join(expression_root, seq)
    json_files = sorted(glob.glob(os.path.join(seq_dir, "*.json")))
    expressions = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            sentence = data.get("sentence", os.path.basename(jf))
            labels = data.get("label", {})
            expressions.append((os.path.basename(jf), sentence, labels))
        except Exception:
            continue
    return expressions


def write_seqmap(seq_names: list, out_seqmap_file: str):
    ensure_dir(os.path.dirname(out_seqmap_file))
    with open(out_seqmap_file, "w") as f:
        f.write("name\n")
        for s in seq_names:
            f.write(f"{s}\n")


def main():
    parser = argparse.ArgumentParser("Prepare MOTChallenge-style GT and seqmap from Refer-KITTI (per expression) using labels_with_ids")
    parser.add_argument("--kitti-root", type=str, default="/data/sq_2023/refer_kitti_v2/KITTI/training", help="Path to KITTI/training")
    parser.add_argument("--labels-with-ids-root", type=str, default="/data/sq_2023/refer_kitti_v2/KITTI/labels_with_ids/image_02", help="Path to KITTI/labels_with_ids or KITTI/labels_with_ids/image_02")
    parser.add_argument("--expression-root", type=str, default="/data/sq_2023/refer_kitti_v2/expression", help="Path to expression root")
    parser.add_argument("--out-gt-folder", type=str, default="outputs/refer_kitti_motc_gt_v2", help="Output GT folder for MOTChallenge layout")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="val")
    parser.add_argument("--seqmap-train", type=str, default=None, help="Output seqmap file path for train split")
    parser.add_argument("--seqmap-val", type=str, default='outputs/refer_kitti_motc_gt_v2/seqmaps/val.txt', help="Output seqmap file path for val split")
    parser.add_argument("--width", type=int, default=1242)
    parser.add_argument("--height", type=int, default=375)
    parser.add_argument(
        "--label-key-mode",
        type=str,
        choices=["strict0", "strict1", "fallback0to1", "fallback1to0", "auto"],
        default="strict0",
        help=(
            "expression json 中 label 的帧 key 解释方式："
            "strict0=严格按 0-based(key=str(frame0))；"
            "strict1=严格按 1-based(key=str(frame0+1))；"
            "fallback0to1=先 0-based 再 fallback 到 1-based（旧行为，可能造成错位）；"
            "fallback1to0=先 1-based 再 fallback 到 0-based；"
            "auto=基于是否存在 key '0'/'1' 自动判断。"
        ),
    )
    args = parser.parse_args()

    all_seqs = list_seq_names(args.kitti_root)
    train_names = [f"{i:04d}" for i in TRAIN_SEQS if f"{i:04d}" in all_seqs]
    val_names = [f"{i:04d}" for i in VAL_SEQS if f"{i:04d}" in all_seqs]

    targets = []
    if args.split in ["train", "both"]:
        targets.append(("train", train_names, args.seqmap_train))
    if args.split in ["val", "both"]:
        targets.append(("val", val_names, args.seqmap_val))

    for split, seq_names, seqmap_path in targets:
        out_seq_names = []
        for seq in seq_names:
            length = count_frames(args.kitti_root, seq)
            # Load labels_with_ids boxes for this seq (as authoritative GT boxes)
            per_frame_boxes = read_labels_with_ids_boxes(args.labels_with_ids_root, seq, args.width, args.height)
            # Load all expressions under this seq
            expressions = load_expressions(args.expression_root, seq)
            for expr_file, sentence, labels in expressions:
                # Compose a unique sequence name per expression
                expr_name = os.path.splitext(expr_file)[0]
                seq_name = f"{seq}__{expr_name}"
                out_seq_names.append(seq_name)
                out_seq_dir = os.path.join(args.out_gt_folder, seq_name)
                ensure_dir(out_seq_dir)
                write_seqinfo_ini(out_seq_dir, seq_name, args.width, args.height, length)
                # Write GT by filtering per expression IDs per frame
                gt_dir = os.path.join(out_seq_dir, "gt")
                ensure_dir(gt_dir)
                out_file = os.path.join(gt_dir, "gt.txt")
                lines_out = []

                # Decide how to interpret json label keys for this expression
                mode = args.label_key_mode
                if mode == "auto":
                    # Heuristic: if '0' exists -> 0-based, elif '1' exists -> 1-based, else default to 0-based
                    if isinstance(labels, dict) and "0" in labels:
                        mode = "strict0"
                    elif isinstance(labels, dict) and "1" in labels:
                        mode = "strict1"
                    else:
                        mode = "strict0"

                for frame0 in range(length):
                    key0 = str(frame0)
                    key1 = str(frame0 + 1)
                    if mode == "strict0":
                        ref_ids = labels.get(key0, [])
                    elif mode == "strict1":
                        ref_ids = labels.get(key1, [])
                    elif mode == "fallback0to1":
                        ref_ids = labels.get(key0, labels.get(key1, []))
                    elif mode == "fallback1to0":
                        ref_ids = labels.get(key1, labels.get(key0, []))
                    else:
                        ref_ids = labels.get(key0, [])
                    if not isinstance(ref_ids, list) or len(ref_ids) == 0:
                        continue
                    ref_ids = set(int(i) for i in ref_ids)
                    if frame0 not in per_frame_boxes:
                        continue
                    for track_id, x1, y1, w, h in per_frame_boxes[frame0]:
                        if int(track_id) in ref_ids:
                            frame1 = frame0 + 1
                            zero_marked = 1
                            class_id = 1  # pedestrian
                            lines_out.append(
                                f"{frame1},{track_id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{zero_marked},{class_id}\n"
                            )
                with open(out_file, "w") as f:
                    f.writelines(lines_out)
        if seqmap_path is not None:
            write_seqmap(out_seq_names, seqmap_path)
            print(f"Wrote seqmap for {split} to {seqmap_path}")
        print(f"Prepared {len(out_seq_names)} refer sequences for split '{split}' under {args.out_gt_folder}")


if __name__ == "__main__":
    main()


