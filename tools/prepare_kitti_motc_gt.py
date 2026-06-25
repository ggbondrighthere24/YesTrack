import os
import argparse


TRAIN_SEQS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20]
VAL_SEQS = [5, 11, 13]


def list_seq_names(kitti_root: str):
    image_root = os.path.join(kitti_root, "image_02")
    return sorted([d for d in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, d))])


def count_frames(kitti_root: str, seq: str) -> int:
    seq_dir = os.path.join(kitti_root, "image_02", seq)
    return len([f for f in os.listdir(seq_dir) if f.endswith(".png")])


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_seqinfo_ini(out_seq_dir: str, seq: str, width: int, height: int, length: int):
    content = [
        "[Sequence]",
        f"name={seq}",
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


def convert_kitti_label_to_motc_gt(kitti_root: str, seq: str, out_seq_dir: str):
    """
    Convert KITTI label_02 to MOTChallenge gt.txt (single-class pedestrian).
    MOTC GT format columns used by TrackEval mot_challenge_2d_box:
      frame(1-based), id, x, y, w, h, zero_marked, class_id
    We set zero_marked=1 and class_id=1 (pedestrian) for all objects, skip DontCare.
    """
    label_file = os.path.join(kitti_root, "label_02", f"{seq}.txt")
    gt_dir = os.path.join(out_seq_dir, "gt")
    ensure_dir(gt_dir)
    out_file = os.path.join(gt_dir, "gt.txt")
    lines_out = []
    if os.path.isfile(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                frame0 = int(parts[0])  # 0-based in KITTI
                track_id = int(parts[1])
                obj_type = parts[2]
                if obj_type == "DontCare":
                    continue
                x1 = float(parts[6]); y1 = float(parts[7]); x2 = float(parts[8]); y2 = float(parts[9])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                frame1 = frame0 + 1
                zero_marked = 1
                class_id = 1  # pedestrian
                lines_out.append(f"{frame1},{track_id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{zero_marked},{class_id}\n")
    with open(out_file, "w") as f:
        f.writelines(lines_out)


def write_seqmap(seq_names: list, out_seqmap_file: str):
    ensure_dir(os.path.dirname(out_seqmap_file))
    with open(out_seqmap_file, "w") as f:
        f.write("name\n")
        for s in seq_names:
            f.write(f"{s}\n")


def main():
    parser = argparse.ArgumentParser("Prepare MOTChallenge-style GT and seqmap from KITTI Tracking")
    parser.add_argument("--kitti-root", type=str, required=True, help="Path to KITTI/training")
    parser.add_argument("--out-gt-folder", type=str, required=True, help="Output GT folder for MOTChallenge layout")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="val")
    parser.add_argument("--seqmap-train", type=str, default=None, help="Output seqmap file path for train split")
    parser.add_argument("--seqmap-val", type=str, default=None, help="Output seqmap file path for val split")
    parser.add_argument("--width", type=int, default=1242)
    parser.add_argument("--height", type=int, default=375)
    args = parser.parse_args()

    all_seqs = list_seq_names(args.kitti_root)
    # Filter per split
    train_names = [f"{i:04d}" for i in TRAIN_SEQS if f"{i:04d}" in all_seqs]
    val_names = [f"{i:04d}" for i in VAL_SEQS if f"{i:04d}" in all_seqs]

    targets = []
    if args.split in ["train", "both"]:
        targets.append(("train", train_names, args.seqmap_train))
    if args.split in ["val", "both"]:
        targets.append(("val", val_names, args.seqmap_val))

    for split, seq_names, seqmap_path in targets:
        for seq in seq_names:
            out_seq_dir = os.path.join(args.out_gt_folder, seq)
            ensure_dir(out_seq_dir)
            length = count_frames(args.kitti_root, seq)
            write_seqinfo_ini(out_seq_dir, seq, args.width, args.height, length)
            convert_kitti_label_to_motc_gt(args.kitti_root, seq, out_seq_dir)
        if seqmap_path is not None:
            write_seqmap(seq_names, seqmap_path)
            print(f"Wrote seqmap for {split} to {seqmap_path}")
        print(f"Prepared {len(seq_names)} sequences for split '{split}' under {args.out_gt_folder}")


if __name__ == "__main__":
    main()


