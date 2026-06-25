import os
import sys
import argparse
import csv
import shutil
import subprocess


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _rewrite_with_frame_offset(src_file: str, dst_file: str, offset: int, min_score: float, min_area: float):
    """读取 src_file，按给定 offset 整体平移帧号，过滤得分阈值 min_score 与面积阈值 min_area，写到 dst_file。"""
    lines = []
    with open(src_file, "r") as f:
        for raw in f:
            s = raw.strip()
            if len(s) == 0:
                continue
            parts = s.split(",")
            if len(parts) < 7:
                continue
            # 过滤低于阈值的结果
            try:
                score = float(parts[6])
            except Exception:
                continue
            if score < float(min_score):
                continue
            # 面积过滤（与 TempRMOT 一致：按 w*h，默认阈值 100；min_area<=0 则关闭）
            if float(min_area) > 0:
                try:
                    w = float(parts[4])
                    h = float(parts[5])
                except Exception:
                    continue
                if w * h < float(min_area):
                    continue
            lines.append(parts)
    with open(dst_file, "w") as f:
        for parts in lines:
            try:
                frame_idx = int(float(parts[0])) + offset
            except Exception:
                continue
            parts_out = [str(frame_idx)] + [p.strip() for p in parts[1:]]
            f.write(",".join(parts_out) + "\n")


def collect_refer_results(
    refer_root: str,
    out_tracker_data_dir: str,
    frame_offset: int,
    min_score: float,
    min_area: float,
    structure: str = "flat",
    seq_names_to_generate: list | None = None,
):
    """
    收集 refer 结果文件为 MOTChallenge 期望的 tracker/default/data 结构。
    支持两种输入结构：
      1) flat（默认）:
         refer_root/
           <seqid>/
             <expr_name>.txt  (内容为 per-frame 行, 逗号分隔: frame,id,x,y,w,h,score)
      2) nested（用于 runs/.../train/eval_with_mot_result 这种）:
         refer_root/
           <seqid>/
             <expr_name>/
               predict_with_conf.txt  (或任一 .txt，将优先选择 predict_with_conf.txt)
      3) bytetrack（复用同一份跟踪结果到所有 expression）:
         refer_root/
           <seqid>.txt  (内容为 per-frame 行, 逗号分隔: frame,id,x,y,w,h,score,...)
         需要提供 seq_names_to_generate（通常来自 GT seqmap，形如 "<seqid>__<expr_name>"）

    输出命名规则与 GT 保持一致: "<seqid>__<expr_name>.txt"
    """
    ensure_dir(out_tracker_data_dir)
    num_copied = 0
    generated_seq_names = []  # names without .txt, e.g., 0005__black-cars-in-the-left
    for seqid in sorted(os.listdir(refer_root)):
        seq_dir = os.path.join(refer_root, seqid)
        if not os.path.isdir(seq_dir):
            continue
        if structure == "flat":
            for fname in sorted(os.listdir(seq_dir)):
                if not fname.endswith(".txt"):
                    continue
                expr_name = os.path.splitext(fname)[0]
                # 将下划线转换为连字符以匹配 GT/seqmap 命名
                expr_name_hyphen = expr_name.replace("_", "-")
                out_base = f"{seqid}__{expr_name_hyphen}"
                out_name = f"{out_base}.txt"
                src_file = os.path.join(seq_dir, fname)
                dst_file = os.path.join(out_tracker_data_dir, out_name)
                _rewrite_with_frame_offset(src_file, dst_file, frame_offset, min_score, min_area)
                num_copied += 1
                generated_seq_names.append(out_base)
        elif structure == "nested":
            # 处理结构: <seq>/<expr>/predict_with_conf.txt
            for expr_dir_name in sorted(os.listdir(seq_dir)):
                expr_dir = os.path.join(seq_dir, expr_dir_name)
                if not os.path.isdir(expr_dir):
                    continue
                # 优先 predict_with_conf.txt；否则回退任意 .txt
                preferred = os.path.join(expr_dir, "predict_with_conf.txt")
                if os.path.isfile(preferred):
                    src_file = preferred
                else:
                    txts = [os.path.join(expr_dir, f) for f in sorted(os.listdir(expr_dir)) if f.endswith(".txt")]
                    if len(txts) == 0:
                        continue
                    src_file = txts[0]
                expr_name = expr_dir_name
                expr_name_hyphen = expr_name.replace("_", "-")
                out_base = f"{seqid}__{expr_name_hyphen}"
                out_name = f"{out_base}.txt"
                dst_file = os.path.join(out_tracker_data_dir, out_name)
                _rewrite_with_frame_offset(src_file, dst_file, frame_offset, min_score, min_area)
                num_copied += 1
                generated_seq_names.append(out_base)
        elif structure == "bytetrack":
            # 该分支在 main() 中统一处理（基于 seqmap 生成），这里不走逐目录扫描
            continue
        else:
            raise ValueError(f"Unknown structure '{structure}', expected 'flat' or 'nested'.")
    # bytetrack: 基于 seqmap 生成（每个 seq 的所有 expression 复用同一个 {seq}.txt）
    if structure == "bytetrack":
        if not isinstance(seq_names_to_generate, list) or len(seq_names_to_generate) == 0:
            raise ValueError("structure=bytetrack 时必须提供非空 seq_names_to_generate（通常来自 --seqmap-file）")
        for base in seq_names_to_generate:
            s = str(base).strip()
            if len(s) == 0 or "__" not in s:
                continue
            seqid, _expr = s.split("__", 1)
            src_file = os.path.join(refer_root, f"{seqid}.txt")
            if not os.path.isfile(src_file):
                # 跳过缺失 seq 的结果（后续将与 seqmap 做交集）
                continue
            dst_file = os.path.join(out_tracker_data_dir, f"{s}.txt")
            _rewrite_with_frame_offset(src_file, dst_file, frame_offset, min_score, min_area)
            num_copied += 1
            generated_seq_names.append(s)
    return num_copied, sorted(set(generated_seq_names))


def read_seqmap(seqmap_file: str):
    seq_names = []
    with open(seqmap_file, "r", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            value = ",".join(row).strip()
            if i == 0:
                assert value == "name", "Seqmap file should start with header 'name'"
                continue
            if value:
                seq_names.append(value)
    return seq_names


def write_seqmap(seq_names: list, out_seqmap_file: str):
    ensure_dir(os.path.dirname(out_seqmap_file))
    # IMPORTANT:
    # TrackEval reads seqmap via `csv.reader`, so commas are treated as delimiters.
    # Using `csv.writer` ensures sequence names containing commas are properly quoted.
    with open(out_seqmap_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for s in seq_names:
            writer.writerow([s])


def read_summary_metrics(summary_path: str):
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    with open(summary_path, "r") as f:
        names = f.readline().strip().split(" ")
        values = f.readline().strip().split(" ")
    return {k: float(v) for k, v in zip(names, values)}


def main():
    parser = argparse.ArgumentParser("Evaluate Refer-KITTI MOT metrics using TrackEval (MOTChallenge style)")
    parser.add_argument("--refer-root", type=str, default="track_result/noise_eval_new/p1001", help="refer 结果根目录 (含 <seq>/<expr>.txt)")
    parser.add_argument(
        "--refer-structure",
        type=str,
        choices=["flat", "nested", "bytetrack"],
        default="nested",
        help=(
            "refer 输入目录结构：flat / nested / bytetrack。"
            "nested 对应 <seq>/<expr>/predict_with_conf.txt；"
            "bytetrack 对应 <seq>.txt（同一份跟踪结果复用到该 seq 的所有 expression）。"
        ),
    )
    parser.add_argument("--gt-folder", type=str, default="outputs/refer_kitti_motc_gt", help="MOTC 风格 GT 根目录 (per-expression)")
    parser.add_argument("--seqmap-file", type=str, default="outputs/refer_kitti_motc_gt/seqmaps/val.txt", help="per-expression seqmap 文件")
    parser.add_argument("--out-dir", type=str, default='track_result/noise_eval_new/p1001', help="评估输出目录")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--offset", type=int, default=0, help="预测结果帧号整体偏移量，常见地设置为1以将0基帧改为1基帧。")
    parser.add_argument("--min-area", type=float, default=20.0, help="预测框面积过滤阈值（按 w*h）。与 TempRMOT 一致默认 100；设为 0 可关闭。")
    parser.add_argument("--tracker-sub-folder", type=str, default="data")
    parser.add_argument("--classes", type=str, nargs="+", default=["pedestrian"], help="评估类别列表")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1,0.15,0.2,0.25,0.3,0.35], help="评估阈值列表，对 refer 分数进行过滤（bytetrack 模式会忽略该参数）")
    parser.add_argument(
        "--allow-subset-seqmap",
        action="store_true",
        help=(
            "允许只评估 GT seqmap 与结果集合的交集（不完全一致时不报错，仅 warning）。"
            "默认严格：若两者不一致将直接报错，避免静默漏评估/命名不一致。"
        ),
    )
    args = parser.parse_args()

    def _thr_name(v: float) -> str:
        # 0.40 -> t0p40
        s = f"{v:.2f}"
        return "t" + s.replace(".", "p")

    show_keys = ["HOTA", "DetA", "AssA", "DetPr", "DetRe", "AssPr", "AssRe", "MOTA", "IDF1"]
    all_metrics = {}

    # bytetrack：不使用阈值过滤（阈值是 refer 分数概念；bytetrack 第 7 列通常是 detector/track 分数）
    if str(args.refer_structure) == "bytetrack":
        thresholds = [None]  # 单次评估
    else:
        thresholds = list(args.thresholds)

    for thr in thresholds:
        if thr is None:
            tag = "bytetrack"
            tracker_name = "default_bytetrack"
            min_score = float("-inf")  # 禁用分数阈值过滤
        else:
            tag = _thr_name(float(thr))
            tracker_name = f"default_{tag}"
            min_score = float(thr)
        tracker_dir = os.path.join(args.out_dir, "tracker", tracker_name)
        tracker_data_dir = os.path.join(tracker_dir, "data")
        ensure_dir(tracker_data_dir)

        # 读取目标 seqmap（用于 bytetrack 生成与最终取交集）
        target_seq_names = read_seqmap(args.seqmap_file)

        # 收集/复制并按阈值过滤
        copied, available_seq_names = collect_refer_results(
            args.refer_root,
            tracker_data_dir,
            args.offset,
            min_score,
            args.min_area,
            args.refer_structure,
            seq_names_to_generate=(target_seq_names if str(args.refer_structure) == "bytetrack" else None),
        )
        print(f"[{tag}] Collected {copied} result files into {tracker_data_dir}")

        # 读取目标 seqmap，并与可用结果取交集，生成临时 seqmap
        target_set = set(target_seq_names)
        available_set = set(available_seq_names)
        eval_seq_names = sorted(target_set.intersection(available_set))
        if len(eval_seq_names) == 0:
            raise RuntimeError(f"[{tag}] No overlapping sequences between refer results and GT seqmap.")
        # 默认严格：必须完全一致，否则报错，避免静默漏评估/命名不一致（例如逗号、下划线/连字符差异）
        if len(eval_seq_names) != len(target_seq_names):
            missing = sorted(list(target_set - available_set))
            extra = sorted(list(available_set - target_set))
            if bool(getattr(args, "allow_subset_seqmap", False)):
                print(
                    f"[{tag}] Warning: Using subset seqmap with {len(eval_seq_names)}/{len(target_seq_names)} "
                    f"sequences present in refer results. Missing={len(missing)}, Extra={len(extra)}"
                )
            else:
                raise RuntimeError(
                    f"[{tag}] Seqmap mismatch (STRICT MODE): "
                    f"GT seqmap has {len(target_set)} seqs, results provide {len(available_set)} seqs, "
                    f"intersection={len(eval_seq_names)}. "
                    f"Missing_in_results(example)={missing[:10]} | Extra_in_results(example)={extra[:10]}. "
                    f"Hint: check expression naming (commas), and consider running tools/restore_commas_in_refer.py "
                    f"or re-generating results/seqmap with consistent normalization. "
                    f"To override: pass --allow-subset-seqmap."
                )
        tmp_seqmap = os.path.join(args.out_dir, "seqmaps", f"refer_eval_seqmap_{tag}.txt")
        write_seqmap(eval_seq_names, tmp_seqmap)

        # 调用 TrackEval（与 submit_and_evaluate 中 ReferKittiMOT 分支一致）
        cmd = [
            sys.executable,
            "TrackEval/scripts/run_mot_challenge.py",
            "--SPLIT_TO_EVAL", args.split,
            "--METRICS", "HOTA", "CLEAR", "Identity",
            "--GT_FOLDER", args.gt_folder,
            "--SKIP_SPLIT_FOL", "True",
            "--TRACKERS_FOLDER", os.path.join(args.out_dir, "tracker"),
            "--TRACKER_SUB_FOLDER", args.tracker_sub_folder,
            "--TRACKERS_TO_EVAL", tracker_name,
            "--SEQMAP_FILE", tmp_seqmap,
            "--USE_PARALLEL","True", 
            "--NUM_PARALLEL_CORES","16", 
            "--PLOT_CURVES", "False",
            "--CLASSES_TO_EVAL",
        ] + args.classes

        print("Running:", " ".join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"TrackEval failed for tag {tag}")

        # 读取并打印指标
        summary_path = os.path.join(args.out_dir, "tracker", tracker_name, "pedestrian_summary.txt")
        if not os.path.exists(summary_path):
            summary_path = os.path.join(args.out_dir, "tracker", "pedestrian_summary.txt")
        metrics = read_summary_metrics(summary_path)
        all_metrics[tag] = metrics
        if thr is None:
            print(f"=== Refer-KITTI MOT Metrics (bytetrack, no threshold) ===")
        else:
            print(f"=== Refer-KITTI MOT Metrics @ {float(thr):.2f} ===")
        for k in show_keys:
            if k in metrics:
                print(f"{k}: {metrics[k]:.4f}")

    # 汇总不同阈值结果的关键指标
    if len(all_metrics) > 1:
        print("\n=== Summary across thresholds ===")
        # key 可能是 thr(float) 或 tag(str)，此处统一按字符串打印
        for key in sorted(all_metrics.keys(), key=lambda x: str(x)):
            metrics = all_metrics[key]
            line = [f"tag={key}"] + [
                f"{k}={metrics[k]:.4f}" for k in show_keys if k in metrics
            ]
            print(" ".join(line))


if __name__ == "__main__":
    main()

