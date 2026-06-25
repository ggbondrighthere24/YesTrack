import os
import argparse
from collections import Counter
from typing import Counter as CounterType, Tuple, List, Optional, Dict, Any


Det = Tuple[int, int, int, int, int, int]  # (frame, id, x, y, w, h) with xywh truncated to int


def _parse_gt_file(path: str) -> CounterType[Det]:
    """
    Parse a MOT-style gt file with comma-separated columns.
    We only use the first 6 columns: frame,id,x,y,w,h.
    For comparison we round x,y,w,h to nearest int (四舍五入) and then cast to int.
    """
    c: CounterType[Det] = Counter()
    if not os.path.isfile(path):
        return c
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = [p.strip() for p in s.split(",")]
            if len(parts) < 6:
                continue
            try:
                frame = int(float(parts[0]))
                tid = int(float(parts[1]))
                x = int(round(float(parts[2])))
                y = int(round(float(parts[3])))
                w = int(round(float(parts[4])))
                h = int(round(float(parts[5])))
            except Exception:
                continue
            c[(frame, tid, x, y, w, h)] += 1
    return c


def _counter_diff(a: CounterType[Det], b: CounterType[Det]) -> CounterType[Det]:
    """Items in a but not in b (multiset difference, dropping <=0 counts)."""
    out: CounterType[Det] = Counter()
    for k, va in a.items():
        vb = b.get(k, 0)
        if va > vb:
            out[k] = va - vb
    return out


def _format_det(det: Det) -> str:
    f, tid, x, y, w, h = det
    return f"frame={f},id={tid},x={x},y={y},w={w},h={h}"


def main():
    parser = argparse.ArgumentParser("Compare TempRMOT inferred gt.txt vs prepared MOTC gt.txt (truncate xywh to int)")
    parser.add_argument(
        "--temprmot-root",
        type=str,
        default="results_epoch45",
        help="TempRMOT 推理结果根目录（包含 <seq>/<expr>/gt.txt）",
    )
    parser.add_argument(
        "--our-gt-root",
        type=str,
        default="outputs/refer_kitti_motc_gt",
        help="我们提前准备的 GT 根目录（包含 <seq>__<expr>/gt/gt.txt）",
    )
    parser.add_argument(
        "--seqs",
        type=str,
        nargs="*",
        default=None,
        help="只比较指定 seq（例如 0005 0011）。不填则比较 temprmot-root 下所有 seq 目录。",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=5,
        help="每个不一致表达式最多展示的缺失/多余条目数（按 det 展示）。",
    )
    args = parser.parse_args()

    temprmot_root = os.path.abspath(args.temprmot_root)
    our_root = os.path.abspath(args.our_gt_root)

    if args.seqs is None or len(args.seqs) == 0:
        seqs = sorted([d for d in os.listdir(temprmot_root) if os.path.isdir(os.path.join(temprmot_root, d))])
    else:
        seqs = list(args.seqs)

    total_expr = 0
    matched_expr = 0
    missing_our = 0
    missing_temprmot = 0
    mismatch_expr = 0

    # details for summary
    mismatches: List[Dict[str, Any]] = []

    for seq in seqs:
        seq_dir = os.path.join(temprmot_root, seq)
        if not os.path.isdir(seq_dir):
            continue
        exprs = sorted([d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))])
        for expr in exprs:
            total_expr += 1
            tm_gt = os.path.join(seq_dir, expr, "gt.txt")
            our_gt = os.path.join(our_root, f"{seq}__{expr}", "gt", "gt.txt")

            if not os.path.isfile(tm_gt):
                missing_temprmot += 1
                mismatches.append(
                    {"seq": seq, "expr": expr, "reason": "missing_temprmot_gt", "temprmot_gt": tm_gt, "our_gt": our_gt}
                )
                continue
            if not os.path.isfile(our_gt):
                missing_our += 1
                mismatches.append(
                    {"seq": seq, "expr": expr, "reason": "missing_our_gt", "temprmot_gt": tm_gt, "our_gt": our_gt}
                )
                continue

            tm_c = _parse_gt_file(tm_gt)
            our_c = _parse_gt_file(our_gt)

            extra_in_tm = _counter_diff(tm_c, our_c)
            missing_in_tm = _counter_diff(our_c, tm_c)

            if not extra_in_tm and not missing_in_tm:
                matched_expr += 1
                continue

            mismatch_expr += 1
            item = {
                "seq": seq,
                "expr": expr,
                "reason": "content_mismatch",
                "temprmot_gt": tm_gt,
                "our_gt": our_gt,
                "temprmot_lines": int(sum(tm_c.values())),
                "our_lines": int(sum(our_c.values())),
                "extra_in_temprmot": int(sum(extra_in_tm.values())),
                "missing_in_temprmot": int(sum(missing_in_tm.values())),
                "show_extra": [],
                "show_missing": [],
            }

            # show a few examples
            for det, cnt in extra_in_tm.most_common(args.max_show):
                item["show_extra"].append({"det": _format_det(det), "count": int(cnt)})
            for det, cnt in missing_in_tm.most_common(args.max_show):
                item["show_missing"].append({"det": _format_det(det), "count": int(cnt)})

            mismatches.append(item)

    print("=== GT Compare Summary (round xywh to int) ===")
    print(f"TempRMOT root: {temprmot_root}")
    print(f"Our GT root:  {our_root}")
    print(f"Total expressions checked: {total_expr}")
    print(f"Matched expressions:       {matched_expr}")
    print(f"Mismatched expressions:    {mismatch_expr}")
    print(f"Missing our gt.txt:        {missing_our}")
    print(f"Missing temprmot gt.txt:   {missing_temprmot}")

    if mismatch_expr > 0 or missing_our > 0 or missing_temprmot > 0:
        print("\n=== First mismatches (up to 20) ===")
        for item in mismatches[:20]:
            seq = item["seq"]
            expr = item["expr"]
            reason = item["reason"]
            print(f"- {seq}/{expr}: {reason}")
            if reason == "content_mismatch":
                print(f"  temprmot_lines={item['temprmot_lines']} our_lines={item['our_lines']}")
                print(f"  extra_in_temprmot={item['extra_in_temprmot']} missing_in_temprmot={item['missing_in_temprmot']}")
                if item["show_extra"]:
                    print("  extra examples:")
                    for x in item["show_extra"]:
                        print(f"    {x['det']} x{x['count']}")
                if item["show_missing"]:
                    print("  missing examples:")
                    for x in item["show_missing"]:
                        print(f"    {x['det']} x{x['count']}")
            else:
                print(f"  temprmot_gt={item['temprmot_gt']}")
                print(f"  our_gt={item['our_gt']}")


if __name__ == "__main__":
    main()


