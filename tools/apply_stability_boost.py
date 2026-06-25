#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 v2best59_original 这类结果目录做“稳定性加分”后处理。

输入：一个已经生成好的结果根目录（例如 track_result/v2best59_original）
输出：一个新的结果根目录（例如 track_result/v2best59_original_stable）

规则与 refer_llm/llm_eval_from_mot.py 一致（在线、仅依赖历史帧）：
  - 对每个 tid 维护一个 deque(maxlen=window)，存放“历史帧最终输出的 confidence”
  - 当处理当前 detection 时：
      若 deque 长度 >= window 且其中所有值都 >= thresh，则对当前 p 做 p=min(1, p+boost)
      然后将（加分后的）p append 进 deque

目录策略：
  - 递归复制整个输入目录到输出目录
  - 仅对文件名为 predict_with_conf.txt 的文件做重写；其它文件原样复制
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class MotLine:
    frame: int
    tid: int
    x: int
    y: int
    w: int
    h: int
    conf: float
    raw_tail: str = ""  # 保留行尾多余字段（若有）


def _parse_mot_line(line: str) -> Optional[MotLine]:
    s = (line or "").strip()
    if not s:
        return None
    # 支持逗号或空白分隔
    parts = []
    cur = []
    for ch in s:
        if ch in [",", " ", "\t"]:
            if cur:
                parts.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))

    if len(parts) < 6:
        return None

    # 兼容没有 conf 的情况：默认 1.0
    try:
        frame = int(float(parts[0]))
        tid = int(float(parts[1]))
        x = int(float(parts[2]))
        y = int(float(parts[3]))
        w = int(float(parts[4]))
        h = int(float(parts[5]))
        conf = float(parts[6]) if len(parts) > 6 else 1.0
        raw_tail = ""
        if len(parts) > 7:
            raw_tail = "," + ",".join(parts[7:])
        return MotLine(frame=frame, tid=tid, x=x, y=y, w=w, h=h, conf=conf, raw_tail=raw_tail)
    except Exception:
        return None


def _format_mot_line(m: MotLine) -> str:
    return f"{m.frame},{m.tid},{m.x},{m.y},{m.w},{m.h},{m.conf:.6f}{m.raw_tail}"


def apply_stability_boost_to_lines(
    lines: List[str],
    *,
    window: int,
    thresh: float,
    boost: float,
) -> Tuple[List[str], int]:
    """
    对一个 predict_with_conf.txt 的所有行做稳定性加分。
    返回：(新行列表, 被修改的 detection 行数)
    """
    win = max(1, int(window))
    thresh_f = float(thresh)
    boost_f = float(boost)

    parsed: List[Tuple[int, int, MotLine]] = []  # (orig_idx, frame, motline)
    untouched: Dict[int, str] = {}  # orig_idx -> 原始行（无法解析的行原样保留）

    for i, ln in enumerate(lines):
        m = _parse_mot_line(ln)
        if m is None:
            untouched[i] = ln.rstrip("\n")
        else:
            parsed.append((i, int(m.frame), m))

    # 在线：按时间顺序处理。对同一 frame，保持原文件相对顺序（orig_idx）。
    parsed.sort(key=lambda z: (z[1], z[0]))

    # tid -> deque(history_conf)
    cache: Dict[int, deque] = {}
    changed = 0
    updated_by_idx: Dict[int, str] = {}

    for orig_idx, _, m in parsed:
        dq = cache.get(m.tid)
        if dq is None:
            dq = deque(maxlen=win)
            cache[m.tid] = dq

        old_p = float(m.conf)
        new_p = old_p
        if len(dq) >= win and all(float(v) >= thresh_f for v in dq):
            new_p = min(1.0, old_p + boost_f)
        # append 使用“最终输出”的 p（与 llm_eval_from_mot.py 逻辑一致）
        dq.append(float(new_p))
        if abs(new_p - old_p) > 1e-12:
            changed += 1
        m.conf = float(new_p)
        updated_by_idx[orig_idx] = _format_mot_line(m)

    # 复原到原始行顺序
    out_lines: List[str] = []
    for i in range(len(lines)):
        if i in updated_by_idx:
            out_lines.append(updated_by_idx[i])
        elif i in untouched:
            out_lines.append(untouched[i])
        else:
            # 理论不应发生
            out_lines.append(lines[i].rstrip("\n"))
    return out_lines, changed


def _copy_tree_with_transform(
    input_root: str,
    output_root: str,
    *,
    window: int,
    thresh: float,
    boost: float,
    overwrite: bool,
    dry_run: bool,
) -> None:
    input_root = os.path.abspath(input_root)
    output_root = os.path.abspath(output_root)

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"input_root 不存在或不是目录: {input_root}")
    if os.path.exists(output_root):
        if not overwrite:
            raise FileExistsError(f"output_root 已存在（加 --overwrite 可覆盖）: {output_root}")
        if not dry_run:
            shutil.rmtree(output_root)

    total_files = 0
    total_target = 0
    total_changed_lines = 0
    total_modified_dets = 0

    for dirpath, dirnames, filenames in os.walk(input_root):
        rel = os.path.relpath(dirpath, input_root)
        out_dir = output_root if rel == "." else os.path.join(output_root, rel)
        if not dry_run:
            os.makedirs(out_dir, exist_ok=True)

        for fn in filenames:
            total_files += 1
            src = os.path.join(dirpath, fn)
            dst = os.path.join(out_dir, fn)

            if fn == "predict_with_conf.txt":
                total_target += 1
                with open(src, "r", encoding="utf-8") as f:
                    raw_lines = f.read().splitlines()
                new_lines, changed = apply_stability_boost_to_lines(
                    raw_lines, window=window, thresh=thresh, boost=boost
                )
                total_changed_lines += changed
                total_modified_dets += changed
                if not dry_run:
                    with open(dst, "w", encoding="utf-8") as f:
                        f.write("\n".join(new_lines))
            else:
                if not dry_run:
                    shutil.copy2(src, dst)

    print(
        "[DONE] copy+stability_boost\n"
        f"  input_root:  {input_root}\n"
        f"  output_root: {output_root}\n"
        f"  files_total: {total_files}\n"
        f"  targets(predict_with_conf.txt): {total_target}\n"
        f"  modified_detections: {total_modified_dets}\n"
        f"  params: window={int(window)} thresh={float(thresh)} boost={float(boost)}\n"
        f"  dry_run: {bool(dry_run)}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_root",
        type=str,
        default='track_result/v1_original',
        help="输入结果目录（例如 track_result/v2best59_original）",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default='track_result/v1_original_stable',
        help="输出结果目录（例如 track_result/v2best59_original_stable）",
    )
    p.add_argument("--window", type=int, default=3, help="稳定性窗口（最近 N 帧）")
    p.add_argument("--thresh", type=float, default=0.42, help="稳定性阈值（最近 N 帧均需 ≥ thresh）")
    p.add_argument("--boost", type=float, default=0.3, help="满足稳定性条件时固定加分（裁剪到 ≤1.0）")
    p.add_argument("--overwrite", action="store_true", help="若输出目录已存在则删除并重建")
    p.add_argument("--dry_run", action="store_true", help="只统计不写文件")
    args = p.parse_args()

    _copy_tree_with_transform(
        input_root=str(args.input_root),
        output_root=str(args.output_root),
        window=int(args.window),
        thresh=float(args.thresh),
        boost=float(args.boost),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()


