#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare evaluated sequences between two MOT evaluation "pedestrian_detailed.csv" files.

These CSVs typically have a first column named `seq` like:
  0005__automobiles-are-located-to-the-left
where the prefix before "__" is the KITTI sequence id (0005/0011/0013/...),
and the suffix is the referring expression/query.

This script reports:
  - sequence-id (prefix) differences (only in A / only in B)
  - full `seq` key differences (only in A / only in B)
  - per-sequence-id diffs of full keys (for shared prefixes)
  - duplicate `seq` keys inside each CSV (should usually be none)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


@dataclass(frozen=True)
class SeqIndex:
    """Index extracted from one CSV."""

    csv_path: Path
    full_keys: List[str]
    full_key_set: Set[str]
    prefix_keys: List[str]
    prefix_set: Set[str]
    full_by_prefix: Dict[str, Set[str]]
    dup_full_keys: Dict[str, int]


def _read_seq_column(csv_path: Path, seq_col: str = "seq") -> List[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if seq_col not in reader.fieldnames:
            raise KeyError(
                f"CSV missing column '{seq_col}' in {csv_path}. "
                f"Available columns: {reader.fieldnames[:20]}"
            )

        keys: List[str] = []
        for row in reader:
            v = (row.get(seq_col) or "").strip()
            if v == "":
                continue
            keys.append(v)
    return keys


def _prefix_of(full_key: str, sep: str = "__") -> str:
    return full_key.split(sep, 1)[0] if sep in full_key else full_key


def build_index(csv_path: Path, seq_col: str = "seq", sep: str = "__") -> SeqIndex:
    full_keys = _read_seq_column(csv_path, seq_col=seq_col)

    c = Counter(full_keys)
    dup = {k: v for k, v in c.items() if v > 1}

    prefixes = [_prefix_of(k, sep=sep) for k in full_keys]
    full_by_prefix: Dict[str, Set[str]] = defaultdict(set)
    for k in full_keys:
        full_by_prefix[_prefix_of(k, sep=sep)].add(k)

    return SeqIndex(
        csv_path=csv_path,
        full_keys=full_keys,
        full_key_set=set(full_keys),
        prefix_keys=prefixes,
        prefix_set=set(prefixes),
        full_by_prefix=dict(full_by_prefix),
        dup_full_keys=dup,
    )


def _sorted(xs: Iterable[str]) -> List[str]:
    return sorted(xs, key=lambda s: (len(s), s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default='track_result/TrackLLM_on_rfdetr_v2/trackllm/tracker/default_t0p10/pedestrian_detailed.csv', type=Path, help="CSV path A (baseline)")
    ap.add_argument("--b", default='track_result/v2best59_original_stable/tracker/default_t0p10/pedestrian_detailed.csv', type=Path, help="CSV path B (to compare)")
    ap.add_argument("--seq-col", default="seq", help="Column name containing seq key (default: seq)")
    ap.add_argument(
        "--sep",
        default="__",
        help="Separator between sequence id and query in seq key (default: __)",
    )
    ap.add_argument(
        "--max-show",
        type=int,
        default=50,
        help="Max items to print per diff list (default: 50). Use -1 to show all.",
    )
    args = ap.parse_args()

    idx_a = build_index(args.a, seq_col=args.seq_col, sep=args.sep)
    idx_b = build_index(args.b, seq_col=args.seq_col, sep=args.sep)

    def show_list(title: str, items: List[str]) -> None:
        print(f"\n== {title} (n={len(items)}) ==")
        if args.max_show == 0:
            return
        if args.max_show < 0:
            for x in items:
                print(x)
            return
        for x in items[: args.max_show]:
            print(x)
        if len(items) > args.max_show:
            print(f"... ({len(items) - args.max_show} more)")

    print(f"A: {idx_a.csv_path}  rows={len(idx_a.full_keys)} unique_full={len(idx_a.full_key_set)} unique_prefix={len(idx_a.prefix_set)}")
    print(f"B: {idx_b.csv_path}  rows={len(idx_b.full_keys)} unique_full={len(idx_b.full_key_set)} unique_prefix={len(idx_b.prefix_set)}")

    # 1) prefix-level differences (sequence ids)
    only_prefix_a = _sorted(idx_a.prefix_set - idx_b.prefix_set)
    only_prefix_b = _sorted(idx_b.prefix_set - idx_a.prefix_set)
    show_list("Sequence IDs only in A", only_prefix_a)
    show_list("Sequence IDs only in B", only_prefix_b)

    # 2) full-key differences (seq__query)
    only_full_a = _sorted(idx_a.full_key_set - idx_b.full_key_set)
    only_full_b = _sorted(idx_b.full_key_set - idx_a.full_key_set)
    show_list("Full `seq` keys only in A", only_full_a)
    show_list("Full `seq` keys only in B", only_full_b)

    # 3) per-prefix diffs for shared prefixes
    shared_prefixes = _sorted(idx_a.prefix_set & idx_b.prefix_set)
    per_prefix_missing: List[Tuple[str, int, int]] = []
    for p in shared_prefixes:
        a_set = idx_a.full_by_prefix.get(p, set())
        b_set = idx_b.full_by_prefix.get(p, set())
        a_only = a_set - b_set
        b_only = b_set - a_set
        if a_only or b_only:
            per_prefix_missing.append((p, len(a_only), len(b_only)))

    per_prefix_missing.sort(key=lambda t: (-(t[1] + t[2]), t[0]))
    print(f"\n== Per-sequence-id query diffs (shared prefixes) (n={len(per_prefix_missing)}) ==")
    if len(per_prefix_missing) == 0:
        print("None")
    else:
        for (p, na, nb) in per_prefix_missing[: (args.max_show if args.max_show >= 0 else len(per_prefix_missing))]:
            print(f"{p}: A-only {na}, B-only {nb}")
        if args.max_show >= 0 and len(per_prefix_missing) > args.max_show:
            print(f"... ({len(per_prefix_missing) - args.max_show} more)")

    # 4) duplicates inside each file
    def show_dups(tag: str, d: Dict[str, int]) -> None:
        print(f"\n== Duplicate full `seq` keys in {tag} (n={len(d)}) ==")
        if not d:
            print("None")
            return
        items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        max_show = args.max_show if args.max_show >= 0 else len(items)
        for k, v in items[:max_show]:
            print(f"{v}x  {k}")
        if args.max_show >= 0 and len(items) > args.max_show:
            print(f"... ({len(items) - args.max_show} more)")

    show_dups("A", idx_a.dup_full_keys)
    show_dups("B", idx_b.dup_full_keys)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

