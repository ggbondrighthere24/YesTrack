#!/usr/bin/env python3
"""
Compare per-seq tracking metrics between two pedestrian_detailed.csv files,
grouped by attribute presence (object/position/state/color) derived from
masks_extract_output.json.

Rule:
- For a given sentence, if an attribute mask is all 1s => that attribute is NOT present.
- Otherwise (any 0) => attribute present.

Outputs a bar chart PNG saved to --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ASPECTS = ("object", "position", "state", "color")


@dataclass(frozen=True)
class Key:
    scene: str
    sentence: str  # normalized (lowercase, spaces)


def _norm_sentence_from_seq(seq: str) -> Optional[Key]:
    """
    Supports:
    - 0005+black-cars-in-the-left
    - 0005__black-cars-in-the-left
    Returns Key(scene='0005', sentence='black cars in the left')
    """
    if "+" in seq:
        scene, rest = seq.split("+", 1)
    elif "__" in seq:
        scene, rest = seq.split("__", 1)
    else:
        # Unknown format; try best-effort: leading digits + separator
        m = re.match(r"^(\d+)[_+](.+)$", seq)
        if not m:
            return None
        scene, rest = m.group(1), m.group(2)

    sentence = rest.replace("-", " ").strip().lower()
    if not scene or not sentence:
        return None
    return Key(scene=scene, sentence=sentence)


def _parse_mask_str(mask_str: str) -> List[int]:
    """
    mask_str is stored as a JSON string representing a list, e.g. "[1, 0, 1]".
    We parse it into list[int].
    """
    # Some files may have stray spaces; json.loads handles them fine
    arr = json.loads(mask_str)
    if not isinstance(arr, list):
        raise ValueError(f"mask is not list: {mask_str[:50]}...")
    out: List[int] = []
    for x in arr:
        try:
            out.append(int(x))
        except Exception as e:
            raise ValueError(f"mask element not int-ish: {x!r}") from e
    return out


def _mask_has_aspect(mask: List[int]) -> bool:
    # any 0 => aspect present; all 1 => absent
    return any(v == 0 for v in mask)


def load_sentence_aspect_presence_stream(json_path: Path) -> Dict[str, Dict[str, bool]]:
    """
    Stream-parse masks_extract_output.json without loading into memory.

    Expected per-entry structure (order assumed consistent in your file):
      {
        "sentence": "...",
        "object_mask": "[...]",
        "position_mask": "[...]",
        "state_mask": "[...]",
        "color_mask": "[...]"
      },

    Returns mapping: sentence(lowercased) -> {aspect: present_bool}
    """
    sentence_re = re.compile(r'^\s*"sentence"\s*:\s*"(?P<sent>.*)"\s*,?\s*$')
    mask_re = re.compile(r'^\s*"(?P<k>object_mask|position_mask|state_mask|color_mask)"\s*:\s*"(?P<v>\[.*\])"\s*,?\s*$')

    presence: Dict[str, Dict[str, bool]] = {}
    cur_sentence: Optional[str] = None
    cur_masks: Dict[str, List[int]] = {}

    def flush():
        nonlocal cur_sentence, cur_masks
        if cur_sentence is None:
            return
        if all(k in cur_masks for k in ("object_mask", "position_mask", "state_mask", "color_mask")):
            presence[cur_sentence] = {
                "object": _mask_has_aspect(cur_masks["object_mask"]),
                "position": _mask_has_aspect(cur_masks["position_mask"]),
                "state": _mask_has_aspect(cur_masks["state_mask"]),
                "color": _mask_has_aspect(cur_masks["color_mask"]),
            }
        # reset for next entry
        cur_sentence = None
        cur_masks = {}

    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            m_sent = sentence_re.match(line)
            if m_sent:
                # start of new entry: flush previous
                flush()
                cur_sentence = m_sent.group("sent").strip().lower()
                continue

            m_mask = mask_re.match(line)
            if m_mask and cur_sentence is not None:
                k = m_mask.group("k")
                v = m_mask.group("v")
                try:
                    cur_masks[k] = _parse_mask_str(v)
                except Exception:
                    # if parsing fails, keep going but entry won't be counted
                    continue

        # final flush
        flush()

    return presence


def load_metric_by_key(csv_path: Path, metric: str) -> Dict[Key, float]:
    out: Dict[Key, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "seq" not in reader.fieldnames:
            raise ValueError(f"`seq` column not found in {csv_path}")
        if metric not in reader.fieldnames:
            raise ValueError(f"metric column {metric!r} not found in {csv_path}")

        for row in reader:
            seq = (row.get("seq") or "").strip()
            k = _norm_sentence_from_seq(seq)
            if k is None:
                continue
            val_raw = (row.get(metric) or "").strip()
            if val_raw == "":
                continue
            try:
                val = float(val_raw)
            except Exception:
                continue
            out[k] = val
    return out


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def plot_compare(
    means_a: Dict[str, float],
    means_b: Dict[str, float],
    counts: Dict[str, int],
    label_a: str,
    label_b: str,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "matplotlib/numpy not available; please install them (e.g. `pip install matplotlib numpy`)."
        ) from e

    aspects = list(ASPECTS)
    a_vals = [means_a.get(a, float("nan")) for a in aspects]
    b_vals = [means_b.get(a, float("nan")) for a in aspects]
    xlabels = [f"{a}\nN={counts.get(a, 0)}" for a in aspects]

    x = np.arange(len(aspects))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax.bar(x - width / 2, a_vals, width, label=label_a)
    ax.bar(x + width / 2, b_vals, width, label=label_b)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")

    if title is None:
        title = f"Attribute-wise comparison on common seqs ({metric})"
    ax.set_title(title)

    # annotate bars
    def _annot(vals, dx):
        for i, v in enumerate(vals):
            if v != v:  # nan
                continue
            ax.text(i + dx, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    _annot(a_vals, -width / 2)
    _annot(b_vals, +width / 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def combo_label(bits: Dict[str, bool]) -> str:
    """
    Human-friendly label for an attribute-combination.
    Example:
      {object: True, position: False, state: True, color: False} -> "object+state"
    """
    present = [a for a in ASPECTS if bits.get(a, False)]
    if not present:
        return "none"
    if len(present) == len(ASPECTS):
        return "all"
    return "+".join(present)


def all_combo_aspect_sets(
    *,
    include_none: bool = False,
    include_all: bool = True,
) -> List[Tuple[str, Tuple[str, ...]]]:
    """
    Enumerate attribute combinations as (label, aspects_tuple).
    Default: all non-empty subsets, plus 'all'.
    Order: by size then lexicographically.
    """
    aspects = list(ASPECTS)
    out: List[Tuple[str, Tuple[str, ...]]] = []

    start_k = 0 if include_none else 1
    end_k = len(aspects) + 1 if include_all else len(aspects)

    for k in range(start_k, end_k + 1):
        if k == 0:
            out.append(("none", tuple()))
            continue
        if k > len(aspects):
            continue
        for comb in itertools.combinations(aspects, k):
            if len(comb) == len(aspects):
                out.append(("all", tuple(aspects)))
            else:
                out.append(("+".join(comb), tuple(comb)))

    # de-dup and stable sort
    uniq: Dict[str, Tuple[str, ...]] = {}
    for lbl, aset in out:
        uniq[lbl] = aset
    def _k(item: Tuple[str, Tuple[str, ...]]) -> Tuple[int, str]:
        lbl, aset = item
        if lbl == "none":
            return (0, "")
        if lbl == "all":
            return (99, "")
        return (len(aset), lbl)
    return sorted([(lbl, aset) for lbl, aset in uniq.items()], key=_k)


def save_combo_csv(
    out_csv: Path,
    rows: List[Tuple[str, int, float, float]],
    label_a: str,
    label_b: str,
    metric: str,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "N", f"{label_a}_{metric}", f"{label_b}_{metric}"])
        for combo, n, va, vb in rows:
            # If N=0, keep cells empty to make it obvious in spreadsheets
            if n == 0 or (va != va) or (vb != vb):  # nan check
                w.writerow([combo, n, "", ""])
            else:
                w.writerow([combo, n, va, vb])


def plot_combo_compare(
    rows: List[Tuple[str, int, float, float]],
    label_a: str,
    label_b: str,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
) -> None:
    """
    rows: (combo_label, N, mean_a, mean_b)
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "matplotlib/numpy not available; please install them (e.g. `pip install matplotlib numpy`)."
        ) from e

    # sort by N desc, then combo label
    rows_sorted = sorted(rows, key=lambda r: (-r[1], r[0]))
    combos = [r[0] for r in rows_sorted]
    counts = [r[1] for r in rows_sorted]
    a_vals = [r[2] for r in rows_sorted]
    b_vals = [r[3] for r in rows_sorted]

    y = np.arange(len(combos))
    h = 0.38

    fig_h = max(5.0, 0.35 * len(combos) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=160)
    ax.barh(y - h / 2, a_vals, h, label=label_a)
    ax.barh(y + h / 2, b_vals, h, label=label_b)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (N={n})" for c, n in zip(combos, counts)])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(metric)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")

    if title is None:
        title = f"Attribute-combination comparison on common seqs ({metric})"
    ax.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-a", default="ikun_refermllm_gt/ikun/pedestrian_detailed.csv", type=Path, help="First pedestrian_detailed.csv")
    ap.add_argument("--csv-b", default="ikun_refermllm_gt/refermllm/tracker/default_t0p30/pedestrian_detailed.csv", type=Path, help="Second pedestrian_detailed.csv")
    ap.add_argument("--json", default="masks_extract_output.json", type=Path, help="masks_extract_output.json")
    ap.add_argument("--metric", default="HOTA(0)", help="Metric column to compare (default: HOTA(0))")
    ap.add_argument("--label-a", default="iKUN", help="Legend label for csv-a")
    ap.add_argument("--label-b", default="ReferMLLM", help="Legend label for csv-b")
    ap.add_argument("--out",default="ikun_refermllm_gt/plot_attr_perf_compare.png", type=Path, help="Output PNG path")
    ap.add_argument("--title", default=None, help="Optional plot title")
    ap.add_argument(
        "--out-combos",
        default=None,
        type=Path,
        help="Optional output PNG path for 16 attribute-combinations (default: <out>_combos.png)",
    )
    ap.add_argument(
        "--out-combos-csv",
        default=None,
        type=Path,
        help="Optional output CSV path for 16 attribute-combinations (default: <out>_combos.csv)",
    )
    ap.add_argument(
        "--combo-mode",
        default="subset",
        choices=["subset", "exact"],
        help="How to bucket combinations: subset (sentence aspects is a superset) or exact (exact bitmask). Default: subset.",
    )
    ap.add_argument(
        "--include-none",
        action="store_true",
        help="Include the 'none' combination (empty subset). In subset-mode it includes all samples; default off.",
    )
    args = ap.parse_args()

    # TODO t1
    presence = load_sentence_aspect_presence_stream(args.json)

    # TODO t2
    a = load_metric_by_key(args.csv_a, args.metric)
    b = load_metric_by_key(args.csv_b, args.metric)
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        print("No common keys between CSVs after normalization. Check seq formats.", file=sys.stderr)
        return 2

    # For each aspect: filter to sentences where that aspect is present
    means_a: Dict[str, float] = {}
    means_b: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    missing_sentence = 0
    missing_keys_seen: set[Key] = set()
    for aspect in ASPECTS:
        keys_for_aspect: List[Key] = []
        for k in common_keys:
            pres = presence.get(k.sentence)
            if pres is None:
                if k not in missing_keys_seen:
                    missing_sentence += 1
                    missing_keys_seen.add(k)
                continue
            if pres.get(aspect, False):
                keys_for_aspect.append(k)
        counts[aspect] = len(keys_for_aspect)
        means_a[aspect] = mean(a[k] for k in keys_for_aspect)
        means_b[aspect] = mean(b[k] for k in keys_for_aspect)

    # Print a tiny numeric summary for sanity-checking
    print(f"Common keys: {len(common_keys)}")
    print(f"Metric: {args.metric}")
    for aspect in ASPECTS:
        ca = counts.get(aspect, 0)
        va = means_a.get(aspect, float('nan'))
        vb = means_b.get(aspect, float('nan'))
        print(f"- {aspect:8s} N={ca:4d}  {args.label_a}={va:.6f}  {args.label_b}={vb:.6f}")

    # --- combinations stats ---
    combo_vals_a: Dict[str, List[float]] = {}
    combo_vals_b: Dict[str, List[float]] = {}
    combo_counts: Dict[str, int] = {}

    if args.combo_mode == "exact":
        # Each key contributes to exactly one bucket (exact bitmask)
        for k in common_keys:
            pres = presence.get(k.sentence)
            if pres is None:
                continue
            cl = combo_label(pres)
            combo_vals_a.setdefault(cl, []).append(a[k])
            combo_vals_b.setdefault(cl, []).append(b[k])
            combo_counts[cl] = combo_counts.get(cl, 0) + 1
    else:
        # Subset mode: a key contributes to every bucket whose aspects are a subset of present aspects
        combos = all_combo_aspect_sets(include_none=args.include_none, include_all=True)
        for lbl, aset in combos:
            combo_vals_a[lbl] = []
            combo_vals_b[lbl] = []
            combo_counts[lbl] = 0
        for k in common_keys:
            pres = presence.get(k.sentence)
            if pres is None:
                continue
            present_set = {a for a in ASPECTS if pres.get(a, False)}
            for lbl, aset in combos:
                if set(aset).issubset(present_set):
                    combo_vals_a[lbl].append(a[k])
                    combo_vals_b[lbl].append(b[k])
                    combo_counts[lbl] += 1

    combo_rows: List[Tuple[str, int, float, float]] = []
    # Ensure a stable ordering: by size, then name, with 'none' first (if present) and 'all' last.
    ordered = all_combo_aspect_sets(include_none=args.include_none, include_all=True)
    ordered_labels = [lbl for lbl, _ in ordered] if args.combo_mode == "subset" else sorted(combo_counts.keys())
    for cl in ordered_labels:
        n = combo_counts.get(cl, 0)
        va = mean(combo_vals_a.get(cl, []))
        vb = mean(combo_vals_b.get(cl, []))
        combo_rows.append((cl, n, va, vb))

    # print combos summary (sorted by N desc)
    print("Combinations (sorted by N desc):")
    for cl, n, va, vb in sorted(combo_rows, key=lambda r: (-r[1], r[0])):
        if n == 0 or (va != va) or (vb != vb):
            print(f"- {cl:28s} N={n:4d}")
        else:
            print(f"- {cl:28s} N={n:4d}  {args.label_a}={va:.6f}  {args.label_b}={vb:.6f}")

    # TODO t3
    plot_compare(
        means_a=means_a,
        means_b=means_b,
        counts=counts,
        label_a=args.label_a,
        label_b=args.label_b,
        metric=args.metric,
        out_path=args.out,
        title=args.title,
    )

    out_combos = args.out_combos
    if out_combos is None:
        out_combos = args.out.parent / f"{args.out.stem}_combos{args.out.suffix}"
    out_combos_csv = args.out_combos_csv
    if out_combos_csv is None:
        out_combos_csv = args.out.parent / f"{args.out.stem}_combos.csv"

    save_combo_csv(
        out_csv=out_combos_csv,
        rows=combo_rows,
        label_a=args.label_a,
        label_b=args.label_b,
        metric=args.metric,
    )
    plot_combo_compare(
        # For plots, drop N=0 rows to avoid NaNs/blank bars; CSV still contains all rows.
        rows=[r for r in combo_rows if r[1] > 0],
        label_a=args.label_a,
        label_b=args.label_b,
        metric=args.metric,
        out_path=out_combos,
        title=(None if args.title is None else f"{args.title} (combinations)"),
    )

    if missing_sentence:
        print(
            f"Warning: {missing_sentence} common rows had sentences not found in JSON "
            f"(skipped in aspect buckets).",
            file=sys.stderr,
        )

    print(f"Saved plot: {args.out}")
    print(f"Saved combos plot: {out_combos}")
    print(f"Saved combos csv: {out_combos_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

