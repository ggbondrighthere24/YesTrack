import os
import argparse
import csv
import re
from typing import Dict, List, Tuple, Set


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_gt_sequences(gt_root: str, use_seqmap: str = None) -> List[str]:
    """
    Return a list of per-expression GT sequence names (e.g., '0005__on-the-right,-silver-cars-are-located')
    by either reading a seqmap file or scanning the GT folder for directories.
    """
    if use_seqmap is not None and os.path.isfile(use_seqmap):
        seqs = []
        with open(use_seqmap, "r", newline="") as f:
            for i, row in enumerate(csv.reader(f)):
                if not row:
                    continue
                # Joining also recovers legacy, unquoted rows containing commas.
                s = ",".join(row).strip()
                if i == 0 and s == "name":
                    continue
                if not s:
                    continue
                seqs.append(s)
        return sorted(seqs)
    # Fallback: scan directories at gt_root
    seqs = [
        d for d in os.listdir(gt_root)
        if os.path.isdir(os.path.join(gt_root, d)) and "__" in d
    ]
    return sorted(seqs)


def normalize_expr_key(s: str) -> str:
    """
    Normalize an expression string for fuzzy matching between refer results and GT names.
    Steps:
      - lowercase
      - replace hyphens and spaces with underscores
      - remove commas
      - collapse multiple underscores
    """
    t = s.lower()
    t = t.replace("-", "_").replace(" ", "_")
    t = t.replace(",", "")
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def build_gt_mapping(gt_root: str, seqmap_path: str = None) -> Dict[str, Dict[str, Set[str]]]:
    """
    Build mapping per seqid:
      seqid -> { normalized_key -> {target GT expression names} }
    """
    mapping: Dict[str, Dict[str, Set[str]]] = {}
    gt_seqs = list_gt_sequences(gt_root, seqmap_path)
    for seq_name in gt_seqs:
        if "__" not in seq_name:
            continue
        seqid, expr = seq_name.split("__", 1)
        key = normalize_expr_key(expr)
        if seqid not in mapping:
            mapping[seqid] = {}
        mapping[seqid].setdefault(key, set()).add(expr)
    return mapping


def collect_refer_items(refer_root: str, structure: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Collect refer items per seqid.
    Returns:
      seqid -> list of (path, name)
    For 'flat': name is file basename without .txt
    For 'nested': name is subdirectory name
    """
    result: Dict[str, List[Tuple[str, str]]] = {}
    for seqid in sorted(os.listdir(refer_root)):
        seq_dir = os.path.join(refer_root, seqid)
        if not os.path.isdir(seq_dir):
            continue
        items: List[Tuple[str, str]] = []
        if structure == "flat":
            for fname in sorted(os.listdir(seq_dir)):
                if not fname.endswith(".txt"):
                    continue
                name = os.path.splitext(fname)[0]
                items.append((os.path.join(seq_dir, fname), name))
        elif structure == "nested":
            for d in sorted(os.listdir(seq_dir)):
                dpath = os.path.join(seq_dir, d)
                if not os.path.isdir(dpath):
                    continue
                items.append((dpath, d))
        else:
            raise ValueError(f"Unknown structure: {structure}")
        if items:
            result[seqid] = items
    return result


def plan_restorations(
    refer_root: str,
    gt_root: str,
    seqmap_path: str,
    structure: str,
) -> Tuple[List[Tuple[str, str]], Dict[str, int], List[str]]:
    """
    Compute rename plan.
    Returns:
      - operations: list of (old_path, new_path)
      - stats: dict with counters
      - warnings: list of warning strings
    """
    gt_map = build_gt_mapping(gt_root, seqmap_path)
    refer_items = collect_refer_items(refer_root, structure)

    stats = {
        "seqs_scanned": 0,
        "items_total": 0,
        "items_matched": 0,
        "items_ambiguous": 0,
        "items_unmatched": 0,
        "would_rename": 0,
        "conflicts": 0,
    }
    warnings: List[str] = []
    ops: List[Tuple[str, str]] = []

    for seqid, items in refer_items.items():
        stats["seqs_scanned"] += 1
        key_to_exprs = gt_map.get(seqid, {})

        for old_path, name in items:
            stats["items_total"] += 1
            # Normalize refer name to GT key space
            # We first convert refer underscores to hyphens to approach GT, then normalize
            name_hyphen = name.replace("_", "-")
            key = normalize_expr_key(name_hyphen)

            if key not in key_to_exprs:
                stats["items_unmatched"] += 1
                warnings.append(f"[{seqid}] Unmatched refer item: '{name}' (key='{key}')")
                continue

            exprs = sorted(key_to_exprs[key])
            if len(exprs) > 1:
                stats["items_ambiguous"] += 1
                warnings.append(f"[{seqid}] Ambiguous key '{key}' maps to multiple GT exprs: {exprs}")
                continue

            target_expr = exprs[0]  # hyphen style, may include commas
            stats["items_matched"] += 1

            # Determine new path
            if structure == "flat":
                new_basename = f"{target_expr}.txt"
                new_path = os.path.join(os.path.dirname(old_path), new_basename)
            else:
                new_path = os.path.join(os.path.dirname(old_path), target_expr)

            # Skip no-op
            if os.path.normpath(old_path) == os.path.normpath(new_path):
                continue

            if os.path.exists(new_path):
                stats["conflicts"] += 1
                warnings.append(f"[{seqid}] Target exists, skip: {new_path}")
                continue

            stats["would_rename"] += 1
            ops.append((old_path, new_path))

    return ops, stats, warnings


def execute_ops(ops: List[Tuple[str, str]], dry_run: bool):
    for src, dst in ops:
        if dry_run:
            continue
        os.rename(src, dst)


def main():
    parser = argparse.ArgumentParser("Restore commas in refer result names by aligning to GT per-expression names")
    parser.add_argument("--refer-root", type=str, default='track_result/TrackLLM_on_rfdetr_v2/trackllm', help="Path to refer results root (e.g., track_result/refer)")
    parser.add_argument("--gt-root", type=str, default='outputs/refer_kitti_motc_gt_v2', help="Path to GT per-expression root (e.g., outputs/refer_kitti_motc_gt_v2)")
    parser.add_argument("--seqmap-file", type=str, default='outputs/refer_kitti_motc_gt_v2/seqmaps/val.txt', help="Optional seqmap file listing target GT sequences")
    parser.add_argument("--structure", type=str, choices=["flat", "nested"], default="nested", help="Refer input structure: flat or nested")
    parser.add_argument("--commit", action="store_true", help="Actually perform renames. Otherwise only report (dry-run).")
    args = parser.parse_args()

    ops, stats, warnings = plan_restorations(
        refer_root=args.refer_root,
        gt_root=args.gt_root,
        seqmap_path=args.seqmap_file,
        structure=args.structure,
    )

    # Execute
    execute_ops(ops, dry_run=not args.commit)

    # Report
    mode = "DRY-RUN" if not args.commit else "RENAMED"
    print(f"[{mode}] Refer structure = {args.structure}")
    print(f"Sequences scanned: {stats['seqs_scanned']}")
    print(f"Items total: {stats['items_total']}")
    print(f"Matched: {stats['items_matched']}")
    print(f"Ambiguous: {stats['items_ambiguous']}")
    print(f"Unmatched: {stats['items_unmatched']}")
    print(f"Would rename: {stats['would_rename']}")
    print(f"Conflicts: {stats['conflicts']}")
    if ops:
        print("\nPlanned renames (src -> dst):")
        for src, dst in ops[:200]:
            print(f"{src} -> {dst}")
        if len(ops) > 200:
            print(f"... and {len(ops) - 200} more")
    if warnings:
        print("\nWarnings:")
        for w in warnings[:200]:
            print(w)
        if len(warnings) > 200:
            print(f"... and {len(warnings) - 200} more")


if __name__ == "__main__":
    main()

