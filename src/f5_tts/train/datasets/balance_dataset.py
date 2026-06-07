"""Balance a descriptor JSON so each label contributes the same number of records.

Operates on the output of prepare_generic_dataset.py (or any descriptor whose
records carry an "emotion" field).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


TOP_KEYS = ("ESD", "RAVDESS", "CREMA-D")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in TOP_KEYS:
        if key in data:
            return key, data[key]
    raise SystemExit(f"Descriptor {path} has no top key in {TOP_KEYS}.")


def _group_by_label(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["emotion"]].append(r)
    return groups


def _resolve_target(groups, strategy, explicit):
    counts = {lbl: len(rs) for lbl, rs in groups.items()}
    if explicit is not None:
        return explicit
    if strategy == "undersample":
        return min(counts.values())
    if strategy == "oversample":
        return max(counts.values())
    if strategy == "median":
        sorted_counts = sorted(counts.values())
        return sorted_counts[len(sorted_counts) // 2]
    raise SystemExit(f"Unknown strategy {strategy!r}")


def balance(records, strategy, target_count, seed):
    rng = random.Random(seed)
    groups = _group_by_label(records)
    target = _resolve_target(groups, strategy, target_count)

    balanced = []
    per_label_final = Counter()
    for label, items in groups.items():
        if len(items) >= target:
            picked = rng.sample(items, target)
        else:
            picked = list(items)
            picked.extend(rng.choices(items, k=target - len(items)))
        balanced.extend(picked)
        per_label_final[label] = len(picked)

    rng.shuffle(balanced)
    return balanced, per_label_final, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Source descriptor JSON.")
    p.add_argument("--output", required=True, help="Where to write the balanced descriptor.")
    p.add_argument("--strategy", default="undersample",
                   choices=["undersample", "oversample", "median"],
                   help="How to pick the per-label target count if --target-count is unset.")
    p.add_argument("--target-count", type=int, default=None,
                   help="Explicit per-label sample count. Overrides --strategy.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for sampling reproducibility.")
    p.add_argument("--update-labels-file", default=None,
                   help="Optional path to a labels.json to rewrite with the new counts.")
    args = p.parse_args()

    top_key, records = _load(args.input)
    print(f"Loaded {len(records)} records under top key {top_key!r}")
    print("Original counts:")
    for lbl, n in Counter(r["emotion"] for r in records).most_common():
        print(f"  {lbl:40s} {n}")

    balanced, per_label_final, target = balance(
        records, args.strategy, args.target_count, args.seed
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({top_key: balanced}, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(balanced)} records (target={target}) → {out_path}")
    print("Balanced counts:")
    for lbl, n in per_label_final.most_common():
        print(f"  {lbl:40s} {n}")

    if args.update_labels_file:
        labels_path = Path(args.update_labels_file)
        if labels_path.exists():
            with open(labels_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        meta["labels"] = sorted(per_label_final)
        meta["counts"] = dict(per_label_final.most_common())
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Updated label vocab → {labels_path}")


if __name__ == "__main__":
    main()
