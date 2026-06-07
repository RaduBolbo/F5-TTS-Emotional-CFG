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


def _load_many(paths):
    merged = []
    top_keys_seen = set()
    for path in paths:
        key, records = _load(path)
        top_keys_seen.add(key)
        print(f"  {path}: {len(records)} records under {key!r}")
        merged.extend(records)
    if len(top_keys_seen) > 1:
        print(f"Note: inputs use different top keys {top_keys_seen}; "
              f"output will use {sorted(top_keys_seen)[0]!r}")
    return sorted(top_keys_seen)[0], merged


def _group_by_label(records, field):
    groups = defaultdict(list)
    for r in records:
        groups[r[field]].append(r)
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


def balance(records, strategy, target_count, seed, field):
    rng = random.Random(seed)
    groups = _group_by_label(records, field)
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
    p.add_argument("--input", required=True, nargs="+",
                   help="One or more source descriptor JSONs. Records are merged before balancing.")
    p.add_argument("--top-key", default=None,
                   help="Override the top-level key written to --output "
                        "(must be one of ESD/RAVDESS/CREMA-D).")
    p.add_argument("--output", required=True, help="Where to write the balanced descriptor.")
    p.add_argument("--labels", default=None,
                   help="Comma-separated whitelist of labels to keep. "
                        "Records with other labels are dropped before balancing.")
    p.add_argument("--strategy", default="undersample",
                   choices=["undersample", "oversample", "median"],
                   help="How to pick the per-label target count if --target-count is unset.")
    p.add_argument("--target-count", type=int, default=None,
                   help="Explicit per-label sample count. Overrides --strategy.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for sampling reproducibility.")
    p.add_argument("--update-labels-file", default=None,
                   help="Optional path to a labels.json to rewrite with the new counts.")
    p.add_argument("--label-field", default="emotion",
                   help="Record field naming the label (default: emotion).")
    args = p.parse_args()

    print(f"Loading {len(args.input)} descriptor(s):")
    top_key, records = _load_many(args.input)
    if args.top_key:
        if args.top_key not in TOP_KEYS:
            raise SystemExit(f"--top-key must be one of {TOP_KEYS} (got {args.top_key!r})")
        top_key = args.top_key
    print(f"Merged {len(records)} records; output top key = {top_key!r}")

    field = args.label_field
    if args.labels:
        wanted = {s.strip() for s in args.labels.split(",") if s.strip()}
        before = len(records)
        records = [r for r in records if r[field] in wanted]
        print(f"Filtered to {len(wanted)} label(s): kept {len(records)}/{before}")
        missing = wanted - {r[field] for r in records}
        if missing:
            print(f"Warning: no records found for labels: {sorted(missing)}")
    print("Original counts:")
    for lbl, n in Counter(r[field] for r in records).most_common():
        print(f"  {lbl:40s} {n}")

    balanced, per_label_final, target = balance(
        records, args.strategy, args.target_count, args.seed, field
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
