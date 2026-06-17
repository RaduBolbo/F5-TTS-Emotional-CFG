"""Print per-label sample counts for one or more descriptor JSONs.

Useful before running balance_dataset.py to decide on a target count, label
whitelist, or strategy.
"""

import argparse
import json
from collections import Counter


TOP_KEYS = ("ESD", "RAVDESS", "CREMA-D")
LABEL_FIELDS = ("emotion", "label")


def _load(path):
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path}:{i}: invalid JSON line ({e})")
        return path, records
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in TOP_KEYS:
        if key in data:
            return key, data[key]
    raise SystemExit(f"Descriptor {path} has no top key in {TOP_KEYS}.")


def _print_table(counts, total):
    if not counts:
        print("  (empty)")
        return
    width = max(len(lbl) for lbl in counts)
    for lbl, n in counts.most_common():
        pct = 100.0 * n / total if total else 0.0
        print(f"  {lbl:<{width}}  {n:>8d}  {pct:5.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, nargs="+",
                   help="One or more descriptor JSONs.")
    p.add_argument("--merged-only", action="store_true",
                   help="Skip per-file breakdown; print only merged totals.")
    p.add_argument("--json", action="store_true",
                   help="Emit counts as JSON instead of a table.")
    p.add_argument("--label-field", default=None,
                   help=f"Record field to count. If unset, auto-detects from {LABEL_FIELDS}.")
    args = p.parse_args()

    per_file = {}
    merged = Counter()
    for path in args.input:
        _, records = _load(path)
        field = args.label_field or next(
            (k for k in LABEL_FIELDS if records and k in records[0]), None
        )
        if field is None:
            raise SystemExit(
                f"{path}: no label field found (looked for {LABEL_FIELDS}); "
                f"pass --label-field explicitly."
            )
        c = Counter(r[field] for r in records)
        per_file[path] = c
        merged.update(c)

    if args.json:
        print(json.dumps(
            {
                "per_file": {p: dict(c.most_common()) for p, c in per_file.items()},
                "merged": dict(merged.most_common()),
                "total": sum(merged.values()),
                "num_labels": len(merged),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    if not args.merged_only and len(args.input) > 1:
        for path, c in per_file.items():
            total = sum(c.values())
            print(f"\n{path}  ({total} records, {len(c)} labels)")
            _print_table(c, total)

    total = sum(merged.values())
    header = "Merged" if len(args.input) > 1 else args.input[0]
    print(f"\n{header}  ({total} records, {len(merged)} labels)")
    _print_table(merged, total)

    if merged:
        counts = list(merged.values())
        print(f"\nmin={min(counts)}  max={max(counts)}  "
              f"mean={sum(counts)/len(counts):.1f}  "
              f"imbalance={max(counts)/min(counts):.1f}x")


if __name__ == "__main__":
    main()
