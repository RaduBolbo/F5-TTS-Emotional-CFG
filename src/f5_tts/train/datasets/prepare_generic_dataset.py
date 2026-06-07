"""Convert a generic JSON / JSONL metadata file into the descriptor format
expected by CustomDatasetConditioned.

The chosen --label-field is copied into the "emotion" key the loader reads,
so any categorical attribute (persona, dialect, speaking rate, ...) can drive
conditioning without touching the model code.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


def _iter_records(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048).lstrip()
        f.seek(0)
        if head.startswith("["):
            for r in json.load(f):
                yield r
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _get(row, dotted_key):
    """Look up a possibly nested key, e.g. 'gemini_response_json.Persona'."""
    cur = row
    for part in dotted_key.split("."):
        if isinstance(cur, str):
            try:
                cur = json.loads(cur)
            except json.JSONDecodeError:
                return None
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _stable_speaker_id(row, speaker_field):
    if speaker_field:
        val = _get(row, speaker_field)
        if val is not None:
            return str(val)
    audio = _get(row, "audio_path") or ""
    parent = os.path.basename(os.path.dirname(audio)) or audio
    return "spk_" + hashlib.md5(parent.encode("utf-8")).hexdigest()[:10]


def build_descriptor(args):
    records = []
    skipped = Counter()
    label_counts = Counter()

    allowed = (
        {s.strip() for s in args.allowed_labels.split(",") if s.strip()}
        if args.allowed_labels
        else None
    )

    for row in _iter_records(args.input):
        audio = _get(row, args.audio_field)
        text = _get(row, args.text_field)
        label = _get(row, args.label_field)

        if not audio or not text or label is None:
            skipped["missing_field"] += 1
            continue
        label = str(label)
        if allowed is not None and label not in allowed:
            skipped["not_in_allowed_labels"] += 1
            continue

        label_counts[label] += 1
        records.append(
            {
                "phrase_idx": str(_get(row, args.phrase_field) or len(records)),
                "audio_path": audio,
                "text": text,
                "speaker_id": _stable_speaker_id(row, args.speaker_field),
                "emotion": label,
                "text_alignment": [],
            }
        )

    if args.min_count > 1:
        rare = {lbl for lbl, n in label_counts.items() if n < args.min_count}
        if rare:
            records = [r for r in records if r["emotion"] not in rare]
            for lbl in rare:
                skipped[f"min_count<{args.min_count}:{lbl}"] = label_counts.pop(lbl)

    return records, label_counts, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="JSON array or JSONL of metadata rows.")
    p.add_argument("--output", required=True, help="Where to write the descriptor JSON.")
    p.add_argument("--label-field", required=True,
                   help="Field whose value becomes the categorical label (e.g. gemini_persona).")
    p.add_argument("--text-field", default="text",
                   help="Field with the transcript. Default: 'text'.")
    p.add_argument("--audio-field", default="audio_path",
                   help="Field with the absolute audio path. Default: 'audio_path'.")
    p.add_argument("--speaker-field", default=None,
                   help="Field with a speaker id. If absent, a hash of the audio dir is used.")
    p.add_argument("--phrase-field", default=None,
                   help="Field with a per-utterance id. If absent, the row index is used.")
    p.add_argument("--allowed-labels", default=None,
                   help="Comma-separated whitelist of labels to keep.")
    p.add_argument("--min-count", type=int, default=1,
                   help="Drop labels that occur fewer than this many times.")
    p.add_argument("--top-key", default="ESD",
                   help="Top-level key in the descriptor. Must be one of "
                        "'ESD' / 'RAVDESS' / 'CREMA-D' (the loader's hard-coded set).")
    args = p.parse_args()

    if args.top_key not in {"ESD", "RAVDESS", "CREMA-D"}:
        raise SystemExit(f"--top-key must be ESD, RAVDESS, or CREMA-D (got {args.top_key!r})")

    records, label_counts, skipped = build_descriptor(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({args.top_key: records}, f, ensure_ascii=False, indent=2)

    labels_path = out_path.parent / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_field": args.label_field,
                "labels": sorted(label_counts),
                "counts": dict(label_counts.most_common()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Wrote {len(records)} records → {out_path}")
    print(f"Wrote label vocab ({len(label_counts)}) → {labels_path}")
    print("Label counts:")
    for lbl, n in label_counts.most_common():
        print(f"  {lbl:40s} {n}")
    if skipped:
        print("Skipped:")
        for k, n in skipped.items():
            print(f"  {k:40s} {n}")


if __name__ == "__main__":
    main()
