"""Convert a JSONL metadata file into F5-TTS training data.

Expected JSONL row:
    {"audio": "/abs/path/sample.wav", "text": "(Levantine)...", "duration": 3.9}

The leading "(label)" tag is parsed out of the transcript: the stripped text is
what gets written to the arrow dataset, and the captured label becomes the
"emotion" field in a sibling descriptor.json consumed by CustomDatasetConditioned.

Outputs in --out-dir:
    raw.arrow        # {audio_path, text, duration} for the standard loader
    duration.json    # durations list for DynamicBatchSampler
    vocab.txt        # char vocab (or pretrained Emilia vocab when finetuning)
    descriptor.json  # {top_key: [{audio_path, text, speaker_id, emotion, ...}]}
    labels.json      # label vocab + counts

Use --pinyin only for ZH/EN text; for other scripts (Arabic, etc.) leave it off
so the transcript is tokenized as raw characters.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from importlib.resources import files
from pathlib import Path

from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter


PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")
DESCRIPTOR_TOP_KEYS = ("ESD", "RAVDESS", "CREMA-D")
LABEL_TAG_RE = re.compile(r"^\s*\(([^)]+)\)\s*")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i}: invalid JSON line ({e})")


def split_label(text):
    """Return (label, text_without_tag). label is None if no '(...)' prefix."""
    m = LABEL_TAG_RE.match(text)
    if not m:
        return None, text
    return m.group(1).strip(), text[m.end():]


def speaker_id_for(audio_path):
    parent = os.path.basename(os.path.dirname(audio_path)) or audio_path
    return "spk_" + hashlib.md5(parent.encode("utf-8")).hexdigest()[:10]


def build_records(
    jsonl_path,
    audio_field,
    text_field,
    duration_field,
    use_pinyin,
    check_exists,
    min_duration,
    max_duration,
):
    if use_pinyin:
        from f5_tts.model.utils import convert_char_to_pinyin

    arrow_records, descriptor_records, durations = [], [], []
    vocab = set()
    label_counts = Counter()
    skipped_missing_audio = 0
    skipped_missing_field = 0
    skipped_duration = 0
    skipped_no_tag = 0

    for row in iter_jsonl(jsonl_path):
        audio = row.get(audio_field)
        text = row.get(text_field)
        dur = row.get(duration_field)
        if not audio or not text or dur is None:
            skipped_missing_field += 1
            continue
        dur = float(dur)
        if dur <= min_duration or dur >= max_duration:
            skipped_duration += 1
            continue
        if check_exists and not Path(audio).exists():
            skipped_missing_audio += 1
            continue

        label, raw_text = split_label(text)
        if label is None:
            skipped_no_tag += 1
            continue

        arrow_text = convert_char_to_pinyin([raw_text], polyphone=True)[0] if use_pinyin else raw_text

        arrow_records.append({"audio_path": audio, "text": arrow_text, "duration": dur})
        descriptor_records.append({
            "phrase_idx": str(len(descriptor_records)),
            "audio_path": audio,
            "text": raw_text,
            "speaker_id": speaker_id_for(audio),
            "emotion": label,
            "text_alignment": [],
        })
        durations.append(dur)
        vocab.update(list(arrow_text))
        label_counts[label] += 1

    return {
        "arrow": arrow_records,
        "descriptor": descriptor_records,
        "durations": durations,
        "vocab": vocab,
        "label_counts": label_counts,
        "skipped": {
            "missing_field": skipped_missing_field,
            "duration_out_of_range": skipped_duration,
            "missing_audio": skipped_missing_audio,
            "no_label_tag": skipped_no_tag,
        },
    }


def save_arrow(out_dir, records, durations, vocab, is_finetune):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving arrow dataset to {out_dir} ...")

    raw_arrow = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow.as_posix(), writer_batch_size=1) as writer:
        for r in tqdm(records, desc="Writing raw.arrow"):
            writer.write(r)

    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)

    vocab_path = out_dir / "vocab.txt"
    if is_finetune:
        if not PRETRAINED_VOCAB_PATH.exists():
            raise SystemExit(f"pretrained vocab.txt not found: {PRETRAINED_VOCAB_PATH}")
        shutil.copy2(PRETRAINED_VOCAB_PATH.as_posix(), vocab_path)
    else:
        with open(vocab_path, "w", encoding="utf-8") as f:
            for v in sorted(vocab):
                f.write(v + "\n")

    name = out_dir.stem
    print(f"[{name}] samples: {len(records)}")
    print(f"[{name}] vocab size: {len(vocab)}")
    print(f"[{name}] total duration: {sum(durations)/3600:.2f} h")


def save_descriptor(out_dir, descriptor_records, label_counts, top_key):
    out_dir = Path(out_dir)
    descriptor_path = out_dir / "descriptor.json"
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump({top_key: descriptor_records}, f, ensure_ascii=False, indent=2)

    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_field": "emotion",
                "labels": sorted(label_counts),
                "counts": dict(label_counts.most_common()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nWrote descriptor ({len(descriptor_records)} records) → {descriptor_path}")
    print(f"Wrote label vocab ({len(label_counts)}) → {labels_path}")
    print("Label counts:")
    for lbl, n in label_counts.most_common():
        print(f"  {lbl:40s} {n}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input JSONL file.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    p.add_argument("--audio-field", default="audio")
    p.add_argument("--text-field", default="text")
    p.add_argument("--duration-field", default="duration")
    p.add_argument("--pinyin", action="store_true",
                   help="Apply ZH/EN pinyin conversion to text. Leave off for Arabic / other scripts.")
    p.add_argument("--pretrain", action="store_true",
                   help="Write a fresh vocab.txt from the data. Default copies the pretrained Emilia vocab (finetune).")
    p.add_argument("--check-audio-exists", action="store_true",
                   help="Stat every audio path and skip rows whose file is missing.")
    p.add_argument("--min-duration", type=float, default=3.0,
                   help="Drop rows with duration <= this (seconds). Default: 3.0.")
    p.add_argument("--max-duration", type=float, default=15.0,
                   help="Drop rows with duration >= this (seconds). Default: 15.0.")
    p.add_argument("--top-key", default="ESD", choices=DESCRIPTOR_TOP_KEYS,
                   help="Top-level key in descriptor.json (loader hard-codes this set). Default: ESD.")
    p.add_argument("--no-descriptor", action="store_true",
                   help="Skip writing descriptor.json / labels.json.")
    args = p.parse_args()

    out = build_records(
        args.input,
        args.audio_field,
        args.text_field,
        args.duration_field,
        use_pinyin=args.pinyin,
        check_exists=args.check_audio_exists,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    skipped = out["skipped"]
    if skipped["missing_field"]:
        print(f"Skipped {skipped['missing_field']} rows missing required field(s).")
    if skipped["duration_out_of_range"]:
        print(f"Skipped {skipped['duration_out_of_range']} rows outside duration "
              f"({args.min_duration}, {args.max_duration})s.")
    if skipped["missing_audio"]:
        print(f"Skipped {skipped['missing_audio']} rows whose audio file was not found.")
    if skipped["no_label_tag"]:
        print(f"Skipped {skipped['no_label_tag']} rows with no leading '(label)' tag.")
    if not out["arrow"]:
        raise SystemExit("No usable records.")

    save_arrow(args.out_dir, out["arrow"], out["durations"], out["vocab"], is_finetune=not args.pretrain)
    if not args.no_descriptor:
        save_descriptor(args.out_dir, out["descriptor"], out["label_counts"], args.top_key)


if __name__ == "__main__":
    main()
