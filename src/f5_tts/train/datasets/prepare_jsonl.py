"""Convert a JSONL metadata file into F5-TTS training data (raw.arrow + duration.json + vocab.txt).

Expected JSONL row:
    {"audio": "/abs/path/sample.wav", "text": "(Levantine)...", "duration": 3.9}

Use --pinyin only for ZH/EN text; for other scripts (Arabic, etc.) leave it off
so the transcript is tokenized as raw characters.
"""

import argparse
import json
import shutil
from importlib.resources import files
from pathlib import Path

from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter


PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")


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

    records, durations, vocab = [], [], set()
    skipped_missing_audio = 0
    skipped_missing_field = 0
    skipped_duration = 0

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

        if use_pinyin:
            text = convert_char_to_pinyin([text], polyphone=True)[0]

        records.append({"audio_path": audio, "text": text, "duration": dur})
        durations.append(dur)
        vocab.update(list(text))

    return records, durations, vocab, skipped_missing_field, skipped_missing_audio, skipped_duration


def save(out_dir, records, durations, vocab, is_finetune):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {out_dir} ...")

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
    print(f"\n[{name}] samples: {len(records)}")
    print(f"[{name}] vocab size: {len(vocab)}")
    print(f"[{name}] total duration: {sum(durations)/3600:.2f} h")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input JSONL file.")
    p.add_argument("--out-dir", required=True, help="Output directory for raw.arrow / duration.json / vocab.txt.")
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
    args = p.parse_args()

    records, durations, vocab, skipped_field, skipped_audio, skipped_duration = build_records(
        args.input,
        args.audio_field,
        args.text_field,
        args.duration_field,
        use_pinyin=args.pinyin,
        check_exists=args.check_audio_exists,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    if skipped_field:
        print(f"Skipped {skipped_field} rows missing required field(s).")
    if skipped_duration:
        print(f"Skipped {skipped_duration} rows outside duration "
              f"({args.min_duration}, {args.max_duration})s.")
    if skipped_audio:
        print(f"Skipped {skipped_audio} rows whose audio file was not found.")
    if not records:
        raise SystemExit("No usable records.")

    save(args.out_dir, records, durations, vocab, is_finetune=not args.pretrain)


if __name__ == "__main__":
    main()
