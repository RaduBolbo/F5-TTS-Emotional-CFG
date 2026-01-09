#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def _normalize_hf_repo_id(repo_id: str) -> str:
    """
    Accept either 'username/repo' or a full URL like
    'https://huggingface.co/username/repo' and normalize to 'username/repo'.
    """
    prefixes = ("https://huggingface.co/", "http://huggingface.co/")
    for p in prefixes:
        if repo_id.startswith(p):
            repo_id = repo_id[len(p):]
            break
    repo_id = repo_id.strip().strip("/")
    if "/blob/" in repo_id or "/tree/" in repo_id:
        repo_id = repo_id.split("/blob/")[0].split("/tree/")[0]
    return repo_id


def _normalize_ms_repo_id(repo_id: str) -> str:
    """
    Accept either 'username/repo' or a full URL like:
      - https://www.modelscope.ai/models/username/repo
      - https://www.modelscope.ai/models/username/repo/files
      - https://modelscope.cn/models/username/repo
      - https://modelscope.cn/models/username/repo/files
    and normalize to 'username/repo'.
    """
    prefixes = (
        "https://www.modelscope.ai/models/",
        "http://www.modelscope.ai/models/",
        "https://modelscope.ai/models/",
        "http://modelscope.ai/models/",
        "https://modelscope.cn/models/",
        "http://modelscope.cn/models/",
        "https://www.modelscope.cn/models/",
        "http://www.modelscope.cn/models/",
    )
    for p in prefixes:
        if repo_id.startswith(p):
            repo_id = repo_id[len(p):]
            break

    repo_id = repo_id.strip().strip("/")
    if repo_id.endswith("/files"):
        repo_id = repo_id[: -len("/files")]
    return repo_id


def _copy_targets(cache_dir: Path, out_dir: Path, targets: list[str], force: bool) -> list[str]:
    saved_paths = []
    for fname in targets:
        src = cache_dir / fname
        if not src.exists():
            matches = list(cache_dir.rglob(fname))
            if len(matches) == 1:
                src = matches[0]
            else:
                print(f"✖️  File not found in downloaded snapshot: {fname}")
                raise SystemExit(1)

        dst = out_dir / fname
        if dst.exists() and not force:
            print(f"✔️  Already exists (use --force to overwrite): {dst}")
            saved_paths.append(str(dst))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✔️  Saved: {dst}")
        saved_paths.append(str(dst))
    return saved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download two .pt files from Hugging Face; fall back to ModelScope if HF fails."
    )
    parser.add_argument(
        "--repo-id",
        default="RaduBolbo/F5-TTS-Emotion-CFG-1",
        help="Hugging Face repo ID like 'username/repo' or full URL.",
    )
    parser.add_argument(
        "--ms-repo-id",
        default="https://www.modelscope.ai/models/RaduBolbo/F5-TTS-Emotion-CFG-1/files",
        help="ModelScope repo ID like 'username/repo' or full URL (including /files is OK).",
    )
    parser.add_argument(
        "--file1",
        default="model_emo.pt",
        help="First .pt filename to download (path inside repo).",
    )
    parser.add_argument(
        "--file2",
        default="model_0.pt",
        help="Second .pt filename to download (path inside repo).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch/tag/commit (default: repo default).",
    )
    parser.add_argument(
        "--out-dir",
        default="./ckpts",
        help="Where to place the downloaded files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite if files already exist in out-dir.",
    )
    args = parser.parse_args()

    hf_repo_id = _normalize_hf_repo_id(args.repo_id)
    ms_repo_id = _normalize_ms_repo_id(args.ms_repo_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [args.file1, args.file2]
    for f in targets:
        if not f.endswith(".pt"):
            print(f"Warning: '{f}' does not end with .pt (continuing anyway).")

    # 1) Try Hugging Face
    cache_dir = None
    source = None
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        print(f"→ Trying Hugging Face: {hf_repo_id}")
        cache_dir = hf_snapshot_download(
            repo_id=hf_repo_id,
            revision=args.revision,
            allow_patterns=targets,
        )
        cache_dir = Path(cache_dir)
        source = f"Hugging Face ({hf_repo_id})"
    except Exception as e:
        print(f"✖️  Hugging Face download failed: {e}")
        print(f"→ Falling back to ModelScope: {ms_repo_id}")

        # 2) Fallback to ModelScope
        try:
            from modelscope import snapshot_download as ms_snapshot_download
        except Exception:
            print("✖️  ModelScope fallback requires the 'modelscope' package.")
            print("    Install it with: pip install -U modelscope")
            raise SystemExit(1)

        try:
            cache_dir = ms_snapshot_download(
                ms_repo_id,
                revision=args.revision,
                allow_patterns=targets,
            )
            cache_dir = Path(cache_dir)
            source = f"ModelScope ({ms_repo_id})"
        except Exception as e2:
            print(f"✖️  ModelScope download also failed: {e2}")
            raise SystemExit(1)

    saved_paths = _copy_targets(cache_dir, out_dir, targets, args.force)

    print(f"\nDone (source: {source}). Files:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
