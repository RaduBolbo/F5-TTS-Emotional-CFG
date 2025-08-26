#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
# Backward-compatible error import
try:
    from huggingface_hub.utils import HfHubHTTPError  # >= 0.19
except Exception:  # older versions
    try:
        from requests import HTTPError as HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception):
            pass


def _normalize_repo_id(repo_id: str) -> str:
    """
    Accept either 'username/repo' or a full URL like
    'https://huggingface.co/username/repo' and normalize to 'username/repo'.
    """
    prefixes = ("https://huggingface.co/", "http://huggingface.co/")
    for p in prefixes:
        if repo_id.startswith(p):
            repo_id = repo_id[len(p):]
            break
    # Strip trailing slashes or 'tree/<rev>' etc.
    repo_id = repo_id.strip().strip("/")
    if "/blob/" in repo_id or "/tree/" in repo_id:
        repo_id = repo_id.split("/blob/")[0].split("/tree/")[0]
    return repo_id


def main():
    parser = argparse.ArgumentParser(description="Download two .pt files from a Hugging Face repo.")
    parser.add_argument("--repo-id", default="RaduBolbo/F5-TTS-Emotion-CFG-1",
                        help="Repo ID like 'username/repo_name' or full URL.")
    parser.add_argument("--file1", default="model_emo.pt",
                        help="First .pt filename to download (path inside repo).")
    parser.add_argument("--file2", default="model_0.pt",
                        help="Second .pt filename to download (path inside repo).")
    parser.add_argument("--revision", default=None,
                        help="Branch, tag, or commit (default: repo default).")
    parser.add_argument("--out-dir", default="./ckpts",
                        help="Where to place the downloaded files.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite if files already exist in out-dir.")
    args = parser.parse_args()

    # Normalize repo id (handles full URLs)
    repo_id = _normalize_repo_id(args.repo_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [args.file1, args.file2]
    for f in targets:
        if not f.endswith(".pt"):
            print(f"Warning: '{f}' does not end with .pt (continuing anyway).")

    # Download only the requested files into local cache
    try:
        cache_dir = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            allow_patterns=targets,  # only fetch these files
        )
    except Exception as e:
        # Older hub versions may not raise HfHubHTTPError specifically
        print(f"✖️  Download failed from {repo_id}: {e}")
        raise SystemExit(1)

    cache_dir = Path(cache_dir)

    # Copy the requested files from cache to out_dir
    saved_paths = []
    for fname in targets:
        src = cache_dir / fname
        if not src.exists():
            print(f"✖️  File not found in repo: {fname}")
            raise SystemExit(1)

        dst = out_dir / fname
        if dst.exists() and not args.force:
            print(f"✔️  Already exists (use --force to overwrite): {dst}")
            saved_paths.append(str(dst))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"✔️  Saved: {dst}")
        saved_paths.append(str(dst))

    print("\nDone. Files:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
