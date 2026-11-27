"""Downloads the two pretraining datasets to a local folder so that training runs from local files.

Usage: python scripts/fetch_datasets.py [--root data] [--token HF_TOKEN]
       python scripts/fetch_datasets.py --from-github <owner>/<repo>[@ref]     (public repository made by
                                                                            scripts/mirror_datasets_to_github.py)
       python scripts/fetch_datasets.py --archive-url <url of a .zip>          (e.g. a GitHub Release asset)
Without --from-github/--archive-url it needs network access to huggingface.co. Afterwards point `hf_data.text_repo` / `hf_data.multimodal_repo` of a config
at the printed folders (configs/pretrain_sample.yaml does this); `datasets.load_dataset` accepts local folders.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from kairos.hf_data import fetch_archive, github_archive_url, local_dirs

REPOS = {"text_repo": "ffurfaro/keep-it-simple", "multimodal_repo": "ffurfaro/keep-it-simple-multimodal"}


def fetch(root: Path, token: str | None = None) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    dirs = local_dirs(root)
    for key, repo in REPOS.items():
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dirs[key], token=token)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--from-github", default=None, help="owner/repo[@ref] of a public mirror repository")
    parser.add_argument("--archive-url", default=None, help="URL of a zip with the two dataset folders")
    args = parser.parse_args()
    url = args.archive_url or (github_archive_url(args.from_github) if args.from_github else None)
    dirs = fetch_archive(url, args.root) if url else fetch(args.root, args.token)
    for key, path in dirs.items():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        print(f"{key}: {path} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
