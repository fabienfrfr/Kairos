"""Loads the pretraining datasets named in a config's `hf_data` section.

Three sources: Hugging Face repo ids, local folders, or a GitHub branch / zip that holds the two dataset folders
(`keep-it-simple/` and `keep-it-simple-multimodal/`, made by scripts/mirror_datasets_to_github.py). For the last one set
`github: owner/repo@branch` (or `archive_url: <zip url>`); the archive is downloaded once into a cache folder.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .dataset import split_examples

DATASET_FOLDERS = {"text_repo": "keep-it-simple", "multimodal_repo": "keep-it-simple-multimodal"}
DEFAULT_CACHE = Path.home() / ".cache" / "kairos" / "datasets"


def _text_rows(rows) -> list[dict]:
    return [{"modality": "text", "text": f"{r['prompt']} {r['text']}".strip()} for r in rows]


def local_dirs(root: Path) -> dict[str, Path]:
    """Folder of each dataset under root, keyed like the `hf_data` section of the configs."""
    return {key: Path(root) / folder for key, folder in DATASET_FOLDERS.items()}


def github_archive_url(spec: str) -> str:
    """`owner/repo` or `owner/repo@ref` -> the codeload zip URL of that branch, tag or commit (default HEAD)."""
    name, _, ref = spec.partition("@")
    if name.count("/") != 1 or not all(name.split("/")):
        raise ValueError(f"expected owner/repo[@ref], got {spec!r}")
    return f"https://codeload.github.com/{name}/zip/{ref or 'HEAD'}"


def fetch_archive(url: str, root: Path) -> dict[str, Path]:
    """Downloads a zip, extracts it into root (dropping GitHub's single top-level folder) and checks the layout."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "archive.zip"
        with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if n and not n.endswith("/")]
            tops = {n.split("/")[0] for n in names}
            strip = len(tops) == 1 and not (tops & set(DATASET_FOLDERS.values()))
            for member in names:
                parts = Path(member).parts[1:] if strip else Path(member).parts
                if not parts:
                    continue
                target = (root / Path(*parts)).resolve()
                if root.resolve() not in target.parents:
                    raise ValueError(f"unsafe path in archive: {member}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    dirs = local_dirs(root)
    missing = [str(p) for p in dirs.values() if not p.is_dir()]
    if missing:
        raise ValueError(f"the archive does not contain the expected folders: {missing}")
    return dirs


def fetch_github(
    spec: str | None = None, archive_url: str | None = None, cache_dir=None, refresh: bool = False
) -> dict[str, Path]:
    """Folders of the two datasets from a GitHub branch (`owner/repo@ref`) or a zip URL, downloaded once into the cache."""
    if not (spec or archive_url):
        raise ValueError("give a GitHub spec (owner/repo@ref) or an archive_url")
    root = Path(cache_dir or DEFAULT_CACHE) / re.sub(r"[^A-Za-z0-9._-]+", "_", archive_url or spec)
    dirs = local_dirs(root)
    if not refresh and all(p.is_dir() for p in dirs.values()):
        return dirs
    if root.exists():
        shutil.rmtree(root)
    return fetch_archive(archive_url or github_archive_url(spec), root)


def load_github_datasets(spec: str, cache_dir=None, refresh: bool = False, loader=None) -> dict:
    """Both datasets as `datasets.Dataset` objects, e.g. load_github_datasets("user/repo@datasets")["multimodal"]."""
    if loader is None:
        from datasets import load_dataset as loader

    dirs = fetch_github(spec, None, cache_dir, refresh)
    return {
        "text": loader(str(dirs["text_repo"]), split="train"),
        "multimodal": loader(str(dirs["multimodal_repo"]), split="train"),
    }


def load_hf_examples(spec: dict, loader=None) -> tuple[list, list]:
    """Returns (train, eval) examples from the text and multimodal repos or folders named in `spec`.

    `spec` may also carry `github` (owner/repo@ref) or `archive_url` (+ `cache_dir`, `refresh`) to read the two datasets
    from a GitHub branch or zip instead of the Hub; explicit `text_repo` / `multimodal_repo` entries win.
    """
    if loader is None:
        from datasets import load_dataset as loader

    spec = dict(spec)
    if spec.get("github") or spec.get("archive_url"):
        dirs = fetch_github(
            spec.get("github"), spec.get("archive_url"), spec.get("cache_dir"), spec.get("refresh", False)
        )
        for key, path in dirs.items():
            spec[key] = spec.get(key) or str(path)
    pct = spec.get("text_pct", 100)
    text = _text_rows(loader(spec["text_repo"], split=f"train[:{pct}%]")) if spec.get("text_repo") else []
    multimodal = list(loader(spec["multimodal_repo"], split="train")) if spec.get("multimodal_repo") else []
    return split_examples(text + multimodal, eval_pct=spec.get("eval_pct", 0.001), seed=spec.get("seed", 0))
