"""Copies the two Hugging Face pretraining datasets into a git repository that can be pushed to GitHub.

Run it on a machine that can reach huggingface.co. GitHub refuses files above 100 MB, so every parquet file is re-sharded
below --max-file-mb, and the text dataset can be reduced to its first --text-rows rows. Nothing is pushed unless --push is
given; the script never handles credentials (git uses yours).

Usage:
  python scripts/mirror_datasets_to_github.py --out ../kairos-data --text-rows 20000
  python scripts/mirror_datasets_to_github.py --out ../kairos-data --text-rows 20000 --push \\
      --remote https://github.com/<user>/kairos-data.git
  python scripts/mirror_datasets_to_github.py --out ../kairos-data --push --branch datasets \\
      --remote https://github.com/<user>/<existing repo>.git       # data on its own branch of an existing repository
  python scripts/mirror_datasets_to_github.py --out ../kairos-data --zip     # also writes kairos-data.zip for a GitHub Release

Check the licences of the upstream sources (see scripts/pretrain/readme_multimodal.md) before publishing: a public repository
republishes the data. Afterwards: python scripts/fetch_datasets.py --from-github <user>/kairos-data
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPOS = {"keep-it-simple": "ffurfaro/keep-it-simple", "keep-it-simple-multimodal": "ffurfaro/keep-it-simple-multimodal"}
GITHUB_HARD_LIMIT_MB = 100
REPO_SOFT_LIMIT_MB = 1000


def parquet_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.parquet") if p.is_file())


def write_shards(batches, dst: Path, max_bytes: int, prefix: str = "train") -> list[Path]:
    """Writes record batches as parquet shards of at most max_bytes.

    Each shard is built in memory so that its real size is known after every batch (ParquetWriter does not flush to a file
    until closed): it is closed once it reaches 90% of max_bytes, which leaves room for one more batch (a few percent of the
    cap, see mirror_folder) and the footer.
    """
    import pyarrow as pa

    dst.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    writer = sink = None
    for batch in batches:
        if writer is None:
            written.append(dst / f"{prefix}-{len(written):05d}.parquet")
            sink = pa.BufferOutputStream()
            writer = pq.ParquetWriter(sink, batch.schema)
        writer.write_batch(batch)
        if sink.tell() >= max_bytes * 0.9:
            writer.close()
            written[-1].write_bytes(sink.getvalue())
            writer = None
    if writer is not None:
        writer.close()
        written[-1].write_bytes(sink.getvalue())
    return written


def iter_batches(files: list[Path], max_rows: int | None = None, batch_rows: int = 256):
    seen = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
            if max_rows is not None and seen + batch.num_rows > max_rows:
                batch = batch.slice(0, max_rows - seen)
            seen += batch.num_rows
            if batch.num_rows:
                yield batch
            if max_rows is not None and seen >= max_rows:
                return


def mirror_folder(src: Path, dst: Path, max_bytes: int, max_rows: int | None = None) -> list[Path]:
    """Re-shards the parquet files of src into dst (all rows, or the first max_rows)."""
    files = parquet_files(src)
    if not files:
        raise SystemExit(f"no parquet file under {src}; export the dataset with datasets.Dataset.to_parquet first")
    first = pq.ParquetFile(files[0])
    bytes_per_row = max(1.0, files[0].stat().st_size / max(1, first.metadata.num_rows))
    # a batch is the unit of writing, so keep one batch small against the shard cap (rows can be large)
    batch_rows = max(1, min(256, int(max_bytes * 0.05 / bytes_per_row)))
    return write_shards(iter_batches(files, max_rows, batch_rows), dst, max_bytes)


def check_limits(root: Path, hard_mb: float = GITHUB_HARD_LIMIT_MB, soft_mb: float = REPO_SOFT_LIMIT_MB) -> float:
    """Raises if a file exceeds GitHub's hard limit; returns the total size in MB (warns above the soft limit)."""
    total = 0
    for path in (p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        size = path.stat().st_size
        if size > hard_mb * 2**20:
            raise SystemExit(f"{path} is {size / 2**20:.0f} MB: above GitHub's {hard_mb:g} MB limit")
        total += size
    if total > soft_mb * 2**20:
        print(f"warning: {total / 2**20:.0f} MB in total, above GitHub's recommended {soft_mb:g} MB per repository")
    return total / 2**20


def write_readme(out: Path, text_rows: int | None) -> None:
    lines = [
        "# Kairos pretraining datasets (mirror)",
        "",
        "Parquet copies of the Hugging Face datasets, re-sharded below GitHub's 100 MB file limit:",
        *(f"- `{name}/` from `{repo}`" for name, repo in REPOS.items()),
        "",
        f"Text dataset: {'all rows' if text_rows is None else f'first {text_rows} rows'}. Check the licences of the upstream sources",
        "(Flickr8k, AudioCaps, Molmo2-VideoCapQA, nuScenes-mini, MotionSense, PixelBytes-OptimalControl) before reusing.",
        "",
        "Load with `datasets.load_dataset('<folder>', split='train')` or `python scripts/fetch_datasets.py --from-github <user>/<repo>`.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def download(cache: Path, token: str | None) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    return {
        name: Path(snapshot_download(repo_id=repo, repo_type="dataset", local_dir=cache / name, token=token))
        for name, repo in REPOS.items()
    }


def build(sources: dict[str, Path], out: Path, max_file_mb: float, text_rows: int | None) -> dict[str, list[Path]]:
    """Fills out/<dataset>/ from the downloaded folders and checks GitHub's limits."""
    out.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_file_mb * 2**20)
    shards = {}
    for name, src in sources.items():
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        rows = text_rows if name == "keep-it-simple" else None
        shards[name] = mirror_folder(src, target, max_bytes, rows)
    write_readme(out, text_rows)
    check_limits(out)
    return shards


def git(out: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(out), *args], check=True)


def git_output(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=False, cwd=cwd, capture_output=True, text=True)


def default_remote() -> str | None:
    """URL of `origin` for the repository the script is run from (the project's own GitHub repository)."""
    result = git_output("config", "--get", "remote.origin.url")
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def remote_has_branch(remote: str, branch: str) -> bool:
    return git_output("ls-remote", "--exit-code", "--heads", remote, branch).returncode == 0


def prepare_repo(out: Path, remote: str | None, branch: str) -> None:
    """Makes `out` a working copy of the remote branch if it exists (so a rerun updates it), else a new repository."""
    exists = bool(remote) and remote_has_branch(remote, branch)
    if (out / ".git").exists():
        if exists:
            git(out, "fetch", "origin", branch)
            git(out, "checkout", "-B", branch, "FETCH_HEAD")
        return
    if exists:
        subprocess.run(
            ["git", "clone", "--branch", branch, "--single-branch", "--depth", "1", remote, str(out)], check=True
        )
    else:
        out.mkdir(parents=True, exist_ok=True)
        git(out, "init", "-b", branch)


def commit(out: Path, remote: str | None, push: bool, branch: str = "main") -> bool:
    """Commits the changes (if any) and optionally pushes them to the branch; returns whether a commit was made."""
    if not (out / ".git").exists():
        git(out, "init", "-b", branch)
    git(out, "add", "-A")
    changed = git_output("-C", str(out), "diff", "--cached", "--quiet").returncode != 0
    if changed:
        identity = ["-c", "user.name=kairos", "-c", "user.email=kairos@example.invalid"]
        git(out, *identity, "commit", "-m", "Update pretraining dataset mirror")
    if remote:
        remotes = git_output("-C", str(out), "remote").stdout.split()
        git(out, "remote", "set-url" if "origin" in remotes else "add", "origin", remote)
    if push and (changed or not (remote and remote_has_branch(remote, branch))):
        git(out, "push", "-u", "origin", f"HEAD:refs/heads/{branch}")
    return changed


def make_zip(out: Path) -> Path:
    return Path(shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder of the git repository to create")
    parser.add_argument("--cache", type=Path, default=Path("data"), help="where the Hub datasets are downloaded")
    parser.add_argument("--text-rows", type=int, default=None, help="keep only the first N rows of keep-it-simple")
    parser.add_argument("--max-file-mb", type=float, default=90, help="shard size cap (GitHub refuses 100 MB)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--remote", default=None, help="https URL of the empty GitHub repository")
    parser.add_argument(
        "--branch", default="main", help="remote branch to push to (a new branch of an existing repo works)"
    )
    parser.add_argument("--push", action="store_true", help="push to --remote (git uses your own credentials)")
    parser.add_argument("--zip", action="store_true", help="also write <out>.zip, e.g. to attach to a GitHub Release")
    args = parser.parse_args()
    if args.push and not args.remote:
        args.remote = default_remote()
        if not args.remote:
            sys.exit("--push needs --remote (this folder has no git `origin`)")
    prepare_repo(args.out, args.remote, args.branch)
    shards = build(download(args.cache, args.token), args.out, args.max_file_mb, args.text_rows)
    for name, files in shards.items():
        print(f"{name}: {len(files)} file(s), {sum(f.stat().st_size for f in files) / 2**20:.1f} MB")
    if args.zip:
        print("zip:", make_zip(args.out))
    changed = commit(args.out, args.remote, args.push, args.branch)
    state = "committed" if changed else "already up to date"
    print(f"{state}{' and pushed' if args.push else f' locally in {args.out}'} (branch {args.branch})")


if __name__ == "__main__":
    main()
