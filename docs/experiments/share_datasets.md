# Making the datasets reachable from the analysis sandbox (GitHub branch)

## One command

On a machine that can reach huggingface.co, from the project folder (needs push access to its GitHub `origin`):

```bash
make datasets-branch                       # creates the branch `datasets` of this repository, or updates it
make datasets-branch BRANCH=data TEXT_ROWS=20000 REMOTE=https://github.com/<user>/<repo>.git     # options
```

It downloads `ffurfaro/keep-it-simple` and `ffurfaro/keep-it-simple-multimodal`, re-shards them below GitHub's 100 MB file limit
into `keep-it-simple/` and `keep-it-simple-multimodal/`, and pushes them as a branch with its own history. Rerunning it clones
the branch, replaces the content, commits only if something changed and pushes a fast-forward (unchanged data gives "already up
to date"). Defaults: remote = `origin` of this project, branch = `datasets`, all rows (about 546 MB; `TEXT_ROWS=20000` keeps
about 16 MB of text, which is enough for the checks). `.datasets-branch/` (the working copy) and `data/` are git-ignored; git uses
your own credentials, set `HF_TOKEN` if the Hub repos are private.

## Import the datasets from anywhere

```python
from kairos.hf_data import load_github_datasets

ds = load_github_datasets("fabienfrfr/Kairos@datasets")   # downloads the branch zip once into ~/.cache/kairos/datasets
ds["text"], ds["multimodal"]                              # datasets.Dataset objects with the columns of the Hub versions
```

In a training config (`configs/pretrain_sample_github.yaml` does this):

```yaml
hf_data:
  github: fabienfrfr/Kairos@datasets      # or archive_url: <zip url, e.g. a GitHub Release asset>
  text_pct: 1
```

Or to local folders: `python scripts/fetch_datasets.py --from-github fabienfrfr/Kairos@datasets`. The repository must be public
(the download uses `codeload.github.com` without credentials); pass `refresh: true` (or `refresh=True`) to download again after an
update, and explicit `text_repo` / `multimodal_repo` entries override the branch. Any environment with the project installed
(`pip install -e .`) can use it, so it also works where the Hub is blocked.

## Alternative: a separate repository, step by step

The sandbox that ran the experiments cannot reach `huggingface.co` but can reach GitHub (`github.com`,
`codeload.github.com`, release assets). `scripts/mirror_datasets_to_github.py` copies the two Hub datasets into a git
repository that stays under GitHub's limits; `scripts/fetch_datasets.py --from-github` reads it back. Nothing is pushed
unless you pass `--push`, and the script never handles credentials.

## Before you publish

Both dataset cards state the MIT licence, but the multimodal dataset is built from third-party sources (Flickr8k, AudioCaps,
Molmo2-VideoCapQA, nuScenes-mini, MotionSense, PixelBytes-OptimalControl, COCO detection; see
`scripts/pretrain/readme_multimodal.md`): a public repository republishes the data, so keep their terms in mind. A private
repository cannot be read from the sandbox without sharing a credential, which is not recommended.

## Sizes, and why plain git is enough (no Git LFS)

From the dataset cards: `keep-it-simple` 640 646 rows, 513 MB; `keep-it-simple-multimodal` 6 336 rows, 33.2 MB. GitHub only
blocks single files above 100 MiB, so parquet shards of at most 90 MB go into a normal repository. Git LFS stores large files
outside git behind small pointer files (free plan: 10 GiB of storage and 10 GiB of downloads per month); it is not needed here and
is not reachable from the analysis sandbox. For the checks only a sample of the text is needed (`--text-rows 20000`, about
16 MB; omit it to copy all 513 MB).

## A branch of an existing repository

`--branch datasets` pushes the data as a separate branch of an existing repository (for example the Kairos repository). The
history of that branch is independent of `main`, so `git clone --single-branch` of `main` stays light; a plain `git clone`
downloads every branch, so a separate repository is cleaner. Fetch with `--from-github <user>/<repo>@datasets`.

## On your PC (with access to huggingface.co)

1. Create an empty repository on GitHub (for example `kairos-data`) and make sure `git push` works from your machine.
2. In the project folder:

   ```bash
   uv sync
   # data on its own branch of an existing repository (needs push access; the branch must not exist yet):
   uv run python scripts/mirror_datasets_to_github.py --out ../kairos-data --text-rows 20000 --push --branch datasets \
       --remote https://github.com/<user>/<repo>.git
   # or in a separate repository:
   make mirror-datasets OUT=../kairos-data TEXT_ROWS=20000          # download, re-shard, commit locally
   cd ../kairos-data && git remote add origin https://github.com/<user>/kairos-data.git && git push -u origin main
   # or in one step:
   uv run python scripts/mirror_datasets_to_github.py --out ../kairos-data --text-rows 20000 --push \
       --remote https://github.com/<user>/kairos-data.git
   ```

3. Give the repository name (and branch if any), for example `<user>/kairos-data` or `<user>/<repo>@datasets`. The sandbox then runs
   `python scripts/fetch_datasets.py --from-github <user>/kairos-data` and
   `python scripts/check_pretraining_readiness.py data|train --source hf` on the real rows.

## What the script does

- Downloads `ffurfaro/keep-it-simple` and `ffurfaro/keep-it-simple-multimodal` with `huggingface_hub` (set `HF_TOKEN` if
  they are private or gated).
- Rewrites every parquet file into shards of at most `--max-file-mb` (default 90, GitHub refuses 100 MB), keeping every row
  of the multimodal dataset and, with `--text-rows N`, the first N rows of the text dataset (the sample config uses 1% of it).
- Refuses to continue if any file is still above 100 MB, and warns above 1 GB per repository.
- Writes a `README.md` with the provenance, then `git init`, `git add`, `git commit` (a local commit named `kairos`).

## Larger data, other hosts

- For more than about 1 GB, use `--zip` (writes `<out>.zip`), create a GitHub Release with it
  (`gh release create v1 ../kairos-data.zip`), and fetch with
  `python scripts/fetch_datasets.py --archive-url <asset url>`.
- Git LFS is not usable from the sandbox (`media.githubusercontent.com` does not resolve there).
- Not verified: the download from the Hub (the sandbox has no access); the re-sharding, limits check, commit, archive
  extraction (including rejection of unsafe paths) and the GitHub URL format are covered by
  `tests/test_mirror_datasets_script.py`, and the `codeload.github.com` download path was checked on a public repository.
