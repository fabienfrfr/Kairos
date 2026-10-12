# Releasing (maintainers)

* **Normal path**: merge to `main`; `release.yml` tags from conventional commits, `release.yml` then dispatches
  `publish.yml` on that tag (tags pushed with `GITHUB_TOKEN` do not trigger `on: push` workflows), which builds
  and uploads to PyPI via OIDC. A green release job without a new tag means no `feat`/`fix` commit was found.
* **Manual fallback**: `git fetch --tags`, copy `.env.example` to `.env`, set `UV_PUBLISH_TOKEN`, then run
  `make publish v0.X.X`. It builds that tag in a temporary git worktree with the version pinned to the tag, so
  no `.dev` version is uploaded (current branch and local changes are ignored). Without an argument, it uses
  the tag exactly on HEAD.
* If CI already published that tag, PyPI rejects the duplicate upload (versions are immutable).
