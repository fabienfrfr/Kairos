# Releasing (maintainers)

* **Normal path**: merge to `main`; `release.yml` tags from conventional commits, `publish.yml` builds and
  uploads to PyPI via OIDC on the `v*.*.*` tag.
* **Manual fallback**: check out the release tag on a clean tree, copy `.env.example` to `.env`, set
  `UV_PUBLISH_TOKEN`, run `make publish`. It refuses untagged or dirty commits and pins the version to the tag,
  so no `.dev` version is uploaded.
