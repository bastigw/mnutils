---
name: release
description: Cut an MNUtils release — run the full CI matrix via a release branch, land the bump on main through a PR, then tag the merged commit to trigger the PyPI publish. User-triggered only.
disable-model-invocation: true
---

# Release MNUtils

Guided checklist for cutting a release. The workflow, its rationale and the CI/CD diagram live in
[`docs/contribute/publishing.md`](../../../docs/contribute/publishing.md) — this skill is the
operational checklist that runs it. Publishing is automated by GitHub Actions
(`.github/workflows/ci-publish.yml`) using PyPI Trusted Publishing (OIDC) — **you never run
`uv publish` by hand**; pushing a `vX.Y.Z` tag does it.

`$ARGUMENTS` is the bump level (`patch` | `minor` | `major`) or an explicit version (`1.3.0`). If
empty, ask which.

Pushing branches and tags triggers CI and an **irreversible** PyPI publish — a version number,
once uploaded, can never be reused even after a delete. **Confirm with the user before any push**,
and stop if a step fails.

**Never bump the version until the full matrix is green.** The release branch runs first,
*unbumped*; the bump lands only once CI passes. That is the whole point of separating CI from CD —
it avoids the "bump → push → CI fails → bump again" cycle.

```{note}
MNUtils has **no enforced branch protection** (`CLAUDE.md` § Commits). Nothing in the repo will
stop a direct push to `main` — the pull request in step 4 is a deliberate convention, not a gate
GitHub enforces for you. Follow it anyway: it is the only place the five fast checks run against
the merge result.
```

## Checklist

1. **Preconditions.** On `main`, working tree clean, up to date with `origin`. Confirm the fast CI
   on `main` is green (`gh run list --branch main --limit 3`).

2. **Determine the target version.** Show the current version (`uv version`) and the target implied
   by `$ARGUMENTS` (e.g. `1.3.0`). Confirm with the user — but **do not bump yet**. If the
   changelog work in step 3 turns up a `**Breaking**` bullet and the agreed bump is not major, stop
   and raise it.

3. **Run the full matrix — and write the changelog while it runs.** Push an *unbumped*
   `release/vX.Y.Z` branch. That branch name is what `ci-publish.yml` triggers on, so this fires
   the full Ubuntu/Windows/macOS × Python 3.12/3.13/3.14 matrix. Watch it with `gh run watch` /
   `gh run list`. If a leg fails, fix it on the release branch and push again until green. **Every
   leg blocks, macOS included** — it was `continue-on-error` only while official `pyamares` hard-
   required `hlsvdpro` (no arm64 wheel, no sdist); the `fitting` extra now comes via
   `xmris[fitting]`, which depends on `pyamares-xmris`, whose platform marker skips it.

   The matrix takes ~15 minutes and nothing here depends on it. Spend them on the changelog:
   **invoke the `changelog` skill** for `vX.Y.Z`. Its section in `docs/changelog.md` rides the bump
   commit in step 4, so the entry and the version land together in one pull request.

4. **Bump and merge.** Once the matrix is green, bump on the release branch and land it through a
   PR:

   ```bash
   uv version --bump <level>                 # or set the explicit version
   git commit -am "chore: bump version to X.Y.Z"
   git push origin release/vX.Y.Z
   gh pr create --base main --title "chore: bump version to X.Y.Z"
   ```

   The changelog section from step 3 belongs in this commit, and `uv.lock` records the new version
   too — stage it. Drive the PR's five fast checks green (docs style, lint, bare install, tests on
   3.12 and 3.14); the user merges it (squash). Expect **two** CI cycles here: pushing the bump
   re-triggers the full matrix on `release/**` (harmless, and it does test the bumped tree), while
   the five fast checks are the actual merge gate — do not wait on the matrix to merge.

5. **Tag & publish.** Tag the *merged* commit on `main` — **never the release branch**. PRs are
   squash-merged, so a tag cut before the merge would sit on a commit that never enters `main`'s
   history, leaving `git describe` on `main` blind to the release:

   ```bash
   git checkout main && git pull
   git log --oneline -1                          # MUST be the bump commit -- see below
   uv version                                    # MUST print X.Y.Z
   grep -n "^## vX.Y.Z" docs/changelog.md        # MUST exist, and MUST NOT say "unreleased"
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

   The changelog grep guards the one failure this workflow cannot undo cheaply: a tag pushed for a
   version the changelog never described. Catch it here, before the tag — `ci-publish.yml`
   deliberately knows nothing about the changelog, because a check that fires *after* the tag would
   leave you deleting and re-pushing one.

   Check that tip before tagging: anything merged into `main` between the bump and the tag ships in
   this release without ever having seen the pre-flight matrix. If something did land, decide
   deliberately — either ship it (the tag re-runs the full matrix anyway) or tag the bump commit
   explicitly by SHA.

   The tag re-runs the full matrix and then triggers `uv build --no-sources` + `uv publish` to PyPI
   via OIDC. Watch that run to confirm the publish succeeded.

6. **Announce it on GitHub.** Once the publish run is green and the version is live on PyPI, create
   the GitHub Release:

   ```bash
   gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes.md>
   ```

   Write `notes.md` to a scratch path from the `vX.Y.Z` section you produced in step 3, converting
   the MyST target links (`[Displaying image arrays of any shape](#plotting-images)`) to full
   `https://bastigw.github.io/mnutils/…` URLs — GitHub cannot resolve them. Nothing in CI does
   this: creating the release by hand keeps `contents: write` off the job that publishes to PyPI,
   and means the announcement only ever exists for a version that actually shipped.

7. **Wrap up.** Confirm the new version is live on PyPI (`pip index versions mnutils`, or the
   project page) and that `main` is in the expected state. Delete the release branch. Summarize
   what shipped.

## Checklist

<!-- excerpt:start -->
- [ ] Started from a clean, up-to-date `main` with fast CI green
- [ ] Target version agreed **before** any bump; a `**Breaking**` bullet matches a major bump
- [ ] Unbumped `release/vX.Y.Z` branch pushed; full matrix green on **every** leg, macOS included
- [ ] Changelog section for `vX.Y.Z` written by the `changelog` skill and riding the bump commit
- [ ] Bump landed through a PR with its five fast checks green, squash-merged by the user
- [ ] Tag cut on the **merged** commit on `main`, after `uv version` and the `unreleased` grep both pass
- [ ] Publish run watched to green; version confirmed live on PyPI
- [ ] GitHub Release created with MyST targets rewritten to full docs URLs; release branch deleted
<!-- excerpt:end -->
