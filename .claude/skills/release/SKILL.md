---
name: release
description: Cut an MNUtils release — changelog and version bump on a release/X.Y.Z branch (which runs the full CI matrix), land it through a PR, then tag the merged commit to publish to PyPI and create the GitHub Release. User-triggered only.
disable-model-invocation: true
---

# Release MNUtils

Operational checklist for cutting a release. Publishing is automated by
`.github/workflows/ci-publish.yml`: pushing a `vX.Y.Z` tag runs the full matrix, publishes to PyPI
via Trusted Publishing (OIDC), then creates the GitHub Release from the matching `CHANGELOG.md`
section. **Never run `uv publish` or `gh release create` by hand.**

`$ARGUMENTS` is the bump level (`patch` | `minor` | `major`) or an explicit version (`1.3.0`). If
empty, ask which.

Pushing a branch, opening a PR and pushing a tag are all outward-facing, and a PyPI version number
can never be reused once uploaded. **Confirm with the user before each push, PR and tag** — a yes
covers that one action, not the next — and stop if a step fails. Never push to `main`.

## Checklist

1. **Preconditions.** On `main`, clean tree, up to date with `origin` (`git fetch && git status -sb`).
   Everything meant for the release is already merged. Fast CI on `main` is green
   (`gh run list --branch main --limit 3`).

2. **Pick the version.** Show `uv version` and the target. Read `## [Unreleased]` in
   `CHANGELOG.md`: a `### Changed (breaking)` section means at least a **minor** bump (this project
   is 1.x and has shipped breaking changes in minors); if the agreed level is `patch`, stop and
   raise it.

3. **Prepare the release commit on `release/X.Y.Z`.** Branch off `main`, then:
   - `CHANGELOG.md`: rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and leave a fresh empty
     `## [Unreleased]` above it. CI extracts the release notes by that exact heading
     (`## [X.Y.Z]`), so do not reformat it.
   - `uv version X.Y.Z` (updates `pyproject.toml` **and** `uv.lock`; if it is unavailable, edit
     `pyproject.toml` and run `uv lock`). Stage all three files.
   - Commit as `chore(release): X.Y.Z`.

4. **Verify locally.** `uv run ruff check .`, `uv run ruff format --check .` and
   `uv run test-mnutils`. If `ruff format --check` fails, stop and let the user fix it
   (`CLAUDE.md` § Environment) — do not auto-apply `ruff format .`.

5. **Push and PR** (after confirmation). The `release/**` branch name triggers the full
   Ubuntu/Windows/macOS matrix; every leg blocks, macOS included. Open the PR to `main` titled
   `chore(release): X.Y.Z`; its body gives the headline changes and breaking notes, and `Closes #N`
   for issues it resolves. Watch with `gh pr checks` / `gh run watch`. The user merges (squash).

6. **Tag the merged commit** (after confirmation). Always on `main`, never the release branch —
   a squash merge means a tag on the branch would sit on a commit `main` never contains:

   ```bash
   git switch main && git pull --ff-only
   git log --oneline -1                      # MUST be the merged chore(release): X.Y.Z commit
   grep -m1 '^version' pyproject.toml        # MUST print X.Y.Z
   grep -n "^## \[X.Y.Z\]" CHANGELOG.md      # MUST exist
   git tag vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   If another commit landed on `main` after the bump, decide deliberately: ship it (the tag re-runs
   the full matrix) or tag the bump commit by SHA.

7. **Watch the publish run** (`gh run list --workflow ci-publish.yml`) until the publish and
   GitHub Release jobs are green, then confirm the version on PyPI
   (`uv pip index versions mnutils` or the project page).

8. **Clean up.** Delete the merged remote `release/X.Y.Z` branch and the local one (confirm first;
   squash merges need `git branch -D`). Summarize what shipped.

## Checklist

<!-- excerpt:start -->
- [ ] Started from a clean, up-to-date `main` with fast CI green
- [ ] Version agreed; a `Changed (breaking)` section matches at least a minor bump
- [ ] `CHANGELOG.md` heading renamed to `## [X.Y.Z] - date`, `pyproject.toml` and `uv.lock` bumped
- [ ] Ruff and `test-mnutils` pass locally
- [ ] Push, PR and tag each confirmed by the user; full matrix green on **every** leg
- [ ] Tag cut on the **merged** commit on `main`, after the version and changelog greps pass
- [ ] Publish and GitHub Release jobs watched to green; version live on PyPI
- [ ] Release branch deleted (remote and local)
<!-- excerpt:end -->
