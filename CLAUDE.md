# CLAUDE.md

This file provides guidance to Claude Code when working with code in the `MNUtils` repository.

`MNUtils` is a collection of utility functions for working with multi-nuclear MR data from our GE
scanners: loading raw/reconstructed data (`GEExam`, `GESeries`), NIfTI/DICOM conversion, plotting,
and spectral fitting. Fitting is delegated to [`xmris`](https://github.com/andrewendlinger/xmris)
(`xmris[fitting]`, which itself wraps pyAMARES) rather than reimplemented here.

## Documentation style

These four rules govern everything under `docs/` — explanation articles, tutorials, diary entries,
and edits to any of them. This section is their single source of truth; the `docs-page` and
`dev-diary` skills route here for the rules and restate only what binds their own genre.
*Exception: guides under `docs/contribute/` are exempt from the motivated-narrative rule — a
numbered list of commands is the right shape for a setup page.*

- **Motivated narrative, never a FAQ.** One driving question the reader already has, with every
  decision arriving as the answer to a tension they just felt. A cold "Why X?" heading makes a
  sound decision read as an assertion to accept. Concise and conversational; deep or tangential
  rationale goes in a `:::{dropdown}`, off the main line of reasoning.
- **One home per concept.** Consolidate in whichever direction fits — "where does this belong?"
  beats "who had it first." Editing and thinning existing pages is expected work, not scope creep.
  Say what you moved in the PR body. *Exception: a `docs/diary/` entry is a decision record, not a
  concept home — it owns one decision and is rewritten in place as that decision evolves; two
  entries may touch one concept when their decisions differ.*
- **Every article stands alone.** Readers arrive from search and deep links, not by walking the
  TOC. Each page must read start to finish on its own, so cross-reference rather than depend
  silently, and keep the orienting recap when you thin a page. Declare a hard prerequisite in a
  `seealso` at the top.
- **The MyST palette carries the argument.** Mermaid, admonitions, dropdowns, tables, LaTeX,
  executable `code-cell`s — reach for the one that does real work (a decision tree drawn as a
  flowchart is checkable at a glance; the same tree in prose is not). Nothing decorative. Stay
  inside the palette the docs already use.

## Significant changes get a diary entry

If a change picks between ≥2 viable approaches, adds conceptual surface (a new rule, decorator, or
module boundary — not a function that follows an existing pattern), or spans multiple PRs, invoke
the `dev-diary` skill — at the **start** (a one-screen article written from the approved plan, as
the branch's first commit) and again at the **end** (rewritten into the story of how it is now; a
"what changed from the plan" note only where the divergence teaches). When an existing entry
already tells the decision's story, propose updating that entry instead of adding a sibling.

Pass 1 is the change's **master overview** and its review gate: the plan file is right for
executing and too heavy for approving, so the entry is what gets read on the rendered site before
work starts. It never restates the plan's steps. The skill always asks before writing anything —
never decide that autonomously — and after committing the draft the turn **ends** so the user can
review the page; implementation waits for their go-ahead.

The `Dev Diary` group opens with one evergreen intro (`docs/diary/index.md`, pinned first) that
explains what the diary *is*; dated entries follow it chronologically and carry a muted
`Last edited` line rather than a status banner.

## Environment & commands

Package manager is `uv` — never use pip. Add deps with `uv add <pkg>`; sync with
`uv sync --all-extras --dev`.

- Tests: `uv run test` (regenerates notebook tests from `docs/` via `uv run test-gen`, then runs
  pytest). Regenerate notebooks only: `uv run test-gen`. There are no `tests/*.py` files —
  MNUtils's tests **are** its docs pages: any executable page under `docs/basics/`,
  `data-model/`, `plotting/`, `fitting/`, `nifti/`, `matlab/` (i.e. carrying a jupytext
  kernelspec) is converted to a notebook under `tests/autogen_notebooks/<chapter>/` (gitignored)
  and run via nbmake. Single page: `uv run pytest tests/autogen_notebooks/<chapter>/<name>.ipynb`
  after a `test-gen`.
- Lint: `uv run ruff check .` (`--fix` to auto-fix). Format: `uv run ruff format .`.
- Docs API stubs: `uv run docs-api`. Check a page renders: `myst build --html` from `docs/` —
  one-shot, exit 0 (add `--execute` to run notebooks too).
- `uv run docs` **launches a blocking preview server** (`myst start --execute`) and never exits.
  It is for a human reading the site — never put it in a verification step.

Ruff, when configured for a given file, follows NumPy docstring convention (see
`.github/copilot-instructions.md` for the current type-hinting standards, kept separate since it
targets GitHub Copilot specifically).

## Gotchas

- **MATLAB Engine version must match the local MATLAB install.** There is no way to pin one
  version project-wide across contributors' machines — install the `matlabengine` release that
  matches your MATLAB version (see README), not necessarily the one pinned under
  `[project.optional-dependencies] 2025a`.
- **`xmris[fitting]` pulls in pyAMARES.** MNUtils never reimplements fitting logic — extend
  `src/mnutils/fitting/AMARES.py` as a thin wrapper, push algorithmic changes upstream to `xmris`.
- **`hd-bet`/`torch` are optional** (`bet` extra) — brain extraction code must not be imported
  eagerly at package import time, so a bare `uv sync` (no extras) stays usable.
- **Notebook outputs are stripped via the `nbstripout` git filter**, declared in `.gitattributes`
  so it applies on every fresh clone (previously it only lived in local `.git/config`, which is
  why old notebook outputs bloated history — several-MB blobs per commit under `tests/*.ipynb`).
  If a notebook diff shows embedded image data, the filter isn't running — check
  `git config filter.nbstripout.clean`.

## Commits

Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`), matching current
history (e.g. `fix(nifti): ...`, `refactor: ...`). No enforced branch-protection rules on this repo
yet — don't assume PR-only workflow or required CI checks until they actually exist.
