---
name: docs-page
description: Create or edit any hand-authored page in the MNUtils docs — hands-on tutorials (docs/basics/, data-model/, plotting/, fitting/, nifti/, matlab/) or contributor guides (docs/contribute/). Use when adding or restructuring a doc page, writing notebook tests for a function, thinning or consolidating existing pages, or fixing a page that renders wrong.
---

# Write an MNUtils docs page

Hand-authored pages are MyST notebooks — jupytext frontmatter plus a kernelspec — so any of them
can reach for live `code-cell`s, real output and plots. What separates the two genres is **shape,
reader, and where the cells are executed.**

## 0. Git handoff — you stage, the user commits

**Never run `git commit` yourself.** Stage the files (`git add`) and hand the change back: name
what changed, quote the commit message you would have used, and let the user read the diff and
commit it themselves. This binds in auto-accept mode too — a queued commit is still an unreviewed
commit.

**Never touch a remote without explicit confirmation.** `git push`, creating a remote branch or
repo (`gh repo create`), opening a PR — ask with `AskUserQuestion` and wait for a yes. A yes
covers that one action, not the next one.

## 1. Route by genre — read one template, not both

| | **Tutorial** | **Guide** |
|---|---|---|
| Lives in | `docs/basics/`, `data-model/`, `plotting/`, `fitting/`, `nifti/`, `matlab/` | `docs/contribute/` |
| Reader wants | to *do* the task | to contribute |
| Shape | demonstrate → assert, step by step | numbered steps + commands |
| `uv run pytest` (nbmake) | ✅ | ❌ |
| `myst build --execute` | ✅ | ✅ |
| Template | `templates/tutorial.md` | `templates/guide.md` |

`templates/explainer.md` is for a future `docs/concepts/` chapter — MNUtils doesn't have deep
"why is it shaped this way" material yet, but the template is here so a concept has somewhere to
land once one earns a permanent home (see house rule "one home per concept" below), instead of
sprawling across tutorial asides.

`templates/patterns.md` is the shared MyST pattern library — read it alongside whichever genre
template applies.

**One path never routes here:** `docs/diary/` belongs to the **`dev-diary`** skill. Hand off; do
not write an entry from here. (`check_docs.py` still checks entries — the structural rules in §2
bind every page in the tree.)

House style — motivated narrative, one home per concept, every article stands alone, the MyST
palette carries the argument — lives in **`CLAUDE.md` § "Documentation style"** and is not
restated here. One carve-out: the *motivated narrative* rule binds tutorials, not guides. A
numbered list of commands is the correct shape for `setup.md`; novelizing an install guide makes
it worse.

## 2. Rules that survive genre

Every one of these is enforced by `check_docs.py` — run it before you finish (§4). The build is
silent about all of them; the checker is not.

- **Frontmatter is exact**, and `display_name: .venv` is the frozen kernel label. If
  your local Jupyter rewrites it (to `Python 3 (ipykernel)` or similar), fix it back before you
  hand the change over. Execution is unaffected either way; `name: python3` resolves to the uv venv.
- **Nothing before `(target)=` + `# H1`.** mystmd lifts the first heading into the page title but
  only *removes* it from the body when it leads the page. Put anything first — even a hidden
  `remove-cell` — and the title renders **twice**. The setup cell always comes *after* the H1.
  Exactly one H1 per file; never restate the title as another header.
- **Explicit MyST target above every header**, kebab-case, prefixed with the page topic.
  Auto-generated slugs are numbered by *document position* — `id-1-`, `id-2-` — so inserting one
  section silently renumbers every anchor below it and breaks deep links.
- **Never link `.ipynb`.** `docs/myst.yml` excludes notebook files from the build, so the link
  resolves to `null` and dies — with no build warning. Use `[text](#explicit-target)` or a
  relative `.md` path.
- **TOC entry in `docs/myst.yml` is mandatory** (except `testonly_`). The sidebar shows the TOC
  title — keep it consistent with the H1. A page missing from the TOC never renders.
- **Stage only the `.md`.** `docs/**/*.ipynb` is gitignored; edits made in an `.ipynb` twin are
  invisible to both tests and docs until synced back (`uv run jupytext --sync <file>`).

Tone across both genres: conversational, sharp, concise, no filler — but never assuming expertise.
Background an expert would skip goes in a `:::{dropdown}`, off the main line.

## 3. Editing an existing page

Most doc work is editing, and the house rules make thinning **expected work, not scope creep**.

- Run the checker on the page *before* you start.
- Every header already carries a target, so the live risk is **renaming** one: targets are
  page-global in MyST, and a rename breaks every deep link to it. Grep first —
  `git grep -nF "#old-slug" -- docs` — and prefer adding a new section over re-keying an old one.
- Consolidating across pages is the *one home per concept* rule doing its job. Say what you moved
  in the PR body, and keep the orienting recap on the page you thinned.

## 4. Verify before finishing

```bash
uv run python .claude/skills/docs-page/check_docs.py docs/<path>/<page>.md
```

Errors are render-breaking and exit 1. Warnings are drift and exit 0. Then, for a **tutorial**:

```bash
uv run test-mnutils-gen
uv run pytest "tests/autogen_notebooks/<chapter>/<name>.ipynb" -q
# docs/<chapter>/<name>.md becomes tests/autogen_notebooks/<chapter>/<name>.ipynb
```

(`--nbmake` is already in pytest `addopts`.) A page only joins that suite once it carries a
jupytext **kernelspec** — that is the exact condition `_is_executable_page` (in
`src/mnutils/_scripts.py`) tests, so a frontmatter-less page (a bare chapter `index.md`) is
skipped rather than failing with no kernel to start. There are no `tests/*.py` files anymore —
this generated notebook run **is** the test.

`myst build --html --execute` from `docs/` is the same command a docs-deploy CI job would run —
run it locally before finishing any page with live cells. (`uv run docs-mnutils` serves a live preview and
never exits; it is for reading the site, not for checking it.)

Two pathspec traps:

- `git grep … -- 'docs/**/*.md'` **silently skips `docs/index.md`** — no intermediate directory.
  Use `-- docs`.
- `grep -r docs/` walks any gitignored `_build/` output carrying stale copies of everything.

## Checklist

<!-- excerpt:start -->
- [ ] Genre identified; the matching template and `patterns.md` read
- [ ] Frontmatter exact, `display_name: .venv`
- [ ] `(target)=` + single H1 is the very first content; a target above every header
- [ ] TOC entry in `docs/myst.yml` (unless `testonly_`); no `.ipynb` links
- [ ] `check_docs.py` passes on the page (0 errors)
- [ ] Tutorial only: notebook test run is green
- [ ] Only the `.md` staged — staged, not committed; commit message proposed to the user
- [ ] Remote actions (push, PR, remote branch) confirmed by the user first
<!-- excerpt:end -->
