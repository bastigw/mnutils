# Tutorial — one of the hands-on chapters (`docs/basics/`, `data-model/`, `plotting/`, `fitting/`, `nifti/`, `matlab/`)

The genre where **the documentation is the test suite.** Every page here is executed by
`myst build --execute` (docs build), and once it carries a jupytext kernelspec it is also picked
up by the notebook test suite (`--nbmake`, already in `pyproject.toml`'s pytest `addopts`).

So every cell must run headless, deterministically, and reasonably fast. A page that only renders
but proves nothing is half done; a page that asserts but reads like a test file is also half done.

## Placement & naming

- Category dirs: `basics/`, `data-model/`, `plotting/`, `fitting/`, `nifti/`, `matlab/`.
- Lowercase snake_case. The TOC controls display order, not filename prefixes.
- `testonly_` prefix = executed by the test suite, never in the TOC, never rendered. Use for
  internal validation against ground-truth data (e.g. `tests/datasets/`).

## Skeleton

````markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

(my-topic)=
# Descriptive Title

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline

# Crisp retina output + sane default DPI for the rendered docs
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
```

One-paragraph hook: the problem this page solves, in plain language.

| Function | What it does here |
|---|---|
| [`get_mat_data_from_series()`](#mnutils.utils.data_loaders.get_mat_data_from_series) | loads the raw series used in this tutorial |
| [`GESeries`](#mnutils.GESeries.GESeries) | the object this page is about |

```{code-cell} ipython3
from mnutils import get_mat_data_from_series
```

(my-topic-load)=
## 1. Load the data

```{code-cell} ipython3
series = ...
```

(my-topic-apply)=
## 2. Apply the transform

```{code-cell} ipython3
result = ...
```

... plot the result ...

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: <what this proves>
assert ...
```
````

Imports go in a plain visible cell; wrap them in a `:::{dropdown}` only when bundled with a longer
plotting helper. `+++` splits adjacent markdown into separate cells — use it when prose after a
`:::` block would otherwise be swallowed into it.

The **functions-used table** sits under the hook, before the first cell: every MNUtils call the
page makes, linked to its API entry, one line on what it does *here* (not what it does in general
— the API page says that). Anchors are quartodoc's dotted targets, which are project-global, so a
bare `#anchor` resolves from any page: `#mnutils.fitting.AMARES.<func>` for a fitting-module call,
`#mnutils.GESeries.GESeries.<method>` for a class method. Find the exact one with
`grep -n "^(mnutils" docs/api/<module>.md` after `uv run docs-api`. Skip the table on pages that
call nothing (rare for a tutorial).

## Hidden assert cells

Recommended for any cell that demonstrates a computation. A tutorial that runs code and asserts
nothing is a doc, not a test — the checker warns about exactly that.

- Tagged `:tags: [remove-cell]`, placed **immediately after** the demonstration it verifies.
  nbmake still executes it; the site hides it.
- Opens with a `# STRICT TESTS: <what>` comment. Underscore-prefix throwaway variables.
- `np.testing.assert_allclose` / `assert_array_equal` with an `err_msg=`, plus plain `assert`s for
  metadata.

## Data

Real fixtures live under `tests/datasets/`. **Don't hardcode a relative `../..` path to them** —
this same page is executed from two different working directories: `docs/<chapter>/` when mystmd
builds it, `tests/autogen_notebooks/<chapter>/` when `uv run test-gen` converts it to a notebook
for pytest. Those are different depths from the repo root, so no single relative path satisfies
both. Anchor on `pyproject.toml` instead — copy the `_repo_root()` helper from an existing page
(e.g. `docs/basics/loading_data.md`) into your setup cell.

Prefer the smallest fixture that demonstrates the point; the gitignore blocks new data extensions
by default, so a new fixture needs an explicit whitelist entry and must stay small. Seed any
randomness used for synthetic demonstration data.

MNUtils has no synthetic-signal simulator of its own (xmris's `simulate_fid` covers that for
spectral fitting content, if a tutorial under `docs/fitting/` needs one — import it from `xmris`
rather than hand-rolling a damped sinusoid).

## Cell tags

| Tag | Rendered site | Tests (nbmake) | Use for |
|---|---|---|---|
| `remove-cell` | cell gone entirely | executed | matplotlib setup, STRICT TESTS |
| `remove-input` | output only | executed | a call whose output matters, not its call site |
| `remove-output` | input only | executed | a call the reader should type themselves |
| `hide-input` | input collapsed | executed | data-loading boilerplate |
| `hide-output` | output collapsed | executed | verbose prints / long assertion logs |
