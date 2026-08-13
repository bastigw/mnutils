---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (mnutils)
  language: python
  name: python3
---

(data-model-geexam)=
# The GEExam class hierarchy

[`ExamBase`](#basics-loading-data) is the general-purpose entry point: point it at a data folder
and it classifies every series for you. Some study designs are common enough to deserve their own
subclass on top of that — one that already knows which series is the anatomical scan, which are
the pre/post-injection spectra, and how they relate to each other, so you don't have to
re-derive that bookkeeping in every notebook.

| Class | What it adds |
|---|---|
| [`ExamBase`](#mnutils.GEExam.ExamBase) | classifies series into `GESeries` subclasses; `load_series`/`load_all` |
| [`DMIExam`](#mnutils.GEExam.DMIExam) | + sorts series into `anatomical`/`all_MRS`/`all_MRSI` by folder-name convention |
| [`MS_DMIExam`](#mnutils.GEExam.MS_DMIExam) | + picks the one `MRSI` and the two `MRS` series flanking it (pre/post) |
| [`DMIinjExam`](#mnutils.GEExam.DMIinjExam) | + splits every MRS/MRSI series into pre-/post-injection lists around a wash-in series |

```{code-cell} ipython3
from pathlib import Path

import numpy as np

from mnutils.GEExam import DMIinjExam


def _repo_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


DATA_FOLDER = _repo_root() / "tests" / "datasets" / "HeVo-18" / "data"
```

(data-model-geexam-split)=
## Splitting series around an injection

`DMIinjExam` finds a wash-in series (folder name containing `MRS_washin`) and uses its series ID
as a reference point: every other MRS/MRSI series sorts into "pre" (the single one immediately
before it) or "post" (all series after it, in order). This is `_split_series` — a pure function
of a series-ID array and a reference ID, easiest to see without loading any real data.

```{code-cell} ipython3
exam = object.__new__(DMIinjExam)  # bypass __init__: _split_series only reads its arguments

pre, post = DMIinjExam._split_series(exam, np.array([3, 7, 9]), 5, "MRS")
print(pre, post)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: _split_series
# Single series before the reference, everything after is "post".
pre, post = DMIinjExam._split_series(exam, np.array([3, 7, 9]), 5, "MRS")
assert pre == 3
np.testing.assert_array_equal(post, np.array([7, 9]))

# Multiple series before the reference: the LAST one before it wins as "pre" --
# a regression case, this previously raised TypeError trying to int() a slice.
pre, post = DMIinjExam._split_series(exam, np.array([2, 3, 7, 9]), 5, "MRS")
assert pre == 3
np.testing.assert_array_equal(post, np.array([7, 9]))

# No series before the reference: "pre" is the sentinel -1, not a missing value.
pre, post = DMIinjExam._split_series(exam, np.array([7, 9]), 5, "MRS")
assert pre == -1
np.testing.assert_array_equal(post, np.array([7, 9]))

# No series after the reference: "post" is an empty array, not None.
pre, post = DMIinjExam._split_series(exam, np.array([2, 3]), 5, "MRS")
assert pre == 3
assert post.size == 0
```

:::{warning}
`pre` is a **sentinel `-1`**, not `None`, when nothing qualifies — a truthiness check (`if pre:`)
silently passes for `pre == -1` the same as any other series ID. Check `pre == -1` explicitly.
:::

(data-model-geexam-loading)=
## Loading a real exam

`DMIinjExam.load_all_series()` does the equivalent classification against real series data,
setting `pre_MRS`/`post_MRS`/`pre_MRSI`/`post_MRSI`/`washin` once loaded. The general
[`ExamBase`](#basics-loading-data) walkthrough already covers classification and loading in
depth — this page only adds the injection-specific splitting on top.

:::{seealso}
[The GESeries class hierarchy](#data-model-geseries) covers what gets loaded once `ExamBase` has
picked a series's class.
:::
