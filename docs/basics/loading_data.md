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

(basics-loading-data)=

# Loading a series and an exam

A GE exam folder is a pile of numbered series folders — DICOM here, a raw `.mat` recon there,
sometimes both, sometimes neither. Loading one by hand means knowing in advance whether series 6
is an anatomical scan or a spectrum, and picking the matching `GESeries` subclass yourself. `GEExam`
exists so you don't have to: point it at the exam's data folder and it works that out for you.

| Function                                   | What it does here                                           |
| ------------------------------------------ | ----------------------------------------------------------- |
| [`ExamBase`](#mnutils.GEExam.ExamBase)     | scans a data folder and classifies every series             |
| [`MRSSeries`](#mnutils.GESeries.MRSSeries) | loaded directly, for when you already know what a series is |

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from pathlib import Path

from mnutils.GEExam import ExamBase
from mnutils.GESeries import MRSSeries


def _repo_root(start: Path = Path.cwd()) -> Path:
    # Anchor on pyproject.toml rather than a relative "../.." count: this page
    # is executed from two different working directories -- docs/basics/ when
    # mystmd builds it, tests/autogen_notebooks/basics/ when nbmake runs the
    # notebook `uv run test-gen` generates from it -- so no single relative
    # path satisfies both.
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


DATA_FOLDER = _repo_root() / "tests" / "datasets" / "HeVo-18"
```

(basics-loading-data-exam)=

## 1. Point `ExamBase` at the exam

`BASE_FOLDER` is the exam's top-level folder; `ExamBase` looks for a `data/` subfolder inside it
by default (override with `DATA_FOLDER=` if your layout differs). On init it reads every series
folder's DICOM/exam-data presence and prints an overview — useful on its own when you've
forgotten what's in a dataset.

```{code-cell} ipython3
exam = ExamBase(DATA_FOLDER)
```

That overview is also available as a DataFrame, and `series_dict` shows the classification
`ExamBase` settled on _before_ loading anything — nothing here has touched a `.mat` file or
DICOM pixel data yet.

```{code-cell} ipython3
exam.exam_overview.head()
```

```{code-cell} ipython3
exam.series_dict[6], exam.series_dict[8]
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: classification happens without loading data
from mnutils.GESeries import MRISeries, MRSISeries

assert exam.series_dict[6] is MRSSeries  # single-voxel spectrum
assert exam.series_dict[8] is MRSISeries  # spectroscopic-imaging grid
assert 2 in exam.series_dict and exam.series_dict[2] is MRISeries  # anatomical
assert exam.all == {}  # nothing loaded yet
```

(basics-loading-data-series)=

## 2. Load one series

`load_series` instantiates the right class and stores it on `exam.all`, keyed by series ID.
Everything about a series you'd normally want — its spectrum, its header fields, its scan time —
comes from that one object afterward.

```{code-cell} ipython3
exam.load_series(6)
mrs = exam[6]
print(type(mrs).__name__, mrs.protocol_name)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: load_series
assert isinstance(mrs, MRSSeries)
assert mrs.SERIES_ID == 6
assert exam[6] is mrs  # exam[id] is exam.all[id], not a fresh load
```

`GEExam` picked `MRSSeries`, so the spectrum comes back with an `averages` dimension — one
spectrum per repetition. Averaging over it is the usual first step:

```{code-cell} ipython3
print(mrs.spec.dims, mrs.spec.shape)
avg = mrs.avg_spec
print(avg.dims, avg.shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: averaging collapses the averages dimension
assert avg.dims == ("chemical_shift",)
assert avg.shape[0] == mrs.spec.shape[1]
```

(basics-loading-data-direct)=

## 3. Or skip the exam and load a series directly

If you already know a series is single-voxel spectroscopy, you can construct `MRSSeries` (or any
other `GESeries` subclass — see [the class hierarchy](#data-model-geseries)) directly with the
same `(DATA_FOLDER, SERIES_ID)` signature `ExamBase` uses internally. This is what you'd reach
for in a one-off script where pulling in the whole exam overview is more than you need.

```{code-cell} ipython3
mrs_direct = MRSSeries(DATA_FOLDER / "data", 6)
print(mrs_direct.spec.equals(mrs.spec))
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: direct construction agrees with the exam-mediated load
assert mrs_direct.spec.equals(mrs.spec)
```

(basics-loading-data-multiple)=

## 4. Load several series, or all of them

`load_series` also accepts a list; `load_all` loads everything `ExamBase` classified. For an exam
with several MRSI grids this can take a while — load only what you need.

```{code-cell} ipython3
exam.load_series([6, 8])
print(sorted(exam.all.keys()))
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: load_series accepts a list and is additive
assert set(exam.all.keys()) == {6, 8}
assert isinstance(exam[8], MRSISeries)
```

:::{seealso}
[The GESeries class hierarchy](#data-model-geseries) covers what each subclass adds and why —
this page only covers getting one loaded.
:::
