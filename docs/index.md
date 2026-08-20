(home)=
# Welcome to MNUtils

**MNUtils** is a collection of utility functions for working with multi-nuclear MR data from our
GE scanners: loading raw and reconstructed data, converting to NIfTI, plotting, and spectral
fitting.

At its core sit two classes, `GEExam` and `GESeries`, wrapping a single scan session and one
series within it, so the rest of the toolbox — plotting, NIfTI conversion, fitting — takes a
`GESeries` in and gives you back something usable, instead of another positional-array puzzle.

---

(home-quick-start)=
## Quick start

```python
from mnutils.GEExam import ExamBase

# ExamBase scans DATA_FOLDER and figures out which series are which
exam = ExamBase(BASE_FOLDER="data/HeVo-18")
exam.load_series(6)  # loads series 6 as the right GESeries subclass automatically

mrs_series = exam[6]
print(mrs_series.protocol_name, mrs_series.spec.shape)
```

The full walkthrough — including what each `GESeries` subclass gives you — lives in
[Loading a series and an exam](#basics-loading-data).

Fitting has its own path, delegated to [`xmris`](https://github.com/andrewendlinger/xmris) and
pyAMARES — see [Fitting](#fitting-index) once that chapter has content.

(home-install)=
## Install

```bash
uv add mnutils
```

Requires Python 3.12+. Installing the MATLAB Engine for Python separately is required for any
function that calls into MATLAB — see [MATLAB bridge](#matlab-index).

(home-next)=
## Where to go next

| You want to... | Go to |
|---|---|
| Load and inspect a scan | [Basics](#basics-index) |
| Understand `GEExam`/`GESeries` | [Data model](#data-model-index) |
| Plot images or spectra | [Plotting](#plotting-index) |
| Fit a spectrum | [Fitting](#fitting-index) |
| Convert to NIfTI, run brain extraction | [NIfTI & imaging utilities](#nifti-index) |
| Call into MATLAB | [MATLAB bridge](#matlab-index) |
| Contribute a change | [Contribute](#contribute-home) |
