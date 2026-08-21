<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="docs/assets/logo/mnutils-banner-dark.svg">
    <img src="docs/assets/logo/mnutils-banner-light.svg" alt="MNUtils" width="420">
  </picture>
</p>

[![PyPI](https://img.shields.io/pypi/v/mnutils.svg)](https://pypi.org/project/mnutils/)
[![Python](https://img.shields.io/pypi/pyversions/mnutils.svg)](https://pypi.org/project/mnutils/)
[![Tests](https://github.com/bastigw/mnutils/actions/workflows/ci-fast.yml/badge.svg)](https://github.com/bastigw/mnutils/actions/workflows/ci-fast.yml)
[![Docs](https://github.com/bastigw/mnutils/actions/workflows/deploy.yml/badge.svg)](https://bastigw.github.io/mnutils/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

MNUtils is a collection of utility functions for working with multi-nuclear MR data from our GE
scanners.

## Features

- **Data loading** — read GE raw/reconstructed data via `GEExam`/`GESeries`.
- **NIfTI/DICOM conversion** — convert scanner output to NIfTI.
- **Plotting** — spectral and slice plotting utilities.
- **Spectral fitting** — AMARES fitting delegated to [`xmris`](https://github.com/andrewendlinger/xmris)
  (`xmris[fitting]`, which wraps pyAMARES).

See the [documentation](https://bastigw.github.io/mnutils/) for the full basics, data model,
plotting, fitting, NIfTI, and MATLAB-bridge guides.

## Installation

MNUtils is on PyPI:

```bash
uv add mnutils
```

Optional extras:

```bash
uv add "mnutils[bet]"     # brain extraction (hd-bet, torch)
uv add "mnutils[2025a]"   # MATLAB Engine pinned to MATLAB 2025a
```

To track the development version instead:

```bash
uv add git+https://github.com/bastigw/mnutils.git
```

`2025a` pins `matlabengine==25.1.*`. If your local MATLAB install is a different version, install
the matching `matlabengine` release yourself instead (see [MATLAB Engine
Dependency](#matlab-engine-dependency) below) — there's no way to pin one version project-wide
across contributors' machines.

## Usage

The main entry point is `ExamBase`: point it at a GE exam's data folder and it classifies every
series for you — no need to know in advance which folder is an anatomical scan and which is a
spectrum.

```python
from mnutils.GEExam import ExamBase

exam = ExamBase("data/HeVo-11")  # prints an overview of every series it found

exam.exam_overview.head()  # same overview as a DataFrame
exam.series_dict[6]  # the GESeries subclass ExamBase picked for series 6 — nothing loaded yet
```

Load the series you actually need and analyse it — everything about it (spectrum, header fields,
scan time) lives on the returned object:

```python
exam.load_series(6)
mrs = exam[6]

print(mrs.protocol_name, mrs.spec.dims, mrs.spec.shape)
avg = mrs.avg_spec  # average over repetitions
```

`load_series` also takes a list, and `load_all` loads everything `ExamBase` classified. If you
already know a series's type, construct it directly (e.g. `MRSSeries(DATA_FOLDER, 6)`) and skip
the exam-level overview entirely.

From there, spectral fitting is handled by [**xmris**](https://github.com/andrewendlinger/xmris)
— MNUtils deliberately doesn't reimplement fitting, it wraps xmris's AMARES implementation
(itself built on pyAMARES) so a series's spectrum can go straight into a fit:

```bash
uv add "xmris[fitting]"
```

See the [fitting docs](https://bastigw.github.io/mnutils/fitting/) for the full workflow, and the
[loading data](https://bastigw.github.io/mnutils/basics/loading_data/) and
[GEExam](https://bastigw.github.io/mnutils/data-model/geexam/) tutorials for more on exams and
series. Need a raw `.mat` file's path directly instead of a loaded object?
`mnutils.utils.file_helpers.get_mat_data_from_series(base_folder, series_id)` returns it.

## Requirements

- Python 3.12+

## MATLAB Engine Dependency

To use MNUtils, you need to install the MATLAB Engine for Python. The version of the MATLAB Engine
you require depends on your local MATLAB version. Ensure you install the correct version that
matches your MATLAB installation.

You can find the appropriate MATLAB Engine version for your setup here: [MATLAB Engine for
Python](https://pypi.org/project/matlabengine/#history).

To install the MATLAB Engine in your project, use the following command, replacing `<version>`
with the correct version for your MATLAB:

```bash
uv add "matlabengine==<version>"
```

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the [contributor docs](docs/contribute/) for setup,
tests, docs, and lint workflow.

## Contributions and Suggestions

I'm very open to suggestions and improvements! If you have ideas or want to share your own
utilities, feel free to reach out or submit a pull request.

## License

BSD-3-Clause — see [`LICENSE`](LICENSE).
