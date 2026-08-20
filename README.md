# MNUtils

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

```bash
uv add git+https://github.com/bastigw/mnutils.git
```

Optional extras:

```bash
uv add "mnutils[bet] @ git+https://github.com/bastigw/mnutils.git"     # brain extraction (hd-bet, torch)
uv add "mnutils[2025a] @ git+https://github.com/bastigw/mnutils.git"   # MATLAB Engine pinned to MATLAB 2025a
```

`2025a` pins `matlabengine==25.1.*`. If your local MATLAB install is a different version, install
the matching `matlabengine` release yourself instead (see [MATLAB Engine
Dependency](#matlab-engine-dependency) below) — there's no way to pin one version project-wide
across contributors' machines.

## Usage

```python
from mnutils.utils.file_helpers import get_mat_data_from_series

# Example: Get raw data path from a specific series in a dataset
BASE_PATH = "data/HeVo-11"

raw_data_path = get_mat_data_from_series(BASE_PATH, 29)
print(raw_data_path)
```

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
