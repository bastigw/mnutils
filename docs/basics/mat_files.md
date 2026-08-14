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

(basics-mat-files)=
# Loading a `.mat` reconstruction

`RawMRISeries` (and everything built on it — `MRSSeries`, `MRSISeries`, `MRSWashinSeries`) reads
its `recon` dict from `load_mat_file` under the hood. Scanner reconstructions come in two MATLAB
file formats depending on size and MATLAB version — plain (v5/v7, scipy-readable) and v7.3
(HDF5-based, needs `mat73`) — and `load_mat_file` hides that distinction behind one call.

| Function | What it does here |
|---|---|
| [`load_mat_file()`](#mnutils.utils.data_loaders.load_mat_file) | loads either `.mat` format, returns a plain `dict` |

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from pathlib import Path

from mnutils.utils import data_loaders as dl
from mnutils.utils import file_helpers as fh


def _repo_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


DATASETS = _repo_root() / "tests" / "datasets"
```

(basics-mat-files-newer)=
## The current format

```{code-cell} ipython3
mat_path = fh.get_mat_data_from_series(DATASETS / "LG_D19", 7)
data = dl.load_mat_file(mat_path)
print(type(data), sorted(data.keys())[:5])
print(data["spec"].shape)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: current-format .mat
assert isinstance(data, dict)
assert "bb" in data
assert data["spec"].shape[0] == 700
```

(basics-mat-files-legacy)=
## The older (v7.3) format

Older reconstructions saved as MATLAB v7.3 fail the plain `scipy.io.loadmat` path and fall back
to `mat73` transparently — same call, same return type.

```{code-cell} ipython3
legacy_data = dl.load_mat_file(DATASETS / "MRSexampleTE35.mat")
print("pyr" in legacy_data)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: legacy-format .mat
assert isinstance(legacy_data, dict)
assert "pyr" in legacy_data
```

(basics-mat-files-errors)=
## When the path isn't a real `.mat` file

```{code-cell} ipython3
:tags: [remove-cell]

import pytest

with pytest.raises(FileNotFoundError):
    dl.load_mat_file("/nonexistent/file.mat")
```

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError` | path doesn't exist | double-check the series ID / `get_mat_data_from_series` result |
| `ValueError: ...does not have a .mat extension` | wrong file passed | `load_mat_file` only accepts `.mat` |
| `RuntimeError: mat73 is required...` | v7.3 file, but `mat73` isn't installed | `mat73` is a core dependency (see `pyproject.toml`) — reinstall the env |

:::{seealso}
[Finding files in a data folder](#basics-file-discovery) covers `get_mat_data_from_series`, which
usually supplies the path passed here.
:::
