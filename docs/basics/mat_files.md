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
| [`read_mat_header()`](#mnutils.utils.data_loaders.read_mat_header) | reads only the header fields MNUtils uses, fast on either format |

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from mnutils.utils import data_loaders as dl
from mnutils.utils import file_helpers as fh
from mnutils.testing import build_fake_exam

mrsi_missing_series_exam = build_fake_exam("mrsi_missing_series_exam")
```

(basics-mat-files-newer)=
## The current format

```{code-cell} ipython3
mat_path = fh.get_mat_data_from_series(mrsi_missing_series_exam, 7)
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
legacy_data = dl.load_mat_file(build_fake_exam("legacy_matlab73_example"))
print("pyr" in legacy_data)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: legacy-format .mat
assert isinstance(legacy_data, dict)
assert "pyr" in legacy_data
```

(basics-mat-files-header)=
## Reading just the header

Series identification and the `RawMRISeries` properties (`bandwidth`, `centre_frequency`, …) need a
few dozen numbers from the scanner header `h`, not the whole file. That matters on v7.3: `h` alone
is ~1600 tiny HDF5 datasets, and a full `load_mat_file` spends ~0.45 s per file opening them —
about 9 s for an exam with 20 raw series, again for every object built from the same file.
`read_mat_header` opens only the allowlisted fields (`HEADER_FIELDS`), ~20x faster, and returns
the same nested dict you would get from `load_mat_file(path)["h"]`, pruned.

```{code-cell} ipython3
header = dl.read_mat_header(mat_path)
print(header["series"]["se_desc"], "|", sorted(header))
print(len(header["rdb_hdr"]), "of", len(dl.HEADER_FIELDS), "allowlisted fields are under rdb_hdr")
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: header-only read
full_header = dl.load_mat_file(mat_path)["h"]
assert header["series"]["se_desc"] == full_header["series"]["se_desc"]
assert set(header) <= set(full_header)
only_bb = dl.load_mat_file(mat_path, variable_names=["bb"])
assert "bb" in only_bb and "h" not in only_bb
```

`RawMRISeries.header` is exactly this. Its `recon` dict (everything else in the file, `h` pruned to
the same fields) loads on first access. To read a field that isn't allowlisted, pass
`fields=[...]` here or use `load_mat_file` for the full header; a new `RawMRISeries` header
property needs its field added to `HEADER_FIELDS`.

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
