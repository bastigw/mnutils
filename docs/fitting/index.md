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

(fitting-index)=

# Fitting spectra

> **I have a series loaded and want peak amplitudes out of it — one averaged spectrum, or a map
> across an MRSI grid.**

MNUtils does not implement AMARES. [`xmris`](https://github.com/andrewendlinger/xmris) does (on top
of pyAMARES), and a series hands its FIDs straight to it: the series knows its own
`reference_frequency` and `carrier_ppm`, so all you supply is the **prior knowledge** — which peaks
to look for. Every fit returns the `xr.Dataset` that `xmris` produces, unchanged.

| Method                                                                   | Fits                                      |
| ------------------------------------------------------------------------ | ----------------------------------------- |
| [`MRSSeries.fit_average_fid()`](#mnutils.GESeries.MRSSeries.fit_average_fid) | the averaged FID of a single-voxel series |
| [`MRSISeries.fit_single_voxel()`](#mnutils.GESeries.MRSISeries.fit_single_voxel) | one voxel of an MRSI grid                 |
| [`MRSISeries.fit_all_voxels()`](#mnutils.GESeries.MRSISeries.fit_all_voxels) | every voxel above an SNR threshold        |

```{code-cell} ipython3
:tags: [remove-cell]

from loguru import logger

logger.remove()
```

```{code-cell} ipython3
from mnutils.GESeries import MRSISeries, MRSSeries
from mnutils.testing import build_fake_exam

DATA_FOLDER = build_fake_exam("brain_mrs_mrsi_exam") / "data"
```

(fitting-index-prior)=

## 1. Say which peaks to look for

A peak is a name plus a starting guess; `chem_shift` is in absolute ppm. The same mapping is used
for every fit below.

```{code-cell} ipython3
prior_knowledge = {
    "PCr": {"amplitude": 1, "chem_shift": 0.0, "linewidth": 20},
    "gATP": {"amplitude": 1, "chem_shift": -2.5, "linewidth": 30},
}
```

:::{seealso}
`xmris.fitting.build_prior_knowledge` documents every key a peak accepts (bounds, phase, lineshape),
and a prior-knowledge CSV file path works in place of the mapping.
:::

(fitting-index-average)=

## 2. Fit one averaged spectrum

Amplitudes are only comparable between series if they are scaled the same way, so pass a shared
`normalisation_factor`; without one the series scales by its own maximum and warns.

```{code-cell} ipython3
mrs = MRSSeries(DATA_FOLDER, 7)
fit = mrs.fit_average_fid(prior_knowledge, normalisation_factor=1.0)
fit[["amplitude", "chem_shift", "linewidth", "fit_status"]]
```

`fit_status` is `0` for a fitted spectrum, `1` for no signal and `2` for a failed fit, so a failure
is never mistaken for a zero amplitude.

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: fit_average_fid
assert int(fit["fit_status"]) == 0
assert list(fit["metabolite"].values) == ["PCr", "gATP"]
```

(fitting-index-map)=

## 3. Fit a whole MRSI grid

`fit_all_voxels` fits only voxels whose SNR clears `snr_threshold`; the rest stay NaN on the full
`(i, j, k)` grid. `testing_mode=True` limits it to the first 100 such voxels, which is plenty for
trying a prior-knowledge file out.

```{code-cell} ipython3
mrsi = MRSISeries(DATA_FOLDER, 8)
grid_fit = mrsi.fit_all_voxels(prior_knowledge, testing_mode=True)
grid_fit["amplitude"].sel(metabolite="PCr").dims
```

The result is also summarised as plain arrays, ready for `visualize_fitted_metabolite_map`:

```{code-cell} ipython3
list(mrsi.fitted_metabolite_maps), list(mrsi.goodness_of_fit_maps)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: fit_all_voxels
import numpy as np

assert grid_fit["amplitude"].sel(metabolite="PCr").dims == ("i", "j", "k")
assert np.isfinite(mrsi.fitted_metabolite_maps["PCr"]).sum() == 100
assert mrsi.fit_results is grid_fit
single = mrsi.fit_single_voxel(10, 10, 8, prior_knowledge)
assert int(single["fit_status"]) == 0
```
