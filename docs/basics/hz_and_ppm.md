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

(basics-hz-and-ppm)=
# Hz and ppm axes

A spectrum's x-axis is just its point index until something converts it to something a person
can read off a plot: Hz (a frequency offset, symmetric around zero) or ppm (that offset scaled
by the scanner's reference frequency, so it means the same thing across field strengths and
nuclei). `calculate_hz_axis`/`calculate_ppm_axis` do that conversion, and every `GESeries`
spectrum (`.spec` on `MRSSeries`/`MRSISeries`) already carries a `chemical_shift` coordinate
built this way — you rarely call these directly, but it's worth knowing what they assume.

| Function | What it does here |
|---|---|
| [`calculate_hz_axis()`](#mnutils.utils.spectra.calculate_hz_axis) | evenly-spaced Hz offsets, centered on zero |
| [`calculate_ppm_axis()`](#mnutils.utils.spectra.calculate_ppm_axis) | Hz axis, scaled to ppm and shifted by `carrier_ppm` |

```{code-cell} ipython3
import numpy as np

from mnutils.utils.spectra import calculate_hz_axis, calculate_ppm_axis
```

(basics-hz-and-ppm-hz)=
## Hz: a symmetric axis of a given width

`calculate_hz_axis` is `np.linspace(-spectral_width / 2, spectral_width / 2, npts)` — every
point covers `spectral_width / npts` Hz, centered on zero.

```{code-cell} ipython3
hz_axis = calculate_hz_axis(spectral_width=1000.0, npts=5)
print(hz_axis)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: calculate_hz_axis
np.testing.assert_allclose(hz_axis, [-500.0, -250.0, 0.0, 250.0, 500.0])
```

`spectral_width`/`npts` can come from a scanner header instead of being passed explicitly —
`GESeries` does exactly this, reading `rdb_hdr.spectral_width`/`rdb_hdr.user1`:

```{code-cell} ipython3
header = {"rdb_hdr": {"spectral_width": 800.0, "user1": 4}}
hz_from_header = calculate_hz_axis(header=header)
print(hz_from_header)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: header-derived hz axis
np.testing.assert_allclose(hz_from_header, np.linspace(-400.0, 400.0, 4))
```

(basics-hz-and-ppm-ppm)=
## ppm: the same axis, scaled and shifted

ppm takes the Hz axis and divides by the scanner's centre frequency (in MHz — so the result is
field-strength-independent), then shifts it by `carrier_ppm`, the chemical-shift reference point
(4.68 ppm for water in ¹H/²H spectra; 0 for most other nuclei — see
[`GESeries.RawMRISeries.carrier_ppm`](#mnutils.GESeries.RawMRISeries)).

```{code-cell} ipython3
ppm_axis = calculate_ppm_axis(
    spectral_width=1000.0, frequency=1000.0, carrier_ppm=2.0, npts=5
)
print(ppm_axis)
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: calculate_ppm_axis
np.testing.assert_allclose(ppm_axis, [1.5, 1.75, 2.0, 2.25, 2.5])
```

(basics-hz-and-ppm-errors)=
## When the header doesn't have enough to go on

Both functions need a spectral width and a point count from *somewhere* — either passed
explicitly or present in `header`. Missing both raises rather than silently returning a
meaningless axis:

```{code-cell} ipython3
:tags: [remove-cell]

import pytest

incomplete_header = {"rdb_hdr": {}, "image": {"specnuc": 1}}
with pytest.raises(ValueError, match="Spectral width"):
    calculate_ppm_axis(header=incomplete_header, npts=1024)
```

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: Spectral width information is required...` | neither `spectral_width=` nor a usable `header["rdb_hdr"]["spectral_width"]` | pass `spectral_width=` explicitly, or check the header actually has `rdb_hdr` populated |
| `ValueError: Number of points information is required...` | same, for `npts` | pass `npts=` explicitly |

:::{seealso}
[The GESeries class hierarchy](#data-model-geseries) shows the `chemical_shift` coordinate these
functions build, attached to a real spectrum.
:::
