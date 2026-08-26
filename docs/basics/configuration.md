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

(basics-configuration)=
# Setting defaults once, with rcParams

```{code-cell} ipython3
:tags: [remove-cell]

import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
from loguru import logger

# Crisp retina output + sane default DPI for the rendered docs
matplotlib_inline.backend_inline.set_matplotlib_formats("retina")
plt.rcParams["figure.dpi"] = 150
logger.remove()
```

> **I have typed `cmap="viridis"` into eleven calls in this notebook. Is there one place I can
> say it instead?**

There is: [`mnutils.rcParams`](#mnutils.rcparams.rcParams), a single validated mapping of every
default MNUtils ships. It works the way `matplotlib.rcParams` does, deliberately — the muscle
memory transfers, including the escape hatches.

| Object | What it does here |
|---|---|
| [`rcParams`](#mnutils.rcparams.rcParams) | the live mapping every MNUtils function reads its defaults from |
| [`rc_context()`](#mnutils.rcparams.rc_context) | applies overrides for one block, then puts them back |
| [`rcdefaults()`](#mnutils.rcparams.rcdefaults) | restores the built-ins |
| [`rc_presets`](#mnutils.rcparams) | named override bundles (`"poster"`, `"talk"`, `"paper"`) |
| [`display_images()`](#mnutils.plotting.images.display_images) | the grid whose panel size the last section retunes |

(basics-configuration-what)=
## What is in there

```{code-cell} ipython3
import numpy as np

import mnutils
from mnutils.plotting.images import display_images

sorted(mnutils.rcParams)
```

The names are dotted and grouped by what they configure: `image.*` for how image panels are drawn,
`grid.*` for how big the `display_images` grid gets, `spectra.*` for spectral axes, `nifti.*` for
orientation and resampling, `save.*` for figure saving.

```{code-cell} ipython3
mnutils.rcParams["image.cmap"], mnutils.rcParams["nifti.orientation"]
```

(basics-configuration-session)=
## Change it for the session

Assign, and every later call picks it up:

```{code-cell} ipython3
mnutils.rcParams["image.cmap"] = "viridis"

rng = np.random.default_rng(1337)
volume = rng.random((32, 32, 6))

display_images(volume, fig_title="viridis, because rcParams says so")
```

:::{dropdown} Why this works at all — and why it didn't before
A default that lives in a function signature is evaluated **once, at import**:

```python
def display_images(..., cmap: str = DEFAULT_PARAMS["cmap"]):  # ❌ frozen
```

Assigning to `DEFAULT_PARAMS` after `import mnutils` then changes nothing — the signature already
captured the old string. Every MNUtils entry point instead takes `None` and resolves it in the
body, via [`resolve_rc()`](#mnutils.rcparams.resolve_rc):

```python
def display_images(..., cmap: str | None = None):  # ✅ read at call time
    cmap = resolve_rc(cmap, "image.cmap")
```

That is the whole reason `rcParams` is a module of its own rather than a dict per module.
:::

(basics-configuration-block)=
## Change it for one block

A session-wide assignment is the wrong tool when one figure needs different treatment.
[`rc_context()`](#mnutils.rcparams.rc_context) applies overrides and restores whatever was there
before, exception or not:

```{code-cell} ipython3
with mnutils.rc_context({"image.cmap": "gray"}):
    print("inside :", mnutils.rcParams["image.cmap"])

print("outside:", mnutils.rcParams["image.cmap"])
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: rc_context restores the previous value, including on an exception
assert mnutils.rcParams["image.cmap"] == "viridis"
try:
    with mnutils.rc_context({"image.cmap": "gray"}):
        raise RuntimeError("boom")
except RuntimeError:
    pass
assert mnutils.rcParams["image.cmap"] == "viridis"

mnutils.rcdefaults()
assert mnutils.rcParams["image.cmap"] == "magma"
```

[`rcdefaults()`](#mnutils.rcparams.rcdefaults) puts everything back, which is the fastest way out
of a notebook whose defaults have drifted:

```{code-cell} ipython3
mnutils.rcdefaults()
mnutils.rcParams["image.cmap"]
```

The saving contexts are the same mechanism wearing a name. `save_figure(..., context="poster")`
applies the `"poster"` entry of [`rc_presets`](#mnutils.rcparams) through
`rc_context` — there is no separate poster-parameters dict to keep in sync:

```{code-cell} ipython3
mnutils.rc_presets
```

(basics-configuration-validation)=
## A typo raises instead of doing nothing

Every assignment goes through a validator that owns that key, so the failure lands where the
mistake is:

```{code-cell} ipython3
try:
    mnutils.rcParams["image.cmpa"] = "gray"
except KeyError as err:
    print(err)
```

```{code-cell} ipython3
try:
    mnutils.rcParams["nifti.orientation"] = ("L", "P", "Z")
except ValueError as err:
    print(err)
```

:::{warning}
This matters most for the `grid.*` lengths: they are interpolated into a `style` attribute on the
generated widget HTML. The validator whitelists CSS lengths (`"30rem"`, `"400px"`, `"60vh"`, or a
bare number read as rem) rather than accepting any string, so a stray value cannot become markup.
:::

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: unknown keys and invalid values are both rejected, and neither leaves a trace
import pytest

with pytest.raises(KeyError, match="not a valid rcParam"):
    mnutils.rcParams["image.cmpa"] = "gray"
with pytest.raises(ValueError):
    mnutils.rcParams["grid.panel_height"] = "30rem; background: red"
assert mnutils.rcParams["grid.panel_height"] == "17rem"
assert "image.cmpa" not in mnutils.rcParams
```

(basics-configuration-panels)=
## Bigger panels for an anatomical series

The default grid is tuned for a row of low-resolution multi-nuclear panels. A high-resolution
anatomical slice at that size throws away most of what is in the data — so `grid.panel_height`
exists, and so does a per-call argument for the one figure that needs it:

```{code-cell} ipython3
anatomical = rng.random((160, 160, 8))

display_images(anatomical, fig_title="default panel height")
```

```{code-cell} ipython3
display_images(anatomical, panel_height="30rem", fig_title="panel_height='30rem'")
```

Pass `grid_max_height` alongside it when the taller panels start scrolling inside their box, and
set `mnutils.rcParams["grid.panel_height"]` instead when *every* grid in the notebook should be
bigger.

Precedence runs in the order you would guess, and it is the same for every parameter in this page:

```{mermaid}
flowchart LR
    A["explicit argument<br><span style='font-family:monospace;'>panel_height='30rem'</span>"] --> B["rc_context override"]
    B --> C["<span style='font-family:monospace;'>rcParams</span> assignment"]
    C --> D["built-in default"]
```

```{code-cell} ipython3
:tags: [remove-cell]

# STRICT TESTS: an explicit argument beats rcParams, and doesn't leak into it
mnutils.rcParams["grid.panel_height"] = "12rem"
display_images(anatomical[:, :, :2], panel_height="30rem")
assert mnutils.rcParams["grid.panel_height"] == "12rem"

# ...and a bare number is read as rem by the same validator the rcParam uses
assert mnutils.rcparams.validate_css_length(30) == "30rem"

mnutils.rcdefaults()
assert mnutils.rcParams["grid.panel_height"] == "17rem"
```
