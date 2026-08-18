(diary-pil-image-grid)=
# The grid builds a whole Figure just to throw it away

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-18</span>

`inspect_MRSI_spectra()` stopped going through matplotlib for its anatomical frames and got ~130 ms
per slice back, plus sharper images at the volume's native resolution — the composite is really
`cmap(norm(arr))` and an alpha blend, and a `Figure`/Agg canvas is an expensive way to say that
(see [the slice-slider entry](2026-08-14-anywidget-slice-viewer.md)). `display_images()` does the
same arithmetic and still pays the same tax: a `Figure`, a gridspec, per-axis `imshow`, a PNG
encode, once per slice. The reason it wasn't converted with the MRSI path is that its frames carry
things the MRSI frames don't — panel titles, a figure title, a colorbar. Those are what keep the
Figure alive, so the question is not "can PIL draw the pixels" but **where the chrome goes when
matplotlib no longer draws it.**

:::{important}
The chrome moves to the browser: PIL renders bare panels at native resolution, and titles,
figure title and colorbar become DOM in a new grid widget.
:::

(diary-pil-image-grid-shape)=
## What the split looks like

```{mermaid}
flowchart LR
  A["images[:, :, slice, panel]"] --> B["ScalarMappable(Normalize, cmap)"]
  B --> C["RGBA uint8 at native (nrows, ncols)"]
  C --> D["WebP bytes, threaded encode"]
  D --> E["one img per panel per slice"]
  E --> F["ImageGridWidget"]
  G["titles, fig_title, colorbar bounds"] --> F
  F --> H["CSS grid: captions, gradient strip, slider"]
```

matplotlib is not gone — it still *supplies* the colormap through `ScalarMappable`, so the
colorbar gradient and the pixels come from one source and cannot drift. What it stops doing is
laying out and rasterising.

| Chrome today | Where it lands |
|---|---|
| per-panel `titles` | `<figcaption>` under each panel |
| `fig_title` | heading above the grid |
| colorbar, `mode="single"` | one 256-stop CSS `linear-gradient` strip + tick labels |
| colorbar, `mode="each"` | per-panel strip, bounds shipped **per slice** (this mode autoscales) |
| `xlabel`/`xticks`/`aspect`/`imshow_kws`/`fig_kws` | nothing — there is no Axes left |

```python
# 4D input, four panels, a shared colorbar: displays, returns nothing.
mnutils.plotting.display_images(
    [t1, pdff, r2s, mask], titles=["T1", "PDFF", "R2*", "Mask"], colorbar=True
)
```

That last line is the breaking part. `display_images()` currently hands back `(Figure, axes)` for
2D input and `None` for a multi-slice stack; with no Figure anywhere it returns `None` always, and
the single-slice case is just the widget with its slider hidden. Callers that post-processed the
returned axes lose that hook — deliberately, since keeping it means keeping a second full
rendering path for one input shape.

:::{dropdown} Why not draw the chrome into the raster with PIL?
`PIL.ImageDraw` could paint titles and a colorbar strip onto the frame, which keeps the client
side trivial. But it makes us own font discovery and text metrics, bakes the layout at render
resolution, and multiplies payload for `mode="each"` where only the tick labels change per slice.
DOM chrome stays sharp at any zoom and lets a slice change rewrite three text nodes.
:::

:::{dropdown} Why not keep matplotlib whenever chrome is requested?
A titled grid with a colorbar *is* the common call, so a fast path that excludes it optimises the
case nobody makes — while freezing two rendering implementations in place forever.
:::

:::{attention} Assumptions to verify
- That native-resolution WebP panels are a payload *win* here. The MRSI entry found native
  resolution is not free (512² is 1.64× the pixels of the dpi-derived 400²); a 4-panel grid over
  ~100 slices is 400 frames, so this needs measuring against a real dataset, not assuming.
- That `colorbar_mode="each"` can be reproduced faithfully. It currently nulls `vmin`/`vmax` and
  lets matplotlib autoscale each panel per slice; moving that into Python means recomputing bounds
  per (panel, slice) and shipping them — cheap, but the numbers must match what the page showed
  before.
- That nothing in `docs/` or `src/` depends on the `(Figure, axes)` return. Unverified beyond a
  grep of this repo; downstream notebooks are out of reach.
- That a hidden slider is the right shape for 2D input, rather than a plain `<img>` output.
:::
