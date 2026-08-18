(diary-pil-image-grid)=

# The grid builds a whole Figure just to throw it away

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-18</span>

`inspect_MRSI_spectra()` stopped going through matplotlib for its anatomical frames and got ~130 ms
per slice back, plus sharper images at the volume's native resolution — the composite is really
`cmap(norm(arr))` and an alpha blend, and a `Figure`/Agg canvas is an expensive way to create these images
(see [the slice-slider entry](2026-08-14-anywidget-slice-viewer.md)). `display_images()` does the
same arithmetic and still pays the same tax: a `Figure`, a gridspec, per-axis `imshow`, a PNG
encode, once per slice. The reason it wasn't converted with the MRSI path is that its frames carry
things the MRSI frames don't — panel titles, a figure title, a colorbar. Those are what keep the
Figure alive.

:::{important}
The annotations move to the browser: PIL renders bare panels at native resolution, and panel
titles, figure title and colorbar become DOM in a new grid widget.
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

matplotlib is not gone — it still _supplies_ the colormap through `ScalarMappable`, so the
colorbar gradient and the pixels come from one source and cannot drift. What it stops doing is
laying out and rasterising. `_render_image_grid_frames()` returns the frames slice-major together
with the `(vmin, vmax)` each was normalised with; `_widgets/image_grid.{py,js,css}` turns that into
the page.

| Annotation today                                  | Where it lands                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------- |
| per-panel `titles`                                | `<figcaption>` under each panel                                      |
| `fig_title`                                       | heading above the grid                                               |
| colorbar, `mode="single"`                         | one 64-stop CSS `linear-gradient` strip + tick labels                |
| colorbar, `mode="each"`                           | per-panel strip, bounds shipped **per slice** (this mode autoscales) |
| `xlabel`/`xticks`/`aspect`/`imshow_kws`/`fig_kws` | nothing — there is no Axes left                                      |

```python
# 4D input, four panels, a shared colorbar: displays, returns nothing.
mnutils.plotting.display_images(
    [t1, pdff, r2s, mask], titles=["T1", "PDFF", "R2*", "Mask"], colorbar=True
)
```

That last line is the breaking part. `display_images()` used to hand back `(Figure, axes)` for
2D input and `None` for a multi-slice stack; with no Figure anywhere it returns `None` always
(and so does `display_nifti()`, and `NiiBase.display()`). Callers that post-processed the returned
axes lose that hook — deliberately, since keeping it means keeping a second full rendering path
for one input shape. A single-slice input is now the same widget with no slider attached, rather
than a disabled slider whose only honest readout is "0 of 0".

The measurement that settled it, on the 90×90×100×7 volume `docs/plotting/images.md` generates —
700 panels against the old path's 100 whole-grid PNGs:

| | payload | render |
|---|---|---|
| `Figure` + PNG per slice | 14.6 MB | 17.2 s |
| PIL + WebP per panel | 0.85 MB | 0.9 s |

Native resolution being *cheaper* here is the opposite of what the MRSI frames found, and for a
plain reason: an MRSI frame is one 512² image where the dpi-derived alternative was 400², while a
grid frame is one panel out of seven that the old path rendered inside a 1400×600 figure canvas
padded with white margins, ticks and text. Dropping the canvas drops most of the pixels.

:::{warning}
Panels now keep their data's own aspect ratio. The old path defaulted `aspect` to
`ncols / nrows`, which stretched every panel into a square box regardless of shape; a non-square
array will look different — correct, but different — than it did before.
:::

:::{dropdown} Why not draw the annotations into the raster with PIL?
`PIL.ImageDraw` could paint titles and a colorbar strip onto the frame, which keeps the client
side trivial. But it makes us own font discovery and text metrics, bakes the layout at render
resolution, and multiplies payload for `mode="each"` where only the tick labels change per slice.
Text in the DOM stays sharp at any zoom and lets a slice change rewrite three text nodes.
:::

:::{dropdown} Why not keep matplotlib whenever annotations are requested?
A titled grid with a colorbar _is_ the common call, so a fast path that excludes it optimises the
case nobody makes — while freezing two rendering implementations in place forever.
:::

:::{dropdown} Why RGBA, when the MRSI frames are RGB?
An MRSI frame is a T1 with an overlay blended onto it: every pixel has something behind it. A
grid panel doesn't, and `zeros_as_nan=True` deliberately produces pixels with nothing to show.
matplotlib gave those the figure's face colour, which is a light-theme decision baked into the
image. Alpha 0 defers it to the page, so masked voxels read correctly in both themes.
:::

(diary-pil-image-grid-changed)=

## What changed from the plan

- The plan expected `colorbar_mode="each"` to be the hard part, since it nulls `vmin`/`vmax` and
  lets matplotlib autoscale each panel per slice. Recomputing those bounds in Python turned out to
  be the easy half; the awkward half was that a *shared* colorbar can also have a degenerate range
  (a constant image, or `zeros_as_nan` on an all-zero array), where matplotlib silently rendered
  the colormap's low end and a text label would have read "0 to 0". Both paths now widen a
  zero-width range to `(low, low + 1)`.
- Pass 1 assumed the docs page's strict tests could keep asserting on shapes. They can't — there
  is no return value to assert on — so `docs/plotting/images.md` checks the panel count on
  `_render_image_grid_frames()` directly, and says in the cell why it reaches for a private helper.
