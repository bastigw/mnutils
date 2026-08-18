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
for one input shape. A single-slice input is now the same widget with no slider — just the bar's
**Border** checkbox, which draws each frame's real extent back in, since NaN pixels are
transparent and a mostly-masked panel otherwise has no visible edge.

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

(diary-pil-image-grid-layout)=

## Handing the layout to CSS means saying what "too big" means

A `Figure` came with a size. A raster doesn't, so every bound the figure used to imply had to be
written down — and the useful version of each turned out to be the one expressed in the *other*
axis from the one it names.

```{mermaid}
flowchart TD
  A["frame, at the array's own pixel size"] --> B["column width<br/>(1fr, wraps at --mnu-panel-min-w-eff)"]
  B --> C["panel height = width ÷ aspect,<br/>capped by --mnu-panel-max-h"]
  C --> D{"rows outgrow<br/>--mnu-grid-max-h?"}
  D -->|"no"| E["block sized to content"]
  D -->|"yes"| F["panel box scrolls"]
```

Sizing panels by height was the first attempt and it fails in a narrow pane: height is not a thing
CSS can hold constant while width has to give. So the panel fills its column and the column is
what yields — and the *floor* becomes a width too, `--mnu-panel-min-h × aspect`, so a row of 5×20
frames wraps sooner instead of flattening into strips. `--mnu-grid-max-h` likewise bounds the
scroll box rather than the panels: dividing a fixed height budget between rows made a twelve-panel
grid twelve stamps, where scrolling keeps twelve readable panels.

:::{warning}
`.mnu-viewer` sets `overflow: hidden`, so a layout wider than its host pane doesn't scroll — it
disappears. That is what cut the fourth of seven panels off in a narrow editor pane while the
same page looked fine in a browser. Grid columns never wrap, so any fixed track list is a bug
waiting for a narrow pane; `repeat(auto-fit, minmax(…, 1fr))` and a wrapping control bar are the
fix.
:::

Two decisions on the raster side follow from the same "the browser is doing the drawing now"
shift. Frames scaled *up* get `image-rendering: pixelated`, set per image by the JS once it can
compare `naturalWidth` against the rendered width, so a 20×20 mask reads as 400 square pixels
rather than a blur — and frames under 64×64 are encoded losslessly, because lossy WebP rings
around hard edges. The threshold stays low deliberately: at 90×90, lossless costs 4.4× the
payload to fix artifacts nobody can see at that scale.

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
- The plan treated the widget's layout as a detail of the port. It was most of the work: five
  rounds of it, all driven by the difference between a browser window and a VS Code output pane.
  The lesson is in the warning above — when the host clips instead of scrolling, "looks right
  here" is not evidence.
