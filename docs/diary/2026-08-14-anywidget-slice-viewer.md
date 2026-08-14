(diary-anywidget-slice-viewer)=
# The slice slider dies the moment the docs stop running

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-14</span>

`display_images()`, `overlay_image_data_on_T1()`, and `inspect_MRSI_spectra()` all scrub through
slices with `ipywidgets.interact()` / a manual `IntSlider`, redrawn live by the `ipympl` matplotlib
backend. That redraw needs a running Python kernel behind the slider. `docs/plotting/images.md`
fires it today on every 3D/4D demo — but MyST docs are executed once (`myst start --execute` for
preview, `myst build --html --execute` for the static site) and nothing in this repo keeps a
kernel alive after that. The slider that ships to a reader is dead on arrival. Worse, `ipympl` and
`ipywidgets` are dev-only dependencies, yet `images.py` imports `ipywidgets` unconditionally at
module load — so a plain `pip install mnutils` can't even import `mnutils.plotting.images`.

:::{important}
Replace the three `ipywidgets`/`ipympl` interaction paths with `anywidget` widgets whose JS never
calls back into Python — all data a slider move needs is baked into the widget's model up front, so
the same widget works identically live, in `myst start --execute` preview, and in a static
`myst build --html` page with no kernel at all.
:::

(diary-anywidget-slice-viewer-shape)=
## The shape: bake the frames, move the interaction to JS

mystmd already knows how to embed a Jupyter widget's state in static HTML (it recognizes the
`application/vnd.jupyter.widget-state+json` / `widget-view+json` output MIME types and ships the
`@jupyter-widgets/html-manager` runtime for both preview and build). anywidget rides that same
protocol — the only new rule is that its `render()` function must be self-sufficient, since there's
no kernel to ask for more data mid-interaction.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    A["Multi-slice input"] --> B["Render every slice to a PNG frame, in Python"]
    B --> C["Embed frames + (for MRSI) a raw spectra buffer in the widget model"]
    C --> D["display() the widget, return None"]
    D --> E["JS render(): slider swaps img.src, click/keys redraw from the embedded buffer"]
```

Two widgets cover all three functions: `SliceViewerWidget` (one PNG per slice, slider only) for
`display_images`/`overlay_image_data_on_T1`, and `MRSIVoxelInspectorWidget` (per-slice PNGs +
a raw float32 spectra buffer + the display affine) for `inspect_MRSI_spectra`, whose click-to-
select-voxel and keyboard nav get re-derived client-side from that affine rather than from a second
frame per voxel.

```python
# the call site doesn't change — it's still just:
fig, ax = display_images(volume2d)          # 2D: unchanged, still a real (fig, axes)
display_images(volume3d)                     # 3D/4D: shows the widget, returns None
inspect_MRSI_spectra(mrsi_series, t1_images)  # always interactive, returns None
```

A function that displays a widget doesn't also need to fake a `(fig, axes)` return for it — the
2D/no-interactivity path keeps the old contract; the multi-slice path shows the widget and returns
`None`, and callers unpacking the old tuple return get updated.

:::{dropdown} Why not Plotly frames+slider instead?
Plotly's built-in frame/slider mechanism is fully client-side too, and simpler to wire up. It was
rejected because it means re-plotting (colormap, vmin/vmax, contour overlays, per-axis colorbars)
in a second, Plotly-specific way — a maintenance fork of rendering logic that already lives in
matplotlib. Baking matplotlib's own output to PNG keeps one rendering implementation.
:::

:::{dropdown} Why not send raw pixel arrays and colormap in JS?
Smaller payloads, instant redraw — but it means reimplementing matplotlib's colormap, contour, and
mask-as-NaN logic in JavaScript and keeping the two in lockstep forever. Not worth it for a
portability fix, not a new-feature request.
:::

:::{dropdown} Why not one PNG per (slice, voxel) pair for the MRSI inspector?
For a typical MRSI grid (~10×10) times a few dozen anatomical slices, that's thousands of frames —
because only the voxel outline changes per voxel, not the underlying image. Sending the affine and
projecting the outline in JS keeps the payload at one frame per slice, same order as the plain
slice viewer.
:::

:::{attention} Assumptions to verify
- That mystmd's widget-state embedding genuinely renders and stays interactive in a real
  `myst build --html --execute` output, not just in live preview — this is the entire point of the
  change and hasn't been demonstrated against this repo's actual build yet.
- That `tests/datasets/*` MRSI grids stay small enough (~10×10, per the one fixture checked) that
  the raw float32 spectra buffer doesn't balloon into an unreasonable page payload.
- That losing keyboard-nav-without-a-click-first (the widget container needs focus first, since
  there's no matplotlib canvas to bind `key_press_event` to) is an acceptable UX change for
  `inspect_MRSI_spectra`'s only real caller (`tests/test_overlay_visually.ipynb`).
:::
