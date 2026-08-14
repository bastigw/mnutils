(diary-anywidget-slice-viewer)=
# The slice slider dies the moment the docs stop running

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-14</span>

`display_images()`, `overlay_image_data_on_T1()`, and `inspect_MRSI_spectra()` all scrubbed
through slices with `ipywidgets.interact()` / a manual `IntSlider`, redrawn live by the `ipympl`
matplotlib backend. That redraw needed a running Python kernel behind the slider. `docs/plotting/
images.md` fired it on every 3D/4D demo — but MyST docs are executed once (`myst start --execute`
for preview, `myst build --html --execute` for the static site) and nothing in this repo keeps a
kernel alive after that. The slider that shipped to a reader was dead on arrival. Worse, `ipympl`
and `ipywidgets` were dev-only dependencies, yet `images.py` imported `ipywidgets` unconditionally
at module load — so a plain `pip install mnutils` couldn't even import `mnutils.plotting.images`.

:::{important}
Replace the three `ipywidgets`/`ipympl` interaction paths with a plain, self-contained
`_repr_html_` output — every slice pre-rendered to a PNG, embedded as base64 alongside a small
inline `<script type="module">` — so the same output works identically live, in `myst start
--execute` preview, and in a static `myst build --html` page with no kernel and no widget protocol
at all.
:::

(diary-anywidget-slice-viewer-shape)=
## The shape: bake the frames, ship them as one HTML blob

The first version of this change used `anywidget` (the Jupyter-widget protocol), on the theory
that mystmd's built-in handling of the `application/vnd.jupyter.widget-state+json` /
`widget-view+json` output types would embed everything needed for a static page to stay
interactive. Running the real build proved that wrong: mystmd's execution engine has no
`jupyter.widget` comm target registered, so every widget's `comm_open` throws
(`Exception opening new comm`, once per widget-displaying cell) and the state-sync channel that
carries the actual model data — our PNG frames, the affine, the spectra buffer — never opens. Only
a dangling `widget-view+json` reference (a model ID with nothing behind it) made it into the built
page. The slider would have rendered broken in exactly the place this change exists to fix.

The fix drops the widget protocol entirely. `SliceViewerWidget` and `MRSIVoxelInspectorWidget`
(`src/mnutils/plotting/_widgets/`) are plain Python classes with no base class: each base64-encodes
its data into a dict and implements `_repr_html_(self)` via a shared `render_html()` helper
(`_widgets/_html.py`). That returns one self-contained string — a `<div id="mnutils-widget-
{uuid}">`, an inlined `<style>` from the widget's `.css`, and an inlined `<script type="module">`
containing the widget's whole JS source plus a trailing `renderWidget(<data>, <container>)` call.
This is just a standard Jupyter `text/html` output — a well-supported, ordinary MyST rendering path
(the same one a pandas `DataFrame` repr or a Plotly figure uses), not the widget-state protocol
that failed.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    A["Multi-slice input"] --> B["Render every slice to a PNG frame, in Python"]
    B --> C["Base64-encode frames + (for MRSI) the spectra buffer into one dict"]
    C --> D["_repr_html_(): self-contained div + style + inline script + data"]
    D --> E["JS: slider swaps img.src, click/keys redraw from the embedded buffer"]
```

The JS itself barely changed shape: `slice_viewer.js` / `mrsi_inspector.js` are still plain ESM,
still no build step, and still export a render function — it just now reads a plain `data` object
(`function renderWidget(data, el)`) instead of anywidget's `model.get("key")` calling convention,
and decodes the base64 payloads itself (`atob` + `Uint8Array.from`) instead of receiving raw bytes
over a comm channel.

```python
# the call site doesn't change — it's still just:
fig, ax = display_images(volume2d)          # 2D: unchanged, still a real (fig, axes)
display_images(volume3d)                     # 3D/4D: shows the widget, returns None
inspect_MRSI_spectra(mrsi_series, t1_images)  # always interactive, returns None
```

A function that displays a widget doesn't also need to fake a `(fig, axes)` return for it — the
2D/no-interactivity path keeps the old contract; the multi-slice path shows the widget and returns
`None`. `pyproject.toml` reflects the pivot too: the `anywidget` dependency the original plan added
came back out (nothing calls it directly any more — it's still resolved transitively via `xmris`,
just not something mnutils depends on), and `ipython>=8.0` went in as an explicit dependency, since
`IPython.display.display` is now called directly from `images.py` rather than riding along as an
implicit transitive dependency.

:::{dropdown} Why not Plotly frames+slider instead?
Plotly's built-in frame/slider mechanism is fully client-side too, and simpler to wire up. It was
rejected because it means re-plotting (colormap, vmin/vmax, contour overlays, per-axis colorbars)
in a second, Plotly-specific way — a maintenance fork of rendering logic that already lives in
matplotlib. Baking matplotlib's own output to PNG keeps one rendering implementation. This
reasoning didn't depend on the widget-protocol pivot and still holds.
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

(diary-anywidget-slice-viewer-changed)=
## What changed from the plan

- The plan assumed mystmd's built-in `widget-state+json`/`widget-view+json` handling would make an
  `anywidget`-based widget "just work" in a static build, since mystmd recognizes those MIME types.
  Running the actual `myst build --html --execute` showed that recognizing the output type isn't
  the same as supporting the protocol that produces it: mystmd's execution engine never opens the
  `jupyter.widget` comm, so the state payload is never captured, only a dangling view reference is.
  Anything that leans on ipywidgets/anywidget for interactivity in this docs pipeline will hit the
  same wall — the fix that generalizes is: ship interactivity as a self-contained `text/html`
  output instead, whenever the interaction doesn't actually need a live Python kernel round-trip
  (all three widgets here never did).
- The MRSI grid-size payload risk flagged in the original assumptions was never resolved either
  way — the sandbox this was built in doesn't have the real datasets under `tests/datasets/` needed
  to run `inspect_MRSI_spectra` end-to-end, so only the underlying frame-rendering and widget
  construction were smoke-tested directly, not the full precompute loop against real MRSI grid
  dimensions. Worth checking against real data before leaning on this for a large grid.
