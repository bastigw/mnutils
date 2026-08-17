(diary-anywidget-slice-viewer)=
# The slice slider dies the moment the docs stop running

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-17</span>

`display_images()`, `overlay_image_data_on_T1()` and `inspect_MRSI_spectra()` once scrubbed
through slices with `ipywidgets.interact()`, redrawn live by the `ipympl` matplotlib backend.
That redraw needs a running Python kernel behind the slider. MyST docs are executed once
(`myst build --html --execute`) and nothing here keeps a kernel alive afterwards, so the slider
that shipped to a reader was dead on arrival. `ipympl` and `ipywidgets` were dev-only
dependencies too, yet `images.py` imported `ipywidgets` at module load — a plain install
couldn't import `mnutils.plotting.images` at all.

:::{important}
Interactivity ships as a self-contained `text/html` output: every slice pre-rendered in Python,
embedded alongside an inline `<script type="module">`. Identical behaviour under a live kernel,
in preview, and in a static build — no widget protocol anywhere.
:::

## Why not the widget protocol

The first attempt used `anywidget`, on the theory that mystmd's handling of the
`widget-state+json` / `widget-view+json` MIME types would keep a built page interactive.
Recognising an output type turns out not to be the same as supporting the protocol that produces
it. Both backends now exist side by side — `backend="anywidget"` is kept precisely so this stays
checkable — and a real `myst build --html --execute` against `tests/datasets/HeVo-18` says:

| Evidence in the built page | Result |
|---|---|
| `application/vnd.jupyter.widget-state+json` | **0** |
| `application/vnd.jupyter.widget-view+json` | 1, containing only `{"model_id": "dc405eee…"}` |
| `Exception opening new comm` in build stderr | once per widget-displaying cell |
| self-contained HTML widgets written | 3 of 3, 12.8 MB each |

mystmd's execution engine registers no `jupyter.widget` comm target, so `comm_open` throws and
the state channel carrying the frames and spectra never opens. A model ID with nothing behind it
reaches the page. The rule that generalises: **ship interactivity as `text/html` whenever the
interaction doesn't need a live kernel round-trip** — none of these three ever did.

## What the frames actually cost

Baking every slice in Python is only viable if baking is cheap, and at first it wasn't: 15.5 s
and a 41.9 MB output for a 512×512×100 T1 over a 16×16×16×700 grid. Reusing one `Figure` across
slices helped, which left `fig.savefig(png)` as ~130 ms of the remaining per-slice cost — spent
rasterizing an axes with no ticks, labels, titles, colorbar or contours, only two stacked
`imshow`s. The entire Agg pipeline was being paid for `cmap(norm(arr))` plus an alpha blend.

So `_render_mrsi_left_frames()` does that arithmetic directly. Matplotlib still *supplies* the
colormaps through `ScalarMappable`, so nothing is reimplemented and the two cannot drift; only
the canvas is gone. Frames are WebP q92 (41 dB PSNR against the exact composite, ~7× smaller than
PNG) at the volume's native resolution, and encoding releases the GIL so slices go through a
thread pool.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    A["Slice"] --> B["ScalarMappable: gray + magma"]
    B --> C["Alpha blend, weighted by<br>the overlay's own alpha"]
    C --> D["WebP q92, on a thread pool"]
    D --> E["One text/html output"]
```

**15.5 s → 3.0 s, 41.9 MB → 12.8 MB.**

:::{warning}
The blend must weight `alpha=0.5` by the overlay's *per-pixel* alpha, not apply it flat. Voxels
the colormap marks bad (NaN, alpha 0) have to let the T1 through untouched — a flat blend darkens
the whole anatomy toward the bad colour.
:::

## The spectra buffer

Every voxel's spectrum travels with the widget, which was 11.5 MB of float32 inflating to 15.3 MB
of base64. It is now float16 + zlib, and two details there were not obvious:

- **Raw spectra overflow float16.** The grid peaks at 6.0 × 10⁷ against a 65504 limit, so a plain
  cast silently produces `inf`. The buffer is scaled first, and `spectra_scale` travels with it.
- **The scale must be a power of two.** That shifts exponents and leaves every mantissa
  bit-identical, keeping the buffer compressible; an arbitrary divisor perturbs every mantissa and
  cost ~4× on the compressed size when measured.

Worst per-voxel error is 0.039% of that voxel's own peak. The browser inflates it with the
built-in `DecompressionStream` — no JS dependency — and converts only the selected voxel's points
rather than the whole grid.

:::{dropdown} Why float16 and not int16?
int16 with a global scale compresses slightly better, but the grid spans 0.15 to 6 × 10⁷, so a
single scale leaves a weak voxel with ~9% error. float16 holds ~0.1% *relative* precision at every
magnitude, which is what matters when you view one autoscaled voxel at a time.
:::

:::{dropdown} Why not Plotly frames, or colormapping in JS?
Plotly's slider is client-side too, but means re-plotting in a second, Plotly-specific way — a
maintenance fork of rendering logic that already lives in matplotlib. Sending raw pixel arrays and
colormapping in JS is smaller still, but means reimplementing matplotlib's colormap and contour
handling and keeping the two in lockstep forever. Baking matplotlib's own colormap output keeps
one implementation.
:::

:::{dropdown} Why not one frame per (slice, voxel) pair?
Only the voxel outline changes per voxel, not the image under it. For a 16×16 grid across ~100
anatomical slices that would be tens of thousands of frames; sending the affine and projecting the
outline in JS keeps it at one frame per slice.
:::

## What changed from the plan

- The plan assumed mystmd's recognition of the widget MIME types would make an `anywidget` build
  "just work". It doesn't, and the entry above now carries the measured evidence rather than the
  assumption. `backend="anywidget"` survives as the control that keeps the claim falsifiable — it
  does ship the payload as binary comm buffers (9.5 MB against 12.8 MB of base64, the 33% base64
  tax made visible), which is real, and useless without a kernel.
- The payload risk this entry previously left open — flagged because the original work had no
  access to `tests/datasets/` — is now closed with real numbers, and it was justified: the
  un-tuned payload really was 41.9 MB for one grid.
- Native-resolution frames were expected to be a free improvement. They aren't free: 512² is
  1.64× the pixels of the dpi-derived 400², so resolution had to be chosen deliberately and paid
  for with a better codec rather than inherited.
