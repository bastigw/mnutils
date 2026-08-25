(diary-overlay-raster-engine)=

# `overlay_image_data_on_T1` still pays the tax `display_images` stopped paying

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-24</span>

`display_images` and `inspect_MRSI_spectra` both dropped their matplotlib `Figure`/`savefig`
round-trip for `_render_frames` — a direct raster composite, no `Figure` at all — and got ~130 ms
per frame back (see [the grid entry](2026-08-18-pil-image-grid.md)). `overlay_image_data_on_T1`
never got the same treatment, so scrubbing through a NIfTI overlay in VS Code still pays that tax
on every slice. It can't just switch over, though: `overlay_image_data_on_T1_on_ax` — the
single-`Axes` variant used to place an overlay panel into a caller's own publication figure — has
no raster equivalent, because there's no `Axes` for a raster widget to draw into.

:::{important}
`overlay_image_data_on_T1` gains an `engine=` switch (raster by default, matplotlib when `ax=` is
given or requested explicitly) and absorbs `overlay_image_data_on_T1_on_ax` as its `ax=` branch,
instead of keeping two public functions.
:::

(diary-overlay-raster-engine-shape)=

## One function, three answers

Folding the `_on_ax` variant in means one function now answers three different questions
depending on how it's called — and matplotlib's single-slice path already returned something
different from its own multi-slice path before this change. Rather than widen the return type
into a bigger union, each call shape gets its own `@overload`, so a type checker narrows the
answer the way it narrows numpy's dtype-dependent overloads:

```{mermaid}
flowchart TD
  A["overlay_image_data_on_T1(...)"] --> B{"ax= given?"}
  B -->|"yes"| C["matplotlib, single Axes<br/>-> (list[AxesImage], contours)"]
  B -->|"no"| D{"engine="}
  D -->|"'matplotlib'"| E["own Figure<br/>-> (Figure, axes) | None"]
  D -->|"'raster' / default"| F["ImageGridWidget<br/>-> None, always"]
```

```python
# scrubbing in a notebook: fast by default, no Figure to manage
mnutils.plotting.overlay_image_data_on_T1(t1, pdff, mask=roi)

# dropping one overlay panel into a publication figure
overlay_image_data_on_T1(t1_slice, pdff_slice, ax=fig.add_subplot(1, 3, 3))
```

`_render_overlay_raster_frames` calls `_render_frames` up to three times per widget — once per
panel (T1: one gray layer; Data: one colormapped layer; Overlay: both, alpha-blended) — with a
`jobs = [(s,) for s in range(n_slices)]` job per slice, the same job shape
`inspect_MRSI_spectra`'s left panel already used. Bounds for the T1 panel come from `fast_bounds`
at the same `(1.0, 99.0)` percentiles `_draw_single_axis_overlay` uses for its anatomical layer;
Data/Overlay bounds come from `_resolve_display_bounds`, honouring `vmin`/`vmax`/`v_percentile`
the same way the matplotlib engine does. Frames are assembled `frames[slice][panel]` and handed to
the existing `ImageGridWidget` — the same widget `display_images` already drives, so this ships no
new frontend code.

:::{dropdown} Why not keep `_on_ax` as a separate function?
The repo has no enforced back-compat policy, and a thin wrapper forwarding to `ax=` would be dead
weight the moment every internal caller moves. Removing it outright was cheaper than maintaining
two names for one code path. Its `ax is None: create own figure` branch went with it too — no call
site (internal or in the docs) ever used it; `only_overlay=True` with `engine="matplotlib"` is the
equivalent for a caller that still wants a single-panel `Figure`.
:::

:::{dropdown} Why not default to matplotlib for single-slice input, raster otherwise?
That split would have kept the pre-existing `(Figure, axes)` return for single 2D slices
un-broken. It was considered and dropped in favor of a simpler rule — raster is the default engine
full stop, `ax=` is the only thing that forces matplotlib — matching the project's stance that an
interactive-display function returns `None`, not a return value it has to keep faking.
:::

`mask_contour` has no cheap raster equivalent (no contour tracing in `_render_frames`); on the
raster engine it's a `logger.warning` and gets dropped (checked as `mask_contour is not None and
mask_contour is not False`, since a truthiness check on an ndarray mask raises) rather than
silently drawn or erroring. Plain mask *fill* stays a simple `np.where(mask, data, nan)` composite
on either engine.

(diary-overlay-raster-engine-changed)=

## What changed from the plan

The plan's first draft proposed collapsing `inspect_MRSI_spectra`'s inline
`_render_frames` call and the new Overlay panel into one shared helper, since both build the exact
same gray-T1-plus-alpha-data layer stack. That idea didn't survive review: `_render_frames` itself
stays exactly as it was — a single generic compositor, not split or wrapped further — and
`_render_overlay_raster_frames` calls it directly, the same way `inspect_MRSI_spectra` always has.
The two call sites still read identically side by side; consolidating them into a third function
would have bought a few lines at the cost of one more name to hold in mind for a shape that's
already legible by inspection.

The type-safety question that prompted the idea landed on `overlay_image_data_on_T1` instead: its
three call shapes (`ax=`, `engine="matplotlib"`, `engine="raster"`/default) get their own
`@overload`s, matching numpy's dtype-dependent overload pattern, rather than one function
returning a three-way union. `_Layer.bounds`'s `tuple | list[tuple]` union — the thing that would
have motivated splitting `_render_frames` — stays exactly where it already was, confined to
`_render_image_grid_frames`'s per-panel autoscale mode; the overlay engine never needs it.
