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

The raster branch reuses `_render_frames` exactly as `inspect_MRSI_spectra` already calls it — a
gray T1 layer, a colormapped data layer, alpha-blended — once per panel (T1 / data / overlay), fed
into the same `ImageGridWidget` the plain grid already uses. No new renderer.

:::{dropdown} Why not keep `_on_ax` as a separate function?
The repo has no enforced back-compat policy, and a thin wrapper forwarding to `ax=` would be dead
weight the moment every internal caller moves. Removing it outright was cheaper than maintaining
two names for one code path.
:::

:::{dropdown} Why not default to matplotlib for single-slice input, raster otherwise?
That split would have kept the pre-existing `(Figure, axes)` return for single 2D slices
un-broken. It was considered and dropped in favor of a simpler rule — raster is the default engine
full stop, `ax=` is the only thing that forces matplotlib — matching the project's stance that an
interactive-display function returns `None`, not a return value it has to keep faking.
:::

`mask_contour` has no cheap raster equivalent (no contour tracing in `_render_frames`); on the
raster engine it's a warning and gets ignored rather than silently drawn or erroring. Plain mask
*fill* stays a simple alpha composite either engine.

:::{attention} Assumptions to verify
- That the raster engine's T1/data panel bounds (`fast_bounds`, `_resolve_display_bounds`) produce
  a visual match close enough to the matplotlib engine's that switching the default doesn't read
  as a regression to existing notebooks.
- That routing `inspect_MRSI_spectra`'s existing `_render_frames` call and the new overlay panels
  through the same layer shapes doesn't surface a bounds/alpha edge case neither caller hit alone.
:::
