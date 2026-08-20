/**
 * Grid of pre-rendered image panels driven by one shared slice slider.
 *
 * The frames are bare rasters straight from PIL; the figure title, panel
 * captions and colorbar are built here instead of being baked into the pixels
 * (see `image_grid.py`). Frames are slice-major -- `frames[slice][panel]` --
 * so moving the slider swaps every panel's `src` in one pass.
 */

/** Format a colorbar tick the way a reader expects to see a data value. */
function formatTick(value) {
  if (!Number.isFinite(value)) return String(value);
  const magnitude = Math.abs(value);
  if (value === 0) return "0";
  if (magnitude >= 1e4 || magnitude < 1e-2) return value.toExponential(1);
  return String(Number(value.toPrecision(3)));
}

/**
 * Build a colorbar: a `linear-gradient` strip flanked by its two bounds.
 *
 * The gradient is the matplotlib colormap sampled to hex stops in Python, so
 * the bar and the pixels it explains come from one colormap and can't drift.
 * Returns the wrapper plus a `setBounds` that rewrites only the tick text --
 * `colorbar_mode="each"` autoscales per slice, so those two nodes change on
 * every slider move while the gradient never does.
 */
function makeColorbar(stops) {
  const wrap = document.createElement("div");
  wrap.className = "mnutils-image-grid-cbar";

  const lo = document.createElement("span");
  lo.className = "mnutils-image-grid-cbar-tick";

  const strip = document.createElement("div");
  strip.className = "mnutils-image-grid-cbar-strip";
  strip.style.background = `linear-gradient(to right, ${stops.join(", ")})`;

  const hi = document.createElement("span");
  hi.className = "mnutils-image-grid-cbar-tick";

  wrap.append(lo, strip, hi);

  return {
    wrap,
    setBounds: ([low, high]) => {
      lo.textContent = formatTick(low);
      hi.textContent = formatTick(high);
    },
  };
}

function renderWidget(data, el) {
  const {
    frames,
    bounds,
    colormap_stops: stops,
    n_slices: nSlices,
    num_panels: numPanels,
    num_cols: numCols,
    titles,
    fig_title: figTitle,
    colorbar_mode: colorbarMode,
    initial_index: initialIndex,
    slice_label: sliceLabel,
  } = data;

  // WebP with an alpha channel: values the colormap marks "bad" (NaN, and so
  // anything masked out by `zeros_as_nan`) are transparent rather than a
  // colour, so the panel background shows through in both light and dark.
  const urls = frames.map((slice) =>
    slice.map((b64) => URL.createObjectURL(new Blob([b64ToBytes(b64)], { type: "image/webp" }))),
  );

  const viewer = document.createElement("div");
  viewer.className = "mnu-viewer mnutils-image-grid";

  if (figTitle) {
    const heading = document.createElement("div");
    heading.className = "mnutils-image-grid-title";
    heading.textContent = figTitle;
    viewer.append(heading);
  }

  const panel = document.createElement("div");
  panel.className = "mnu-panel mnutils-image-grid-panel";

  const grid = document.createElement("div");
  grid.className = "mnutils-image-grid-cells";
  // The column count is nominal: the CSS wraps to fewer columns when the pane
  // is too narrow for them, and caps the widget's width so a wide pane doesn't
  // fit more. Rows are not counted -- however many they come to, the panel box
  // scrolls once they outgrow it.
  const cols = Math.min(numCols, numPanels);
  viewer.style.setProperty("--mnu-grid-cols", String(cols));

  const images = [];
  const colorbars = [];
  for (let p = 0; p < numPanels; p++) {
    const cell = document.createElement("figure");
    cell.className = "mnutils-image-grid-cell";

    const img = document.createElement("img");
    img.className = "mnutils-image-grid-image";
    img.src = urls[initialIndex][p];
    img.alt = titles[p] ? `Panel: ${titles[p]}` : `Panel ${p + 1}`;
    images.push(img);
    cell.append(img);

    if (titles[p]) {
      const caption = document.createElement("figcaption");
      caption.className = "mnutils-image-grid-caption";
      caption.textContent = titles[p];
      cell.append(caption);
    }

    if (colorbarMode === "each") {
      const cbar = makeColorbar(stops);
      colorbars.push(cbar);
      cell.append(cbar.wrap);
    }

    grid.append(cell);
  }

  panel.append(grid);

  if (colorbarMode === "single") {
    const cbar = makeColorbar(stops);
    cbar.wrap.classList.add("mnutils-image-grid-cbar-shared");
    colorbars.push(cbar);
    panel.append(cbar.wrap);
  }

  // Low-resolution data should look like the grid of values it is. Once a
  // frame is drawn wider than its own pixel count the browser's smooth
  // upscaling turns a 20x20 array into a blur, so those get `pixelated`;
  // frames being scaled *down* keep the smooth default, where nearest
  // -neighbour would only alias. Rechecked on resize, since which side of
  // 1:1 a panel falls on depends on the width it ends up with.
  const syncPixelation = () => {
    for (const img of images) {
      if (!img.naturalWidth) continue;
      img.style.setProperty(
        "--mnu-image-rendering",
        img.getBoundingClientRect().width > img.naturalWidth ? "pixelated" : "auto",
      );
    }
  };

  // The frames' aspect ratio decides how wide the panels want to be, and so
  // how wide the whole widget is allowed to grow (see --mnu-panel-w-cap).
  // Only measurable once a frame has decoded, hence the load handler.
  const syncAspect = () => {
    const img = images.find((candidate) => candidate.naturalWidth);
    if (img) viewer.style.setProperty("--mnu-aspect", String(img.naturalWidth / img.naturalHeight));
  };

  for (const img of images) {
    img.addEventListener("load", () => {
      syncAspect();
      syncPixelation();
    });
  }
  if (typeof ResizeObserver === "function") new ResizeObserver(syncPixelation).observe(grid);

  const showSlice = (idx) => {
    for (let p = 0; p < numPanels; p++) images[p].src = urls[idx][p];
    // "single" has one bar for the whole grid, so panel 0's bounds are the
    // grid's bounds -- with a shared colorbar every panel shares them anyway.
    colorbars.forEach((cbar, i) => cbar.setBounds(bounds[idx][colorbarMode === "each" ? i : 0]));
  };
  showSlice(initialIndex);

  // A panel's edge is invisible wherever the data is NaN, since those pixels
  // are transparent -- which makes a mostly-masked image look like it has an
  // arbitrary shape. The border draws the frame's real extent back in, off by
  // default because it is clutter on an image that fills its own bounds.
  const { toggle } = makeToggle({
    label: "Border",
    checked: false,
    onChange: (on) => grid.classList.toggle("is-bordered", on),
  });

  // A single-slice stack has nothing to scrub, so it gets a bar with just the
  // toggle rather than a disabled slider whose only honest readout is "0 of 0".
  const controls = [];
  if (nSlices > 1) {
    const { lbl, slider, readout } = makeSliceSlider({
      min: 0,
      max: nSlices - 1,
      value: initialIndex,
      label: sliceLabel,
      formatReadout: (idx) => String(idx),
      onInput: showSlice,
    });
    // Reserve the readout's full width up front, sized to the largest index
    // it will ever show -- otherwise a 3-digit slice count only grows the bar
    // past its wrap threshold once the slider is dragged there, so the layout
    // shifts mid-scrub instead of on load.
    readout.style.minWidth = `${String(nSlices - 1).length}ch`;
    controls.push(makeGroup(lbl, readout), slider);
  }
  const bar = makeBar(...controls, toggle);
  bar.classList.add("mnutils-image-grid-bar");
  viewer.append(bar);

  viewer.append(panel);
  el.append(viewer);
}

export default renderWidget;
