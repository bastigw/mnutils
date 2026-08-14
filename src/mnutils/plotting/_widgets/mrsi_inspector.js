const SVG_NS = "http://www.w3.org/2000/svg";

// Apply a row-major 4x4 affine (flattened to 16 floats) to a 3-vector.
function applyAffine(affine, point) {
  const [x, y, z] = point;
  return [
    affine[0] * x + affine[1] * y + affine[2] * z + affine[3],
    affine[4] * x + affine[5] * y + affine[6] * z + affine[7],
    affine[8] * x + affine[9] * y + affine[10] * z + affine[11],
  ];
}

// Pick ~4-6 "nice" tick values in [lo, hi] using a 1/2/5-per-decade step,
// mirroring the spirit of matplotlib's MaxNLocator(steps=[1, 2, 5]).
function niceTicks(lo, hi, targetCount = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const rawStep = span / targetCount;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const candidates = [1, 2, 5, 10].map((m) => m * magnitude);
  const step = candidates.find((c) => span / c <= targetCount) ?? candidates[candidates.length - 1];
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) {
    ticks.push(Math.round(v / step) * step); // snap to avoid float drift
  }
  return ticks;
}

const SPEC_MARGIN = { left: 52, right: 10, top: 10, bottom: 27 };

function renderWidget(data, el) {
  const leftFrames = data.left_frames;
  const sliceTitles = data.slice_titles;
  const nAnatSlices = data.n_anat_slices;
  const imageWidth = data.image_width;
  const imageHeight = data.image_height;
  const mrsiToDisplay = data.mrsi_to_display_affine;
  const displayToMrsi = data.display_to_mrsi_affine;
  const [nx, ny, nMrsiSlices] = data.grid_shape;
  const [dimsI, dimsJ] = data.mrsi_dims;
  const npts = data.npts;
  const ppm = data.ppm;
  const spectraBuf = new Float32Array(b64ToBytes(data.spectra_bytes).buffer);

  // Full acquired ppm extent (don't assume direction, though ppm is
  // monotonic increasing in practice) and the current display window, which
  // starts as the full extent and narrows via the min/max ppm sliders.
  const ppmDataMin = Math.min(ppm[0], ppm[npts - 1]);
  const ppmDataMax = Math.max(ppm[0], ppm[npts - 1]);
  const ppmStep = npts > 1 ? Math.abs(ppm[1] - ppm[0]) || 0.01 : 0.01;
  const view = { ppmMin: ppmDataMin, ppmMax: ppmDataMax };

  // ppm is monotonic, so the in-range indices form one contiguous block
  // regardless of its direction.
  let winStart = 0;
  let winEnd = npts - 1;
  function computeWindow(loPpm, hiPpm) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < npts; i++) {
      if (ppm[i] >= loPpm && ppm[i] <= hiPpm) {
        if (i < lo) lo = i;
        if (i > hi) hi = i;
      }
    }
    if (lo <= hi) {
      winStart = lo;
      winEnd = hi;
    }
  }
  computeWindow(view.ppmMin, view.ppmMax);

  const frameUrls = leftFrames.map(
    (b64) => URL.createObjectURL(new Blob([b64ToBytes(b64)], { type: "image/png" })),
  );

  const state = {
    sliceIdx: data.initial_slice,
    x: data.initial_voxel[0],
    y: data.initial_voxel[1],
    mrsiSliceIdx: data.initial_voxel[2],
  };

  const viewer = document.createElement("div");
  viewer.className = "mnu-viewer mnutils-mrsi-inspector";
  viewer.tabIndex = 0;

  const leftPanel = document.createElement("div");
  leftPanel.className = "mnutils-mrsi-left-panel";

  const imgWrap = document.createElement("div");
  imgWrap.className = "mnu-panel mnutils-mrsi-image-wrap";

  const img = document.createElement("img");
  img.className = "mnutils-mrsi-image";
  img.src = frameUrls[state.sliceIdx];

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `-0.5 -0.5 ${imageWidth} ${imageHeight}`);
  svg.classList.add("mnutils-mrsi-overlay");

  const polygon = document.createElementNS(SVG_NS, "polygon");
  polygon.setAttribute("class", "mnutils-mrsi-voxel-box");
  svg.append(polygon);

  imgWrap.append(img, svg);

  const sliceTitle = document.createElement("div");
  sliceTitle.className = "mnu-lbl mnutils-mrsi-slice-title";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "mnu-slider";
  slider.min = "0";
  slider.max = String(nAnatSlices - 1);
  slider.value = String(state.sliceIdx);
  slider.disabled = nAnatSlices <= 1;

  leftPanel.append(makeBar(slider), imgWrap, sliceTitle);

  const rightPanel = document.createElement("div");
  rightPanel.className = "mnutils-mrsi-right-panel";

  const spectrumTitle = document.createElement("div");
  spectrumTitle.className = "mnu-lbl mnutils-mrsi-spectrum-title";

  const spectrumSvg = document.createElementNS(SVG_NS, "svg");
  spectrumSvg.classList.add("mnutils-mrsi-spectrum");
  // Fallback size before the ResizeObserver's first (always-async) callback
  // fires; resizeSpectrum() below replaces this with the real rendered size.
  let spectrumViewW = 440;
  let spectrumViewH = 220;
  spectrumSvg.setAttribute("viewBox", `0 0 ${spectrumViewW} ${spectrumViewH}`);
  spectrumSvg.setAttribute("preserveAspectRatio", "none");

  let plotW = spectrumViewW - SPEC_MARGIN.left - SPEC_MARGIN.right;
  let plotH = spectrumViewH - SPEC_MARGIN.top - SPEC_MARGIN.bottom;

  // High ppm on the left, decreasing rightward -- matches the NMR/MRS
  // convention used elsewhere in the codebase (plotting/spectra.py's
  // DEFAULT_SPECTRA_AX_PARAMS xlim).
  function pxForPpm(p) {
    return SPEC_MARGIN.left + ((view.ppmMax - p) / (view.ppmMax - view.ppmMin)) * plotW;
  }

  function pyForValue(v, min, max) {
    const frac = max === min ? 0.5 : (v - min) / (max - min);
    return SPEC_MARGIN.top + (1 - frac) * plotH;
  }

  const xAxisGroup = document.createElementNS(SVG_NS, "g");
  xAxisGroup.setAttribute("class", "mnutils-mrsi-spectrum-axis");
  const xAxisLine = document.createElementNS(SVG_NS, "line");
  xAxisLine.setAttribute("y1", String(SPEC_MARGIN.top + plotH));
  xAxisLine.setAttribute("y2", String(SPEC_MARGIN.top + plotH));
  xAxisGroup.append(xAxisLine);

  const xUnitLabel = document.createElementNS(SVG_NS, "text");
  xUnitLabel.setAttribute("class", "mnutils-mrsi-spectrum-tick-label");
  xUnitLabel.setAttribute("text-anchor", "end");
  xUnitLabel.textContent = "ppm";
  xAxisGroup.append(xUnitLabel);

  // x ticks depend on the current display window (view.ppmMin/ppmMax), so
  // they're rebuilt (not just repositioned) whenever that window or the
  // panel size changes.
  function drawXTicks() {
    for (const node of xAxisGroup.querySelectorAll(".mnutils-mrsi-spectrum-xtick")) {
      node.remove();
    }
    xAxisLine.setAttribute("x1", String(SPEC_MARGIN.left));
    xAxisLine.setAttribute("x2", String(SPEC_MARGIN.left + plotW));
    xUnitLabel.setAttribute("x", String(SPEC_MARGIN.left + plotW));
    xUnitLabel.setAttribute("y", String(SPEC_MARGIN.top + plotH + 16));

    const lo = Math.min(view.ppmMin, view.ppmMax);
    const hi = Math.max(view.ppmMin, view.ppmMax);
    for (const tick of niceTicks(lo, hi)) {
      const px = pxForPpm(tick);
      const tickLine = document.createElementNS(SVG_NS, "line");
      tickLine.setAttribute("class", "mnutils-mrsi-spectrum-xtick");
      tickLine.setAttribute("x1", String(px));
      tickLine.setAttribute("x2", String(px));
      tickLine.setAttribute("y1", String(SPEC_MARGIN.top + plotH));
      tickLine.setAttribute("y2", String(SPEC_MARGIN.top + plotH + 4));
      xAxisGroup.append(tickLine);

      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute(
        "class",
        "mnutils-mrsi-spectrum-tick-label mnutils-mrsi-spectrum-xtick",
      );
      label.setAttribute("x", String(px));
      label.setAttribute("y", String(SPEC_MARGIN.top + plotH + 16));
      label.setAttribute("text-anchor", "middle");
      label.textContent = String(Math.round(tick * 100) / 100);
      xAxisGroup.append(label);
    }
  }

  const yAxisGroup = document.createElementNS(SVG_NS, "g");
  yAxisGroup.setAttribute("class", "mnutils-mrsi-spectrum-axis");
  const yAxisLine = document.createElementNS(SVG_NS, "line");
  yAxisLine.setAttribute("x1", String(SPEC_MARGIN.left));
  yAxisLine.setAttribute("x2", String(SPEC_MARGIN.left));
  yAxisLine.setAttribute("y1", String(SPEC_MARGIN.top));
  yAxisLine.setAttribute("y2", String(SPEC_MARGIN.top + plotH));
  yAxisGroup.append(yAxisLine);

  const spectrumLine = document.createElementNS(SVG_NS, "polyline");
  spectrumLine.setAttribute("class", "mnutils-mrsi-spectrum-line");

  spectrumSvg.append(xAxisGroup, yAxisGroup, spectrumLine);

  function updateSpectrumView() {
    computeWindow(view.ppmMin, view.ppmMax);
    drawXTicks();
    redrawVoxel();
  }

  function resizeSpectrum() {
    const w = spectrumSvg.clientWidth;
    const h = spectrumSvg.clientHeight;
    if (w < 1 || h < 1) return;
    spectrumViewW = w;
    spectrumViewH = h;
    spectrumSvg.setAttribute("viewBox", `0 0 ${spectrumViewW} ${spectrumViewH}`);
    plotW = spectrumViewW - SPEC_MARGIN.left - SPEC_MARGIN.right;
    plotH = spectrumViewH - SPEC_MARGIN.top - SPEC_MARGIN.bottom;
    yAxisLine.setAttribute("y2", String(SPEC_MARGIN.top + plotH));
    updateSpectrumView();
  }

  const ppmMinCtl = makeSliceSlider({
    min: ppmDataMin,
    max: ppmDataMax,
    value: view.ppmMin,
    step: ppmStep,
    label: "Min ppm",
    formatReadout: (v) => v.toFixed(2),
    onInput: (v) => {
      view.ppmMin = Math.min(v, view.ppmMax - ppmStep);
      ppmMinCtl.slider.value = String(view.ppmMin);
      updateSpectrumView();
    },
  });
  const ppmMaxCtl = makeSliceSlider({
    min: ppmDataMin,
    max: ppmDataMax,
    value: view.ppmMax,
    step: ppmStep,
    label: "Max ppm",
    formatReadout: (v) => v.toFixed(2),
    onInput: (v) => {
      view.ppmMax = Math.max(v, view.ppmMin + ppmStep);
      ppmMaxCtl.slider.value = String(view.ppmMax);
      updateSpectrumView();
    },
  });
  ppmMinCtl.slider.disabled = ppmMaxCtl.slider.disabled = ppmDataMin >= ppmDataMax;

  rightPanel.append(
    spectrumTitle,
    spectrumSvg,
    makeBar(ppmMinCtl.lbl, ppmMinCtl.slider, ppmMinCtl.readout),
    makeBar(ppmMaxCtl.lbl, ppmMaxCtl.slider, ppmMaxCtl.readout),
  );

  viewer.append(leftPanel, rightPanel);
  el.append(viewer);

  resizeSpectrum();
  new ResizeObserver(resizeSpectrum).observe(spectrumSvg);

  function voxelPolygonPoints(x, y, mrsiSliceIdx) {
    const cornerOffsets = [
      [-0.5, -0.5, 0],
      [0.5, -0.5, 0],
      [0.5, 0.5, 0],
      [-0.5, 0.5, 0],
    ];
    return cornerOffsets
      .map(([dx, dy, dz]) => {
        const [row, col] = applyAffine(mrsiToDisplay, [
          x + dx,
          y + dy,
          mrsiSliceIdx + dz,
        ]);
        return `${col},${row}`;
      })
      .join(" ");
  }

  function spectrumAt(x, y, mrsiSliceIdx) {
    const offset = ((x * ny + y) * nMrsiSlices + mrsiSliceIdx) * npts;
    return spectraBuf.subarray(offset, offset + npts);
  }

  // Y ticks depend on the currently displayed voxel's amplitude range, so
  // they're rebuilt (not just repositioned) on every redraw.
  function drawYTicks(min, max) {
    for (const node of yAxisGroup.querySelectorAll(".mnutils-mrsi-spectrum-ytick")) {
      node.remove();
    }
    const values = [min, (min + max) / 2, max];
    for (const v of values) {
      const py = pyForValue(v, min, max);

      const tickLine = document.createElementNS(SVG_NS, "line");
      tickLine.setAttribute("class", "mnutils-mrsi-spectrum-ytick");
      tickLine.setAttribute("x1", String(SPEC_MARGIN.left - 4));
      tickLine.setAttribute("x2", String(SPEC_MARGIN.left));
      tickLine.setAttribute("y1", String(py));
      tickLine.setAttribute("y2", String(py));
      yAxisGroup.append(tickLine);

      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute(
        "class",
        "mnutils-mrsi-spectrum-tick-label mnutils-mrsi-spectrum-ytick",
      );
      label.setAttribute("x", String(SPEC_MARGIN.left - 6));
      label.setAttribute("y", String(py + 3));
      label.setAttribute("text-anchor", "end");
      label.textContent = v.toExponential(1);
      yAxisGroup.append(label);
    }
  }

  function drawSpectrum(x, y, mrsiSliceIdx) {
    const spectrum = spectrumAt(x, y, mrsiSliceIdx);
    let min = Infinity;
    let max = -Infinity;
    for (let i = winStart; i <= winEnd; i++) {
      const v = spectrum[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const points = [];
    for (let i = winStart; i <= winEnd; i++) {
      points.push(`${pxForPpm(ppm[i])},${pyForValue(spectrum[i], min, max)}`);
    }
    spectrumLine.setAttribute("points", points.join(" "));
    drawYTicks(min, max);

    const specI = dimsI - 1 - y;
    const specJ = dimsJ - 1 - x;
    spectrumTitle.textContent =
      `Spectrum at voxel (i:${specI}, j:${specJ}, slice:${mrsiSliceIdx})`;
  }

  function redrawVoxel() {
    polygon.setAttribute(
      "points",
      voxelPolygonPoints(state.x, state.y, state.mrsiSliceIdx),
    );
    drawSpectrum(state.x, state.y, state.mrsiSliceIdx);
  }

  function goToSlice(sliceIdx) {
    state.sliceIdx = clamp(sliceIdx, 0, nAnatSlices - 1);
    img.src = frameUrls[state.sliceIdx];
    sliceTitle.textContent = sliceTitles[state.sliceIdx];

    const [, , mrsiSlice] = applyAffine(displayToMrsi, [0, 0, state.sliceIdx]);
    state.mrsiSliceIdx = clamp(Math.round(mrsiSlice), 0, nMrsiSlices - 1);
    redrawVoxel();
  }

  function selectVoxelFromClick(row, col) {
    const [rawX, rawY, rawSlice] = applyAffine(displayToMrsi, [
      row,
      col,
      state.sliceIdx,
    ]);
    state.x = clamp(Math.round(rawX), 0, nx - 1);
    state.y = clamp(Math.round(rawY), 0, ny - 1);
    state.mrsiSliceIdx = clamp(Math.round(rawSlice), 0, nMrsiSlices - 1);
    redrawVoxel();
  }

  slider.addEventListener("input", () => goToSlice(Number(slider.value)));

  img.addEventListener("click", (evt) => {
    viewer.focus();
    const rect = img.getBoundingClientRect();
    const px = evt.clientX - rect.left;
    const py = evt.clientY - rect.top;
    const col = -0.5 + (px / rect.width) * imageWidth;
    const row = -0.5 + (py / rect.height) * imageHeight;
    selectVoxelFromClick(row, col);
  });

  // Grab focus so arrow keys reach us without an extra click, and stop the
  // keydown from propagating past the widget: notebook hosts (VS Code,
  // JupyterLab) bind ArrowUp/ArrowDown at the document/window level for cell
  // navigation, and a plain `preventDefault()` on our own listener doesn't
  // stop that outer handler from also firing on the same event.
  viewer.addEventListener("mouseenter", () => viewer.focus());
  viewer.addEventListener(
    "keydown",
    (evt) => {
      if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(evt.key)) {
        return;
      }
      evt.preventDefault();
      evt.stopPropagation();
      if (evt.key === "ArrowDown") {
        state.y = Math.max(0, state.y - 1);
      } else if (evt.key === "ArrowUp") {
        state.y = Math.min(ny - 1, state.y + 1);
      } else if (evt.key === "ArrowRight") {
        state.x = Math.max(0, state.x - 1);
      } else if (evt.key === "ArrowLeft") {
        state.x = Math.min(nx - 1, state.x + 1);
      }
      redrawVoxel();
    },
    { capture: true },
  );

  sliceTitle.textContent = sliceTitles[state.sliceIdx];
}

export default renderWidget;
