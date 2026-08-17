// Preact + htm (JSX-free templating) + Chart.js, loaded straight from a CDN:
// no bundler in this project (see _shared/__init__.py), and this widget's
// interaction (dual-range zoom, per-voxel redraw) is enough state to want a
// real component model rather than hand-rolled DOM diffing.
import Chart from "https://esm.sh/chart.js@4.4.1/auto";
import htm from "https://esm.sh/htm@3.1.1";
import noUiSlider from "https://esm.sh/nouislider@15.8.1";
import { h, render } from "https://esm.sh/preact@10.19.6";
import { useEffect, useMemo, useRef, useState } from "https://esm.sh/preact@10.19.6/hooks";

const html = htm.bind(h);

const NOUISLIDER_CSS_HREF = "https://cdn.jsdelivr.net/npm/nouislider@15.8.1/dist/nouislider.min.css";

// nouislider ships its own stylesheet rather than being themeable via JS
// options; inject it once per document (widgets render into plain divs, not
// shadow roots, so a single document-level <link> covers every instance).
function ensureNoUiSliderCss() {
  if (document.querySelector(`link[href="${NOUISLIDER_CSS_HREF}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = NOUISLIDER_CSS_HREF;
  document.head.appendChild(link);
}

// Apply a row-major 4x4 affine (flattened to 16 floats) to a 3-vector.
// MRSI-specific (the affine and its inverse are baked into this widget's
// data payload), so kept local rather than promoted to _shared/dom.js.
function applyAffine(affine, point) {
  const [x, y, z] = point;
  return [
    affine[0] * x + affine[1] * y + affine[2] * z + affine[3],
    affine[4] * x + affine[5] * y + affine[6] * z + affine[7],
    affine[8] * x + affine[9] * y + affine[10] * z + affine[11],
  ];
}

// Dual-thumb ppm range slider, backed by nouislider (dedicated dual-handle
// range library) instead of the two-overlaid-native-inputs hack. The
// instance is created once and driven imperatively thereafter: external prop
// changes go through `.set(values, false)` (suppresses events) so they don't
// loop back into `onChange`, while user drags fire nouislider's 'update'
// event (also covers programmatic-less internal moves during a drag) into
// `onChange`.
function DualRangeSlider({ min, max, step, valueMin, valueMax, onChange }) {
  const containerRef = useRef(null);
  const sliderRef = useRef(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  useEffect(() => {
    ensureNoUiSliderCss();
    if (!containerRef.current) return undefined;

    const slider = noUiSlider.create(containerRef.current, {
      start: [valueMin, valueMax],
      connect: true,
      range: { min, max },
      step,
      behaviour: "drag-fixed",
    });
    sliderRef.current = slider;

    slider.on("update", (values) => {
      const [lo, hi] = values.map(Number);
      onChangeRef.current(lo, hi);
    });

    return () => {
      slider.destroy();
      sliderRef.current = null;
    };
    // Range bounds/step come from the dataset and never change after mount;
    // only valueMin/valueMax move thereafter, synced imperatively below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [min, max, step]);

  useEffect(() => {
    sliderRef.current?.set([valueMin, valueMax], false);
  }, [valueMin, valueMax]);

  useEffect(() => {
    const slider = sliderRef.current;
    if (!slider) return;
    if (max <= min) slider.disable();
    else slider.enable();
  }, [min, max]);

  return html`<div ref=${containerRef} className="mnutils-mrsi-ppm-slider"></div>`;
}

// Chart.js spectrum plot. The chart instance is created once (empty deps
// effect) and mutated in place on every voxel/range change rather than
// recreated, so panning/zooming/voxel-picking stays cheap.
function SpectrumPlot({ ppm, spectrumData, ppmMin, ppmMax, label }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  // Rescale the y-axis to whatever's actually visible in [ppmMin, ppmMax],
  // not the full spectrum's amplitude range -- otherwise zooming the ppm
  // window in doesn't reveal any more vertical detail.
  const yBounds = useMemo(() => {
    const lo = Math.min(ppmMin, ppmMax);
    const hi = Math.max(ppmMin, ppmMax);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < ppm.length; i++) {
      if (ppm[i] < lo || ppm[i] > hi) continue;
      const v = spectrumData[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (!(min < max)) {
      min -= 1;
      max += 1;
    }
    const pad = (max - min) * 0.05;
    return { min: min - pad, max: max + pad };
  }, [ppm, spectrumData, ppmMin, ppmMax]);

  useEffect(() => {
    if (!canvasRef.current) return undefined;
    const ctx = canvasRef.current.getContext("2d");
    // Canvas 2D strokeStyle doesn't resolve CSS var() the way an element's
    // style does -- read the theme token's computed value instead.
    const accentColor = getComputedStyle(canvasRef.current).getPropertyValue("--mnu-accent").trim() || "#0055aa";

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels: ppm,
        datasets: [
          {
            label: label || "Spectrum",
            data: spectrumData,
            borderColor: accentColor,
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            title: { display: true, text: "ppm" },
            reverse: true, // NMR/MRS convention: high ppm on the left
            min: Math.min(ppmMin, ppmMax),
            max: Math.max(ppmMin, ppmMax),
            ticks: { maxTicksLimit: 8 },
          },
          y: {
            min: yBounds.min,
            max: yBounds.max,
            title: { display: true, text: "Intensity (a.u.)" },
            ticks: { callback: (val) => val.toExponential(1) },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: (ctx) => `Intensity: ${ctx.parsed.y.toExponential(3)}` },
          },
        },
      },
    });

    return () => {
      chartRef.current?.destroy();
      chartRef.current = null;
    };
    // Chart created once; per-update mutation happens in the effect below.
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.data.labels = ppm;
    chart.data.datasets[0].data = spectrumData;
    chart.options.scales.x.min = Math.min(ppmMin, ppmMax);
    chart.options.scales.x.max = Math.max(ppmMin, ppmMax);
    chart.options.scales.y.min = yBounds.min;
    chart.options.scales.y.max = yBounds.max;
    chart.update("none"); // instant, no animation overhead
  }, [ppm, spectrumData, ppmMin, ppmMax, yBounds]);

  return html`<canvas ref=${canvasRef}></canvas>`;
}

function MRSIInspector({ data }) {
  const containerRef = useRef(null);

  const [nx, ny, nMrsiSlices] = data.grid_shape;
  const [dimsI, dimsJ] = data.mrsi_dims;

  const [sliceIdx, setSliceIdx] = useState(data.initial_slice);
  const [voxel, setVoxel] = useState({
    x: data.initial_voxel[0],
    y: data.initial_voxel[1],
    mrsiSliceIdx: data.initial_voxel[2],
  });

  const ppmDataMin = Math.min(data.ppm[0], data.ppm[data.npts - 1]);
  const ppmDataMax = Math.max(data.ppm[0], data.ppm[data.npts - 1]);
  const ppmStep = data.npts > 1 ? Math.abs(data.ppm[1] - data.ppm[0]) || 0.01 : 0.01;

  // Default to the conventional 10 to -2 ppm display window (matches
  // plotting/spectra.py's default xlim), clamped to what the data covers.
  const [ppmRange, setPpmRange] = useState({
    min: clamp(-2, ppmDataMin, ppmDataMax),
    max: clamp(10, ppmDataMin, ppmDataMax),
  });

  // `data.spectra` is a Uint16Array of raw float16 bit patterns and
  // `data.spectra_scale` undoes the range scaling applied before the cast --
  // both prepared by `renderWidget` below, which owns the async decode.
  const currentSpectrum = useMemo(() => {
    const { x, y, mrsiSliceIdx } = voxel;
    const offset = ((x * ny + y) * nMrsiSlices + mrsiSliceIdx) * data.npts;
    // Only this voxel's points are converted, not the whole grid: a 16x16x16
    // grid at 700 points is 2.9M values, and all but `npts` of them would be
    // thrown away on every voxel change.
    const out = new Array(data.npts);
    for (let i = 0; i < data.npts; i++) {
      out[i] = halfToFloat(data.spectra[offset + i]) * data.spectra_scale;
    }
    return out;
  }, [voxel, data.spectra, data.spectra_scale, ny, nMrsiSlices, data.npts]);

  const voxelPolygonPoints = useMemo(() => {
    const cornerOffsets = [
      [-0.5, -0.5, 0],
      [0.5, -0.5, 0],
      [0.5, 0.5, 0],
      [-0.5, 0.5, 0],
    ];
    return cornerOffsets
      .map(([dx, dy, dz]) => {
        const [row, col] = applyAffine(data.mrsi_to_display_affine, [
          voxel.x + dx,
          voxel.y + dy,
          voxel.mrsiSliceIdx + dz,
        ]);
        return `${col},${row}`;
      })
      .join(" ");
  }, [voxel, data.mrsi_to_display_affine]);

  const handleSliceChange = (newSliceIdx) => {
    const clampedSlice = clamp(newSliceIdx, 0, data.n_anat_slices - 1);
    setSliceIdx(clampedSlice);

    const [, , mrsiSlice] = applyAffine(data.display_to_mrsi_affine, [0, 0, clampedSlice]);
    setVoxel((v) => ({
      ...v,
      mrsiSliceIdx: clamp(Math.round(mrsiSlice), 0, nMrsiSlices - 1),
    }));
  };

  const handleImageClick = (e) => {
    containerRef.current?.focus();
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const col = -0.5 + (px / rect.width) * data.image_width;
    const row = -0.5 + (py / rect.height) * data.image_height;

    const [rawX, rawY, rawSlice] = applyAffine(data.display_to_mrsi_affine, [row, col, sliceIdx]);
    setVoxel({
      x: clamp(Math.round(rawX), 0, nx - 1),
      y: clamp(Math.round(rawY), 0, ny - 1),
      mrsiSliceIdx: clamp(Math.round(rawSlice), 0, nMrsiSlices - 1),
    });
  };

  // `Capture`-suffixed Preact event props register on the capture phase, not
  // bubble: notebook hosts (VS Code, JupyterLab) bind ArrowUp/ArrowDown at
  // the document level for cell navigation, and a plain bubble-phase
  // listener plus preventDefault() doesn't stop that outer handler from
  // also firing on the same event.
  const handleKeyDownCapture = (e) => {
    if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) return;
    e.preventDefault();
    e.stopPropagation();
    setVoxel((v) => {
      let nextX = v.x;
      let nextY = v.y;
      if (e.key === "ArrowDown") nextY = Math.max(0, v.y - 1);
      if (e.key === "ArrowUp") nextY = Math.min(ny - 1, v.y + 1);
      if (e.key === "ArrowRight") nextX = Math.max(0, v.x - 1);
      if (e.key === "ArrowLeft") nextX = Math.min(nx - 1, v.x + 1);
      return { ...v, x: nextX, y: nextY };
    });
  };

  const specI = dimsI - 1 - voxel.y;
  const specJ = dimsJ - 1 - voxel.x;

  return html`
    <div
      ref=${containerRef}
      className="mnu-viewer mnutils-mrsi-inspector"
      tabindex="0"
      onMouseEnter=${() => containerRef.current?.focus()}
      onKeyDownCapture=${handleKeyDownCapture}>
      <div className="mnutils-mrsi-left-panel">
        <div className="mnu-lbl mnutils-mrsi-slice-title">${data.slice_titles[sliceIdx]}</div>
        <div className="mnu-panel mnutils-mrsi-image-wrap">
          <img
            className="mnutils-mrsi-image"
            src=${data.frame_urls[sliceIdx]}
            onClick=${handleImageClick}
            title="Click to select voxel" />
          <svg
            className="mnutils-mrsi-overlay"
            viewBox="-0.5 -0.5 ${data.image_width} ${data.image_height}">
            <polygon className="mnutils-mrsi-voxel-box" points=${voxelPolygonPoints} />
          </svg>
        </div>
        <div className="mnu-bar">
          <input
            type="range"
            className="mnu-slider"
            min="0"
            max=${data.n_anat_slices - 1}
            value=${sliceIdx}
            disabled=${data.n_anat_slices <= 1}
            onInput=${(e) => handleSliceChange(Number(e.target.value))} />
        </div>
      </div>

      <div className="mnutils-mrsi-right-panel">
        <div className="mnu-lbl mnutils-mrsi-spectrum-title">
          Spectrum at voxel (i:${specI}, j:${specJ}, slice:${voxel.mrsiSliceIdx})
        </div>
        <div className="mnutils-mrsi-spectrum-container">
          <${SpectrumPlot}
            ppm=${data.ppm}
            spectrumData=${currentSpectrum}
            ppmMin=${ppmRange.min}
            ppmMax=${ppmRange.max}
            label=${data.spectrum_label} />
        </div>
        <div className="mnu-bar mnutils-mrsi-range-bar">
          <span className="mnu-lbl">ppm:</span>
          <${DualRangeSlider}
            min=${ppmDataMin}
            max=${ppmDataMax}
            step=${ppmStep}
            valueMin=${ppmRange.min}
            valueMax=${ppmRange.max}
            onChange=${(min, max) => setPpmRange({ min, max })} />
          <span className="mnu-readout">${ppmRange.min.toFixed(2)} – ${ppmRange.max.toFixed(2)}</span>
        </div>
      </div>
    </div>
  `;
}

/**
 * Turn a backend's raw payload into the shape `MRSIInspector` consumes.
 *
 * Both backends deliver the same fields, differing only in how the two heavy
 * ones travel (base64 text vs. comm buffers), so `toBytes` absorbs that and
 * everything downstream -- the whole component -- is backend-agnostic. Frames
 * become blob: URLs rather than data: URLs so the browser holds one copy of
 * each frame instead of a base64 string plus its decoded form.
 */
async function decodePayload(data) {
  const mime = data.frame_mime || "image/webp";
  const frame_urls = data.left_frames.map((frame) =>
    URL.createObjectURL(new Blob([toBytes(frame)], { type: mime })),
  );
  const raw = toBytes(data.spectra_bytes);
  const inflated = data.spectra_encoding === "zlib-float16" ? await inflate(raw) : raw;
  // Copy through a fresh buffer: the inflated bytes are not guaranteed to sit
  // at a 2-byte-aligned offset, which a Uint16Array view requires.
  const spectra = new Uint16Array(
    inflated.byteOffset % 2 === 0
      ? inflated.buffer.slice(inflated.byteOffset, inflated.byteOffset + inflated.byteLength)
      : Uint8Array.from(inflated).buffer,
  );
  return { ...data, frame_urls, spectra, spectra_scale: data.spectra_scale ?? 1 };
}

// Deliberately not exported: `render_html` calls this directly from the same
// module scope, and the anywidget adapter (concatenated after this file) owns
// the module's single `export default`. Two default exports would not parse.
function renderWidget(data, el) {
  el.textContent = "Loading MRSI inspector…";
  decodePayload(data).then(
    (decoded) => {
      el.textContent = "";
      render(html`<${MRSIInspector} data=${decoded} />`, el);
    },
    (err) => {
      el.textContent = `Failed to decode MRSI inspector data: ${err}`;
      throw err;
    },
  );
}
