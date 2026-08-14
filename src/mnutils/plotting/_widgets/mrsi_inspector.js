// Preact + htm (JSX-free templating) + Chart.js, loaded straight from a CDN:
// no bundler in this project (see _shared/__init__.py), and this widget's
// interaction (dual-range zoom, per-voxel redraw) is enough state to want a
// real component model rather than hand-rolled DOM diffing.
import Chart from "https://esm.sh/chart.js@4.4.1/auto";
import htm from "https://esm.sh/htm@3.1.1";
import { h, render } from "https://esm.sh/preact@10.19.6";
import { useEffect, useMemo, useRef, useState } from "https://esm.sh/preact@10.19.6/hooks";

const html = htm.bind(h);

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

// Dual-thumb ppm range slider. Markup/classes mirror _shared/dom.js's
// `makeDualRangeSlider` (`.mnu-dual-range*`) so it picks up the same themed,
// dark-mode-aware styling from _shared/theme.css instead of a second,
// hand-rolled (and non-themed) implementation.
function DualRangeSlider({ min, max, step, valueMin, valueMax, onChange }) {
  const minRef = useRef(null);
  const maxRef = useRef(null);

  const handleMinInput = (e) => {
    const lo = Math.min(Number(e.target.value), valueMax - step);
    onChange(lo, valueMax);
  };
  const handleMaxInput = (e) => {
    const hi = Math.max(Number(e.target.value), valueMin + step);
    onChange(valueMin, hi);
  };

  // Overlapping thumbs both sit at the same screen position when their
  // values are close; whichever input the pointer actually went down on is
  // the one the user meant to grab, so bring it to the front for the drag.
  const bringToFront = (front, back) => {
    if (front.current) front.current.style.zIndex = "3";
    if (back.current) back.current.style.zIndex = "1";
  };

  const disabled = max <= min;
  const pct = (v) => (max === min ? 0 : ((v - min) / (max - min)) * 100);

  return html`
    <div className="mnu-dual-range">
      <div className="mnu-dual-range-track"></div>
      <div
        className="mnu-dual-range-fill"
        style=${{
          left: `${pct(valueMin)}%`,
          width: `${Math.max(0, pct(valueMax) - pct(valueMin))}%`,
        }}></div>
      <input
        ref=${minRef}
        type="range"
        className="mnu-slider mnu-dual-range-input"
        min=${min}
        max=${max}
        step=${step}
        value=${valueMin}
        disabled=${disabled}
        onPointerDown=${() => bringToFront(minRef, maxRef)}
        onInput=${handleMinInput} />
      <input
        ref=${maxRef}
        type="range"
        className="mnu-slider mnu-dual-range-input"
        min=${min}
        max=${max}
        step=${step}
        value=${valueMax}
        disabled=${disabled}
        onPointerDown=${() => bringToFront(maxRef, minRef)}
        onInput=${handleMaxInput} />
    </div>
  `;
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

  const spectraBuf = useMemo(
    () => new Float32Array(b64ToBytes(data.spectra_bytes).buffer),
    [data.spectra_bytes],
  );

  const currentSpectrum = useMemo(() => {
    const { x, y, mrsiSliceIdx } = voxel;
    const offset = ((x * ny + y) * nMrsiSlices + mrsiSliceIdx) * data.npts;
    return Array.from(spectraBuf.subarray(offset, offset + data.npts));
  }, [voxel, spectraBuf, ny, nMrsiSlices, data.npts]);

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
            src="data:image/png;base64,${data.left_frames[sliceIdx]}"
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

export default function renderWidget(data, el) {
  render(html`<${MRSIInspector} data=${data} />`, el);
}
