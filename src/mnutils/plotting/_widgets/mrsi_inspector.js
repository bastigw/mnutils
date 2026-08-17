// Preact + htm (JSX-free templating) + Chart.js, loaded straight from a CDN:
// no bundler in this project (see _shared/__init__.py), and this widget's
// interaction (dual-range zoom, per-voxel redraw) is enough state to want a
// real component model rather than hand-rolled DOM diffing.
import Chart from 'https://esm.sh/chart.js@4.4.1/auto'
import htm from 'https://esm.sh/htm@3.1.1'
import noUiSlider from 'https://esm.sh/nouislider@15.8.1'
import { h, render } from 'https://esm.sh/preact@10.19.6'
import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'https://esm.sh/preact@10.19.6/hooks'

const html = htm.bind(h)

const NOUISLIDER_CSS_HREF =
  'https://cdn.jsdelivr.net/npm/nouislider@15.8.1/dist/nouislider.min.css'

// nouislider ships its own stylesheet rather than being themeable via JS
// options; inject it once per document (widgets render into plain divs, not
// shadow roots, so a single document-level <link> covers every instance).
function ensureNoUiSliderCss() {
  if (document.querySelector(`link[href="${NOUISLIDER_CSS_HREF}"]`)) return
  const link = document.createElement('link')
  link.rel = 'stylesheet'
  link.href = NOUISLIDER_CSS_HREF
  document.head.appendChild(link)
}

// Apply a row-major 4x4 affine (flattened to 16 floats) to a 3-vector.
// MRSI-specific (the affine and its inverse are baked into this widget's
// data payload), so kept local rather than promoted to _shared/dom.js.
function applyAffine(affine, point) {
  const [x, y, z] = point
  return [
    affine[0] * x + affine[1] * y + affine[2] * z + affine[3],
    affine[4] * x + affine[5] * y + affine[6] * z + affine[7],
    affine[8] * x + affine[9] * y + affine[10] * z + affine[11],
  ]
}

// Dual-thumb range slider, backed by nouislider (dedicated dual-handle range
// library) instead of the two-overlaid-native-inputs hack. Used twice: the
// horizontal ppm window under the chart, and the vertical y-limit slider
// beside it. The instance is created once and driven imperatively thereafter:
// external prop changes go through `.set()`, while user gestures come back out
// through nouislider's 'slide' event (see the listener below for why that one).
// `onStart` fires once per gesture, before any value moves -- the y slider uses
// it to claim the axis on the first touch.
function DualRangeSlider({
  min,
  max,
  step,
  valueMin,
  valueMax,
  onChange,
  onStart,
  orientation = 'horizontal',
  className = 'mnutils-mrsi-ppm-slider',
}) {
  const containerRef = useRef(null)
  const sliderRef = useRef(null)
  const draggingRef = useRef(false)
  const onChangeRef = useRef(onChange)
  onChangeRef.current = onChange
  const onStartRef = useRef(onStart)
  onStartRef.current = onStart

  useEffect(() => {
    ensureNoUiSliderCss()
    if (!containerRef.current) return undefined

    const slider = noUiSlider.create(containerRef.current, {
      start: [valueMin, valueMax],
      connect: true,
      // nouislider throws on a zero-width range (a degenerate dataset: one ppm
      // point, or a grid of constant amplitude). Widen it here and let the
      // disable effect below make the control inert instead.
      range: { min, max: max > min ? max : min + 1 },
      step,
      orientation,
      behaviour: 'drag',
    })
    sliderRef.current = slider

    // 'slide', not 'update': nouislider fires 'update' on programmatic `set`
    // as well (unconditionally, ahead of the `fireSetEvent` guard), and once
    // more on create -- so listening to it turns the sync effect below into a
    // feedback loop that writes every externally-driven value straight back
    // into `onChange`. 'slide' fires only for drags, taps and keyboard steps.
    //
    // It still fires on every mousemove rather than once per step, so the same
    // snapped pair arrives dozens of times per drag. Forwarding each one cost
    // a re-render, a Chart.js repaint and (via the sync effect below) a
    // nouislider `set` that reads geometry back -- i.e. a forced reflow of the
    // whole panel per mouse event, which is what made dragging feel sticky.
    let lastLo = null
    let lastHi = null
    slider.on('slide', (values) => {
      const [lo, hi] = values.map(Number)
      if (lo === lastLo && hi === lastHi) return
      lastLo = lo
      lastHi = hi
      onChangeRef.current(lo, hi)
    })

    // While the user drags, nouislider's own handle positions are already
    // authoritative; writing them back mid-gesture only thrashes layout.
    slider.on('start', () => {
      draggingRef.current = true
      onStartRef.current?.()
    })
    slider.on('end', () => {
      draggingRef.current = false
    })

    return () => {
      slider.destroy()
      sliderRef.current = null
    }
    // Range bounds/step come from the dataset and never change after mount;
    // only valueMin/valueMax move thereafter, synced imperatively below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [min, max, step, orientation])

  useEffect(() => {
    if (draggingRef.current) return
    sliderRef.current?.set([valueMin, valueMax], false)
  }, [valueMin, valueMax])

  useEffect(() => {
    const slider = sliderRef.current
    if (!slider) return
    if (max <= min) slider.disable()
    else slider.enable()
  }, [min, max])

  return html`<div
    ref=${containerRef}
    className="mnutils-mrsi-slider ${className}"></div>`
}

// Chart.js spectrum plot. The chart instance is created once (empty deps
// effect) and mutated in place on every voxel/range change rather than
// recreated, so panning/zooming/voxel-picking stays cheap.
function SpectrumPlot({ ppm, spectrumData, ppmMin, ppmMax, label, yBounds }) {
  const canvasRef = useRef(null)
  const chartRef = useRef(null)

  // {x, y} points against a linear x scale, NOT scalars against `labels`.
  // With `labels` + scalar data Chart.js picks a *category* scale, where
  // `scales.x.min/max` are category indices rather than data values -- so a
  // ppm window of [-2, 10] silently rendered array indices 0..10 instead
  // (11 points near -122 ppm, out of 700), while the slider readout beside
  // the chart still showed the ppm the user asked for.
  const points = useMemo(
    () => ppm.map((p, i) => ({ x: p, y: spectrumData[i] })),
    [ppm, spectrumData],
  )

  useEffect(() => {
    if (!canvasRef.current) return undefined
    const ctx = canvasRef.current.getContext('2d')
    // Canvas 2D strokeStyle doesn't resolve CSS var() the way an element's
    // style does -- read the theme token's computed value instead.
    const accentColor =
      getComputedStyle(canvasRef.current)
        .getPropertyValue('--mnu-accent')
        .trim() || '#0055aa'

    chartRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [
          {
            label: label || 'Spectrum',
            data: points,
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
            type: 'linear', // ppm is a continuous axis, not 700 categories
            title: { display: true, text: 'ppm' },
            reverse: true, // NMR/MRS convention: high ppm on the left
            min: Math.min(ppmMin, ppmMax),
            max: Math.max(ppmMin, ppmMax),
            ticks: {
              maxTicksLimit: 8,
              callback: (val) => Number(val).toFixed(2),
            },
          },
          y: {
            min: yBounds.min,
            max: yBounds.max,
            title: { display: true, text: 'Intensity (a.u.)' },
            ticks: { callback: (val) => val.toExponential(1) },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => `${Number(items[0].parsed.x).toFixed(3)} ppm`,
              label: (ctx) => `Intensity: ${ctx.parsed.y.toExponential(3)}`,
            },
          },
        },
      },
    })

    return () => {
      chartRef.current?.destroy()
      chartRef.current = null
    }
    // Chart created once; per-update mutation happens in the effect below.
  }, [])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    chart.data.datasets[0].data = points
    chart.options.scales.x.min = Math.min(ppmMin, ppmMax)
    chart.options.scales.x.max = Math.max(ppmMin, ppmMax)
    chart.options.scales.y.min = yBounds.min
    chart.options.scales.y.max = yBounds.max
    chart.update('none') // instant, no animation overhead
  }, [points, ppmMin, ppmMax, yBounds])

  return html`<canvas ref=${canvasRef}></canvas>`
}

function MRSIInspector({ data }) {
  const containerRef = useRef(null)

  const [nx, ny, nMrsiSlices] = data.grid_shape
  const [dimsI, dimsJ] = data.mrsi_dims

  const [sliceIdx, setSliceIdx] = useState(data.initial_slice)
  const [voxel, setVoxel] = useState({
    x: data.initial_voxel[0],
    y: data.initial_voxel[1],
    mrsiSliceIdx: data.initial_voxel[2],
  })

  const ppmDataMin = Math.min(data.ppm[0], data.ppm[data.npts - 1])
  const ppmDataMax = Math.max(data.ppm[0], data.ppm[data.npts - 1])

  // Whole-ppm steps only: a window is something you read off in ppm ("2 to 8"),
  // and stepping by the sample spacing (~0.02 ppm) gave the slider hundreds of
  // indistinguishable positions. The bounds move inwards to the nearest
  // integers so the steps actually land on round numbers rather than on
  // `ppmDataMin + k`. Data spanning less than 1 ppm has no integer grid to
  // snap to, so it keeps the sample spacing.
  const ppmIntMin = Math.ceil(ppmDataMin)
  const ppmIntMax = Math.floor(ppmDataMax)
  const wholePpmSteps = ppmIntMax > ppmIntMin
  const ppmSliderMin = wholePpmSteps ? ppmIntMin : ppmDataMin
  const ppmSliderMax = wholePpmSteps ? ppmIntMax : ppmDataMax
  const ppmStep = wholePpmSteps
    ? 1
    : data.npts > 1
      ? Math.abs(data.ppm[1] - data.ppm[0]) || 0.01
      : 0.01

  // The chart's x axis is reversed (NMR convention: high ppm left), so the
  // slider has to run the same way or dragging the left thumb moves the right
  // edge of the window. nouislider's own `direction: 'rtl'` also reverses
  // handle order in every get/set payload; mirroring the value about the
  // midpoint of the slider range instead keeps the slider a plain ascending
  // ltr one and confines the flip to this single involution (mirror(mirror(p))
  // === p, so the same helper converts both ways). Mirroring about the
  // *slider* bounds, not the data bounds, is what keeps integers integral.
  const mirrorPpm = (p) => ppmSliderMin + ppmSliderMax - p
  const ppmDecimals = wholePpmSteps ? 0 : 2

  // Default to the conventional 10 to -2 ppm display window (matches
  // plotting/spectra.py's default xlim), clamped to what the data covers.
  const [ppmRange, setPpmRange] = useState({
    min: clamp(-2, ppmSliderMin, ppmSliderMax),
    max: clamp(10, ppmSliderMin, ppmSliderMax),
  })

  // `data.spectra` is a Uint16Array of raw float16 bit patterns and
  // `data.spectra_scale` undoes the range scaling applied before the cast --
  // both prepared by `renderWidget` below, which owns the async decode.
  const currentSpectrum = useMemo(() => {
    const { x, y, mrsiSliceIdx } = voxel
    const offset = ((x * ny + y) * nMrsiSlices + mrsiSliceIdx) * data.npts
    // Only this voxel's points are converted, not the whole grid: a 16x16x16
    // grid at 700 points is 2.9M values, and all but `npts` of them would be
    // thrown away on every voxel change.
    const out = new Array(data.npts)
    for (let i = 0; i < data.npts; i++) {
      out[i] = halfToFloat(data.spectra[offset + i]) * data.spectra_scale
    }
    return out
  }, [voxel, data.spectra, data.spectra_scale, ny, nMrsiSlices, data.npts])

  // Per-point min/max across *every* voxel, computed once on first use and
  // cached: the fixed-y axis has to cover the loudest voxel anywhere in the
  // grid, and recomputing that on each ppm-slider step (2.9M half-float
  // conversions for a 16^3 grid at 700 points) would stall the drag. Reducing
  // to two length-`npts` envelopes up front makes every later window change an
  // O(npts) scan in `SpectrumPlot`.
  const [fixedY, setFixedY] = useState(false)
  const envelopeRef = useRef(null)
  const yEnvelope = useMemo(() => {
    if (!fixedY) return null
    if (envelopeRef.current?.source === data.spectra) return envelopeRef.current
    const { npts, spectra, spectra_scale } = data
    const min = new Float64Array(npts).fill(Infinity)
    const max = new Float64Array(npts).fill(-Infinity)
    const nVoxels = Math.floor(spectra.length / npts)
    for (let v = 0; v < nVoxels; v++) {
      const offset = v * npts
      for (let i = 0; i < npts; i++) {
        const val = halfToFloat(spectra[offset + i]) * spectra_scale
        if (val < min[i]) min[i] = val
        if (val > max[i]) max[i] = val
      }
    }
    envelopeRef.current = { min, max, source: spectra }
    return envelopeRef.current
  }, [fixedY, data.spectra, data.spectra_scale, data.npts])

  // Manual y limits, set by dragging the vertical slider; `null` means the
  // axis is still derived from the data (per voxel, or -- with `fixedY` --
  // from the whole-grid envelope).
  const [yLim, setYLim] = useState(null)

  // Bounds of whatever is visible in [ppmMin, ppmMax], not of the full sweep:
  // otherwise zooming the ppm window in doesn't reveal any more vertical
  // detail. With `yEnvelope` given (fixed-y mode) they come from the whole
  // grid's per-point min/max instead of this one voxel, so the axis stops
  // moving as you step between voxels and their amplitudes stay comparable.
  // Still restricted to the ppm window: "fixed" means independent of the
  // voxel, not of the zoom -- a window-wide axis would flatten every peak
  // whenever one voxel somewhere has a huge off-window artefact.
  const autoYBounds = useMemo(() => {
    const ppm = data.ppm
    const lo = Math.min(ppmRange.min, ppmRange.max)
    const hi = Math.max(ppmRange.min, ppmRange.max)
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < ppm.length; i++) {
      if (ppm[i] < lo || ppm[i] > hi) continue
      const vLo = yEnvelope ? yEnvelope.min[i] : currentSpectrum[i]
      const vHi = yEnvelope ? yEnvelope.max[i] : currentSpectrum[i]
      if (vLo < min) min = vLo
      if (vHi > max) max = vHi
    }
    // A window narrower than one ppm step can contain no sample at all, which
    // leaves min/max at +/-Infinity -- widening those by 1 keeps them infinite
    // and hands Chart.js an unrenderable y axis. Fall back to a finite range.
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      return { min: 0, max: 1 }
    }
    if (!(min < max)) {
      min -= 1
      max += 1
    }
    const pad = (max - min) * 0.05
    return { min: min - pad, max: max + pad }
  }, [data.ppm, currentSpectrum, ppmRange.min, ppmRange.max, yEnvelope])

  // Manual limits win outright -- no 5% padding on top of them, since they are
  // the numbers the user dialled in.
  const yBounds = yLim ?? autoYBounds

  // The y slider travels over the grid's full amplitude range, which Python
  // measured while encoding the buffer: the browser would otherwise have to
  // scan every voxel before it could draw the control. 400 steps is finer than
  // the track is wide in pixels, so the thumb never feels notched.
  const yDataMin = data.spectra_min
  const yDataMax = data.spectra_max
  const yStep = (yDataMax - yDataMin) / 400 || 1

  // Vertical nouislider runs top-down (lowest value at the top) under
  // `direction: 'ltr'`, the opposite of a y axis. Same fix as the ppm slider:
  // keep the slider plain-ascending and mirror the values about the range
  // midpoint, an involution that converts in both directions.
  const mirrorY = (v) => yDataMin + yDataMax - v

  // With no manual limits set the slider sits fully open rather than tracking
  // the automatic bounds. Tracking them looks tidier but is unusable: a quiet
  // voxel's window is a couple of percent of the grid's amplitude range, so
  // both handles land inside the same 16 px and read as a single one. Fully
  // open also says the right thing -- nothing is being clamped yet.
  const ySliderLo = yLim
    ? clamp(mirrorY(yLim.max), yDataMin, yDataMax)
    : yDataMin
  const ySliderHi = yLim
    ? clamp(mirrorY(yLim.min), yDataMin, yDataMax)
    : yDataMax

  // Taking hold of the slider claims the axis: hand-set limits only mean
  // anything if it stops rescaling itself. The limits start at what the handles
  // already show (the full range), so the axis matches the control from the
  // first frame of the gesture instead of jumping to it on the first step.
  const claimYAxis = () => {
    setFixedY(true)
    setYLim((prev) => prev ?? { min: yDataMin, max: yDataMax })
  }

  const voxelPolygonPoints = useMemo(() => {
    const cornerOffsets = [
      [-0.5, -0.5, 0],
      [0.5, -0.5, 0],
      [0.5, 0.5, 0],
      [-0.5, 0.5, 0],
    ]
    return cornerOffsets
      .map(([dx, dy, dz]) => {
        const [row, col] = applyAffine(data.mrsi_to_display_affine, [
          voxel.x + dx,
          voxel.y + dy,
          voxel.mrsiSliceIdx + dz,
        ])
        return `${col},${row}`
      })
      .join(' ')
  }, [voxel, data.mrsi_to_display_affine])

  const handleSliceChange = (newSliceIdx) => {
    const clampedSlice = clamp(newSliceIdx, 0, data.n_anat_slices - 1)
    setSliceIdx(clampedSlice)

    const [, , mrsiSlice] = applyAffine(data.display_to_mrsi_affine, [
      0,
      0,
      clampedSlice,
    ])
    setVoxel((v) => ({
      ...v,
      mrsiSliceIdx: clamp(Math.round(mrsiSlice), 0, nMrsiSlices - 1),
    }))
  }

  const handleImageClick = (e) => {
    containerRef.current?.focus()
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const col = -0.5 + (px / rect.width) * data.image_width
    const row = -0.5 + (py / rect.height) * data.image_height

    const [rawX, rawY, rawSlice] = applyAffine(data.display_to_mrsi_affine, [
      row,
      col,
      sliceIdx,
    ])
    setVoxel({
      x: clamp(Math.round(rawX), 0, nx - 1),
      y: clamp(Math.round(rawY), 0, ny - 1),
      mrsiSliceIdx: clamp(Math.round(rawSlice), 0, nMrsiSlices - 1),
    })
  }

  // `Capture`-suffixed Preact event props register on the capture phase, not
  // bubble: notebook hosts (VS Code, JupyterLab) bind ArrowUp/ArrowDown at
  // the document level for cell navigation, and a plain bubble-phase
  // listener plus preventDefault() doesn't stop that outer handler from
  // also firing on the same event.
  const handleKeyDownCapture = (e) => {
    if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key))
      return
    e.preventDefault()
    e.stopPropagation()
    setVoxel((v) => {
      let nextX = v.x
      let nextY = v.y
      if (e.key === 'ArrowDown') nextY = Math.max(0, v.y - 1)
      if (e.key === 'ArrowUp') nextY = Math.min(ny - 1, v.y + 1)
      if (e.key === 'ArrowRight') nextX = Math.max(0, v.x - 1)
      if (e.key === 'ArrowLeft') nextX = Math.min(nx - 1, v.x + 1)
      return { ...v, x: nextX, y: nextY }
    })
  }

  // Image and chart share one fixed-height media row (`--mnu-mrsi-media-h`, set
  // in CSS) so they line up top and bottom regardless of the frame's aspect
  // ratio. That only holds if nothing but the title sits above either of them,
  // which is why the fixed-y toggle renders under the ppm bar rather than
  // between the title and the chart.
  const specI = dimsI - 1 - voxel.y
  const specJ = dimsJ - 1 - voxel.x

  return html`
    <div
      ref=${containerRef}
      className="mnu-viewer mnutils-mrsi-inspector"
      tabindex="0"
      onMouseEnter=${() => containerRef.current?.focus()}
      onKeyDownCapture=${handleKeyDownCapture}>
      <div className="mnutils-mrsi-left-panel">
        <div className="mnu-lbl mnutils-mrsi-slice-title">
          ${data.slice_titles[sliceIdx]}
        </div>
        <div className="mnutils-mrsi-media">
          <div
            className="mnutils-mrsi-image-wrap"
            style="--mnu-mrsi-ar: ${data.image_width / data.image_height}">
            <img
              className="mnutils-mrsi-image"
              src=${data.frame_urls[sliceIdx]}
              onClick=${handleImageClick}
              title="Click to select voxel" />
            <svg
              className="mnutils-mrsi-overlay"
              viewBox="-0.5 -0.5 ${data.image_width} ${data.image_height}">
              <polygon
                className="mnutils-mrsi-voxel-box-halo"
                points=${voxelPolygonPoints} />
              <polygon
                className="mnutils-mrsi-voxel-box"
                points=${voxelPolygonPoints} />
            </svg>
          </div>
        </div>
        <div className="mnu-bar mnutils-mrsi-control-bar">
          <span className="mnu-lbl">slice:</span>
          <input
            type="range"
            className="mnu-slider"
            min="0"
            max=${data.n_anat_slices - 1}
            value=${sliceIdx}
            disabled=${data.n_anat_slices <= 1}
            onInput=${(e) => handleSliceChange(Number(e.target.value))} />
          <span className="mnu-readout"
            >${sliceIdx + 1} / ${data.n_anat_slices}</span
          >
        </div>
      </div>

      <div className="mnutils-mrsi-right-panel">
        <div className="mnu-lbl mnutils-mrsi-spectrum-title">
          Spectrum at voxel (i:${specI}, j:${specJ},
          slice:${voxel.mrsiSliceIdx})
        </div>
        <div className="mnutils-mrsi-media">
          <${DualRangeSlider}
            orientation="vertical"
            className="mnutils-mrsi-ylim-slider"
            min=${yDataMin}
            max=${yDataMax}
            step=${yStep}
            valueMin=${ySliderLo}
            valueMax=${ySliderHi}
            onStart=${claimYAxis}
            onChange=${(lo, hi) => {
              // Also here, not just in `onStart`: a tap on the track moves a
              // handle without a preceding drag gesture on some inputs.
              claimYAxis()
              setYLim({ min: mirrorY(hi), max: mirrorY(lo) })
            }} />
          <div className="mnutils-mrsi-spectrum-container">
            <${SpectrumPlot}
              ppm=${data.ppm}
              spectrumData=${currentSpectrum}
              ppmMin=${ppmRange.min}
              ppmMax=${ppmRange.max}
              label=${data.spectrum_label}
              yBounds=${yBounds} />
          </div>
        </div>
        <div
          className="mnu-bar mnutils-mrsi-control-bar mnutils-mrsi-range-bar">
          <span className="mnu-lbl">ppm:</span>
          <${DualRangeSlider}
            min=${ppmSliderMin}
            max=${ppmSliderMax}
            step=${ppmStep}
            valueMin=${mirrorPpm(ppmRange.max)}
            valueMax=${mirrorPpm(ppmRange.min)}
            onChange=${(lo, hi) =>
              setPpmRange({ min: mirrorPpm(hi), max: mirrorPpm(lo) })} />
          <span className="mnu-readout"
            >${ppmRange.max.toFixed(ppmDecimals)} –
            ${ppmRange.min.toFixed(ppmDecimals)}</span
          >
        </div>
        <label className="mnutils-mrsi-ylim-toggle">
          <input
            type="checkbox"
            checked=${fixedY}
            onChange=${(e) => {
              setFixedY(e.target.checked)
              // Unticking is the only way back to the per-voxel autoscale, so
              // it has to drop any manual limits as well -- they would
              // otherwise keep the axis pinned with the box unticked.
              if (!e.target.checked) setYLim(null)
            }} />
          <span
            >Fixed y-axis
            (${yLim ? 'manual limits' : 'max over all voxels'})</span
          >
        </label>
      </div>
    </div>
  `
}

/**
 * Turn the raw payload baked in by `mrsi_inspector.py` into the shape
 * `MRSIInspector` consumes.
 *
 * The two heavy fields arrive base64-encoded: WebP frames, and the spectra as
 * a zlib-compressed block of float16 bit patterns. Frames become blob: URLs
 * rather than data: URLs so the browser holds one copy of each frame instead
 * of a base64 string plus its decoded form.
 */
async function decodePayload(data) {
  const frame_urls = data.left_frames.map((frame) =>
    URL.createObjectURL(new Blob([b64ToBytes(frame)], { type: 'image/webp' })),
  )
  const inflated = await inflate(b64ToBytes(data.spectra_bytes))
  // Copy through a fresh buffer: the inflated bytes are not guaranteed to sit
  // at a 2-byte-aligned offset, which a Uint16Array view requires.
  const spectra = new Uint16Array(
    inflated.byteOffset % 2 === 0
      ? inflated.buffer.slice(
          inflated.byteOffset,
          inflated.byteOffset + inflated.byteLength,
        )
      : Uint8Array.from(inflated).buffer,
  )
  return { ...data, frame_urls, spectra }
}

// Deliberately not exported: `render_html` calls this directly from the same
// module scope.
function renderWidget(data, el) {
  el.textContent = 'Loading MRSI inspector…'
  decodePayload(data).then(
    (decoded) => {
      el.textContent = ''
      render(html`<${MRSIInspector} data=${decoded} />`, el)
    },
    (err) => {
      el.textContent = `Failed to decode MRSI inspector data: ${err}`
      throw err
    },
  )
}
