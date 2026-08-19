/**
 * Shared frontend helpers for MNUtils plotting widgets.
 *
 * This module intentionally uses NO `import`/`export` statements: the Python
 * asset loader (`_shared/__init__.py::load_esm`) concatenates it *ahead* of a
 * widget's own `<name>.js`, so both end up in one module scope and every
 * widget can call these helpers directly.
 */

/** Decode a base64 string (as produced by Python's base64.b64encode) to bytes. */
function b64ToBytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

/**
 * Inflate a zlib stream produced by Python's `zlib.compress`.
 *
 * Uses the platform's own DecompressionStream, so decompressing the spectra
 * buffer costs no JS dependency. `zlib.compress` emits the zlib wrapper, which
 * is what the "deflate" format expects ("deflate-raw" would be the headerless
 * variant).
 */
async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * Convert one IEEE-754 half-precision bit pattern to a JS number.
 *
 * Spectra travel as float16 to halve the payload. They are decoded a single
 * voxel at a time rather than a whole grid at a time, so this stays cheap and
 * there is no reason to depend on `Float16Array` (only Baseline 2025, and
 * these widgets have to render on whatever browser opens the docs page).
 */
function halfToFloat(half) {
  const sign = (half & 0x8000) ? -1 : 1;
  const exponent = (half & 0x7c00) >> 10;
  const fraction = half & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

/** Clamp `value` into the inclusive [lo, hi] range. */
function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

/** Build a `<div class="mnu-bar">` control row from a list of child elements. */
function makeBar(...children) {
  const bar = document.createElement("div");
  bar.className = "mnu-bar";
  bar.append(...children);
  return bar;
}

/** Build a `<div class="mnu-grp">` grouping a related set of controls. */
function makeGroup(...children) {
  const grp = document.createElement("div");
  grp.className = "mnu-grp";
  grp.append(...children);
  return grp;
}

/**
 * Build a labeled slice slider: `<span class="mnu-lbl">` + `<input
 * type="range" class="mnu-slider">` + `<span class="mnu-readout">`, wired so
 * moving the slider calls `onInput(index)` and updates the readout text via
 * `formatReadout(index)`.
 */
function makeSliceSlider({ min, max, value, step = 1, label, formatReadout, onInput }) {
  const lbl = document.createElement("span");
  lbl.className = "mnu-lbl";
  lbl.textContent = label;

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "mnu-slider";
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  slider.value = String(value);
  slider.disabled = max <= min;

  const readout = document.createElement("span");
  readout.className = "mnu-readout";
  readout.textContent = formatReadout(value);

  slider.addEventListener("input", () => {
    const idx = Number(slider.value);
    readout.textContent = formatReadout(idx);
    onInput(idx);
  });

  return { lbl, slider, readout };
}

/**
 * Build a pill-shaped checkbox: a `<label class="mnu-toggle">` wrapping an
 * `<input type="checkbox">` and its text, wired so ticking it calls
 * `onChange(checked)`.
 *
 * A label-wrapped checkbox rather than a styled button: it stays a real
 * checkbox for assistive technology and for keyboard users, and the whole
 * pill becomes the hit target because the input sits inside the label.
 */
function makeToggle({ label, checked = false, onChange }) {
  const toggle = document.createElement("label");
  toggle.className = "mnu-toggle";

  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = checked;

  const text = document.createElement("span");
  text.textContent = label;

  input.addEventListener("change", () => onChange(input.checked));
  toggle.append(input, text);

  return { toggle, input };
}

