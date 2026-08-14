/**
 * Shared frontend helpers for MNUtils plotting widgets.
 *
 * This module intentionally uses NO `import`/`export` statements: the Python
 * asset loader (`_shared/__init__.py::load_esm`) concatenates it *ahead* of a
 * widget's own `<name>.js`, and the widget module owns the single
 * `export default renderWidget`. Keeping these helpers export-free lets them
 * live in the same module scope as `renderWidget` in every widget.
 */

/** Decode a base64 string (as produced by Python's base64.b64encode) to bytes. */
function b64ToBytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
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
 * Build a labeled two-knob range slider: `<span class="mnu-lbl">` + a
 * `<div class="mnu-dual-range">` holding two overlaid native
 * `<input type="range">` thumbs (the standard "two transparent-track
 * sliders, thumb-only pointer-events" technique for a dual-handle range
 * with no extra dependency) + a `<span class="mnu-readout">`. Moving either
 * knob clamps it against the other (keeping at least `step` between them)
 * and calls `onInput(loValue, hiValue)`.
 */
function makeDualRangeSlider({ min, max, valueMin, valueMax, step = 1, label, formatReadout, onInput }) {
  const lbl = document.createElement("span");
  lbl.className = "mnu-lbl";
  lbl.textContent = label;

  const container = document.createElement("div");
  container.className = "mnu-dual-range";

  const track = document.createElement("div");
  track.className = "mnu-dual-range-track";
  const fill = document.createElement("div");
  fill.className = "mnu-dual-range-fill";

  const minInput = document.createElement("input");
  minInput.type = "range";
  minInput.className = "mnu-slider mnu-dual-range-input";
  minInput.min = String(min);
  minInput.max = String(max);
  minInput.step = String(step);
  minInput.value = String(valueMin);

  const maxInput = document.createElement("input");
  maxInput.type = "range";
  maxInput.className = "mnu-slider mnu-dual-range-input";
  maxInput.min = String(min);
  maxInput.max = String(max);
  maxInput.step = String(step);
  maxInput.value = String(valueMax);

  const disabled = max <= min;
  minInput.disabled = disabled;
  maxInput.disabled = disabled;
  minInput.style.zIndex = "1";
  maxInput.style.zIndex = "2";

  container.append(track, fill, minInput, maxInput);

  const readout = document.createElement("span");
  readout.className = "mnu-readout";
  readout.textContent = formatReadout(valueMin, valueMax);

  function updateFill() {
    const lo = Number(minInput.value);
    const hi = Number(maxInput.value);
    const pct = (v) => (max === min ? 0 : ((v - min) / (max - min)) * 100);
    fill.style.left = `${pct(lo)}%`;
    fill.style.width = `${Math.max(0, pct(hi) - pct(lo))}%`;
  }
  updateFill();

  // Overlapping thumbs both sit at the same screen position when their
  // values are close; whichever input the pointer actually went down on is
  // the one the user meant to grab, so bring it to the front for the
  // duration of the drag.
  minInput.addEventListener("pointerdown", () => {
    minInput.style.zIndex = "3";
    maxInput.style.zIndex = "1";
  });
  maxInput.addEventListener("pointerdown", () => {
    maxInput.style.zIndex = "3";
    minInput.style.zIndex = "1";
  });

  minInput.addEventListener("input", () => {
    const hi = Number(maxInput.value);
    const lo = Math.min(Number(minInput.value), hi - step);
    minInput.value = String(lo);
    updateFill();
    readout.textContent = formatReadout(lo, hi);
    onInput(lo, hi);
  });
  maxInput.addEventListener("input", () => {
    const lo = Number(minInput.value);
    const hi = Math.max(Number(maxInput.value), lo + step);
    maxInput.value = String(hi);
    updateFill();
    readout.textContent = formatReadout(lo, hi);
    onInput(lo, hi);
  });

  return { lbl, container, minInput, maxInput, readout };
}
