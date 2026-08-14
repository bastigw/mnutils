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
function makeSliceSlider({ min, max, value, label, formatReadout, onInput }) {
  const lbl = document.createElement("span");
  lbl.className = "mnu-lbl";
  lbl.textContent = label;

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "mnu-slider";
  slider.min = String(min);
  slider.max = String(max);
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
