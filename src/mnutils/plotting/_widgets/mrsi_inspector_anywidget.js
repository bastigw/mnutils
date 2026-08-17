/**
 * anywidget adapter for the shared MRSI inspector component.
 *
 * `_shared/__init__.py::load_esm` concatenates this *after* `mrsi_inspector.js`,
 * so `renderWidget` is already in scope. The only job here is to read the
 * traitlets off the model into the same plain object the HTML backend bakes
 * into its output, and to own the `export default` that anywidget looks for --
 * the component itself is shared byte-for-byte between the two backends.
 */

const KEYS = [
  "left_frames",
  "slice_titles",
  "n_anat_slices",
  "initial_slice",
  "image_width",
  "image_height",
  "mrsi_to_display_affine",
  "display_to_mrsi_affine",
  "grid_shape",
  "mrsi_dims",
  "initial_voxel",
  "ppm",
  "spectra_bytes",
  "spectra_scale",
  "npts",
  "spectrum_label",
  "transport",
  "frame_mime",
  "spectra_encoding",
];

export default {
  render({ model, el }) {
    const data = {};
    for (const key of KEYS) data[key] = model.get(key);
    // Everything is set once at construction and never mutated from Python,
    // so there is nothing to subscribe to -- the comm is used purely as a
    // one-shot binary transport. `renderWidget` handles the async decode.
    renderWidget(data, el);
  },
};
