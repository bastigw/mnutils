function renderWidget(data, el) {
  const frames = data.frames;
  const nSlices = data.n_slices;
  const label = data.slice_label;
  const initialIndex = data.initial_index;

  const urls = frames.map(
    (b64) => URL.createObjectURL(new Blob([b64ToBytes(b64)], { type: "image/png" })),
  );

  const viewer = document.createElement("div");
  viewer.className = "mnu-viewer mnutils-slice-viewer";

  const panel = document.createElement("div");
  panel.className = "mnu-panel mnutils-slice-viewer-panel";

  const img = document.createElement("img");
  img.className = "mnutils-slice-viewer-image";
  img.src = urls[initialIndex];

  panel.append(img);

  const { lbl, slider, readout } = makeSliceSlider({
    min: 0,
    max: nSlices - 1,
    value: initialIndex,
    label,
    formatReadout: (idx) => String(idx),
    onInput: (idx) => {
      img.src = urls[idx];
    },
  });

  const bar = makeBar(makeGroup(lbl, readout), slider);
  bar.classList.add("mnutils-slice-viewer-bar");

  viewer.append(bar, panel);
  el.append(viewer);
}

export default renderWidget;
