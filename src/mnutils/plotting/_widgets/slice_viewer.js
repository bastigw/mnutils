function b64ToBytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

function renderWidget(data, el) {
  const frames = data.frames;
  const nSlices = data.n_slices;
  const label = data.slice_label;
  const initialIndex = data.initial_index;

  const urls = frames.map(
    (b64) => URL.createObjectURL(new Blob([b64ToBytes(b64)], { type: "image/png" })),
  );

  const container = document.createElement("div");
  container.className = "mnutils-slice-viewer";

  const img = document.createElement("img");
  img.className = "mnutils-slice-viewer-image";
  img.src = urls[initialIndex];

  const controls = document.createElement("div");
  controls.className = "mnutils-slice-viewer-controls";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(nSlices - 1);
  slider.value = String(initialIndex);
  slider.disabled = nSlices <= 1;

  const readout = document.createElement("span");
  readout.className = "mnutils-slice-viewer-readout";
  readout.textContent = `${label} ${slider.value}`;

  slider.addEventListener("input", () => {
    const idx = Number(slider.value);
    img.src = urls[idx];
    readout.textContent = `${label} ${idx}`;
  });

  controls.append(slider, readout);
  container.append(img, controls);
  el.append(container);
}

export default renderWidget;
