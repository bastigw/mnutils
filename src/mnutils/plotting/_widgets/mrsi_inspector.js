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

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

function b64ToBytes(b64) {
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

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
  const spectraBuf = new Float32Array(b64ToBytes(data.spectra_bytes).buffer);

  const frameUrls = leftFrames.map(
    (b64) => URL.createObjectURL(new Blob([b64ToBytes(b64)], { type: "image/png" })),
  );

  const state = {
    sliceIdx: data.initial_slice,
    x: data.initial_voxel[0],
    y: data.initial_voxel[1],
    mrsiSliceIdx: data.initial_voxel[2],
  };

  const container = document.createElement("div");
  container.className = "mnutils-mrsi-inspector";
  container.tabIndex = 0;

  const leftPanel = document.createElement("div");
  leftPanel.className = "mnutils-mrsi-left-panel";

  const imgWrap = document.createElement("div");
  imgWrap.className = "mnutils-mrsi-image-wrap";

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
  sliceTitle.className = "mnutils-mrsi-slice-title";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(nAnatSlices - 1);
  slider.value = String(state.sliceIdx);
  slider.disabled = nAnatSlices <= 1;

  leftPanel.append(imgWrap, sliceTitle, slider);

  const rightPanel = document.createElement("div");
  rightPanel.className = "mnutils-mrsi-right-panel";

  const spectrumTitle = document.createElement("div");
  spectrumTitle.className = "mnutils-mrsi-spectrum-title";

  const spectrumSvg = document.createElementNS(SVG_NS, "svg");
  spectrumSvg.classList.add("mnutils-mrsi-spectrum");
  const spectrumViewW = 400;
  const spectrumViewH = 200;
  spectrumSvg.setAttribute("viewBox", `0 0 ${spectrumViewW} ${spectrumViewH}`);
  spectrumSvg.setAttribute("preserveAspectRatio", "none");

  const spectrumLine = document.createElementNS(SVG_NS, "polyline");
  spectrumLine.setAttribute("class", "mnutils-mrsi-spectrum-line");
  spectrumSvg.append(spectrumLine);

  rightPanel.append(spectrumTitle, spectrumSvg);

  container.append(leftPanel, rightPanel);
  el.append(container);

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

  function drawSpectrum(x, y, mrsiSliceIdx) {
    const spectrum = spectrumAt(x, y, mrsiSliceIdx);
    let min = Infinity;
    let max = -Infinity;
    for (const v of spectrum) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const points = [];
    for (let i = 0; i < npts; i++) {
      const px = (i / (npts - 1)) * spectrumViewW;
      const py =
        spectrumViewH - ((spectrum[i] - min) / (max - min)) * spectrumViewH;
      points.push(`${px},${py}`);
    }
    spectrumLine.setAttribute("points", points.join(" "));

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
    const rect = img.getBoundingClientRect();
    const px = evt.clientX - rect.left;
    const py = evt.clientY - rect.top;
    const col = -0.5 + (px / rect.width) * imageWidth;
    const row = -0.5 + (py / rect.height) * imageHeight;
    selectVoxelFromClick(row, col);
  });

  container.addEventListener("keydown", (evt) => {
    if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(evt.key)) {
      return;
    }
    evt.preventDefault();
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
  });

  sliceTitle.textContent = sliceTitles[state.sliceIdx];
  redrawVoxel();
}

export default renderWidget;
