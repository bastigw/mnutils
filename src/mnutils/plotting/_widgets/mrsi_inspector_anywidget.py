"""An `anywidget` backend for the MRSI inspector, kept for comparison.

This exists to be measured against `MRSIVoxelInspectorWidget`, not to replace
it. It renders the *same* frontend component -- `mrsi_inspector.js` is shared
verbatim, with only a small adapter mapping traitlets onto the plain payload
object -- so any difference observed between the two backends is a difference
in transport, not in the widget.

The transport is the whole point. Traits declared as `Bytes` are shipped as
binary comm buffers, so the frames and the spectra buffer skip base64 and its
33% inflation entirely. What that costs is a live kernel: the Jupyter widget
protocol has no meaning in a statically built page, which is the tradeoff
`docs/diary/2026-08-14-anywidget-slice-viewer.md` records.
"""

from pathlib import Path

import anywidget
import traitlets

from ._shared import load_css, load_esm

_DIR = Path(__file__).parent


class MRSIVoxelInspectorAnyWidget(anywidget.AnyWidget):
    """Interactive T1 + MRSI overlay served over the Jupyter widget protocol."""

    _esm = load_esm(_DIR / "mrsi_inspector.js", adapter=_DIR / "mrsi_inspector_anywidget.js")
    _css = load_css(_DIR / "mrsi_inspector.css")

    # Binary payloads -- the reason this backend exists. `List(Bytes)` and
    # `Bytes` are serialised as comm buffers rather than JSON strings.
    left_frames = traitlets.List(traitlets.Bytes()).tag(sync=True)
    spectra_bytes = traitlets.Bytes().tag(sync=True)

    slice_titles = traitlets.List(traitlets.Unicode()).tag(sync=True)
    n_anat_slices = traitlets.Int().tag(sync=True)
    initial_slice = traitlets.Int().tag(sync=True)
    image_width = traitlets.Int().tag(sync=True)
    image_height = traitlets.Int().tag(sync=True)
    mrsi_to_display_affine = traitlets.List(traitlets.Float()).tag(sync=True)
    display_to_mrsi_affine = traitlets.List(traitlets.Float()).tag(sync=True)
    grid_shape = traitlets.List(traitlets.Int()).tag(sync=True)
    mrsi_dims = traitlets.List(traitlets.Int()).tag(sync=True)
    initial_voxel = traitlets.List(traitlets.Int()).tag(sync=True)
    ppm = traitlets.List(traitlets.Float()).tag(sync=True)
    spectra_scale = traitlets.Float(1.0).tag(sync=True)
    npts = traitlets.Int().tag(sync=True)
    spectrum_label = traitlets.Unicode("Spectrum").tag(sync=True)

    # Read by the shared frontend to pick a decode path. "binary" is what makes
    # it call `toBytes` on ArrayBuffers instead of base64 strings.
    transport = traitlets.Unicode("binary").tag(sync=True)
    frame_mime = traitlets.Unicode("image/webp").tag(sync=True)
    spectra_encoding = traitlets.Unicode("zlib-float16").tag(sync=True)

    def __init__(self, *, grid_shape, mrsi_dims, initial_voxel, **kwargs) -> None:
        # `inspect_MRSI_spectra` builds one payload dict for both backends and
        # passes tuples where traitlets wants lists.
        super().__init__(
            grid_shape=list(grid_shape),
            mrsi_dims=list(mrsi_dims),
            initial_voxel=list(initial_voxel),
            **kwargs,
        )
