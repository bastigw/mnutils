"""Synthetic GE exam fixtures for docs pages and tests -- no real scanner data required.

See docs/diary/2026-08-19-synthetic-exam-fixtures.md for why this exists, what it does and doesn't
fake, and where the spectra/anatomy come from. `build_fake_exam()` writes a fresh tree under a
process-local temp directory (via `tempfile.mkdtemp`) the first time it's called, caches the path
for the rest of the process, and registers an `atexit` cleanup so nothing survives past the process
that created it.

`_TEMPLATE_ATTRIBUTION` in `_spectra.py` covers the one piece of real (downloaded, cached) data
used here -- a CC BY-NC 4.0 T1 template.
"""

import atexit
import functools
import shutil
import tempfile
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io
from numpy.typing import ArrayLike

from . import _spectra

_RNG_SEED = 20260819


def _mat_header(
    *,
    nucleus: float,
    protocol: str,
    description: str,
    exam_number: int,
    bandwidth: float,
    centre_freq_x1e7: float,
    fov_mm: float,
) -> dict:
    """Build the nested dict `RawMRISeries._get_header_value` walks (`h.rdb_hdr.*` etc.)."""
    return {
        "rdb_hdr": {
            "data_collect_type": 2.0,
            "run_int": 1.0,
            "nframes": 8.0,
            "user0": bandwidth,
            "ps_mps_freq": centre_freq_x1e7,
            "te": 30000.0,
            "fov": fov_mm,
            "user25": 0.0,
            "user14": 0.0,
            "user3": 0.0,
            "ps_mps_r1": 100.0,
            "ps_mps_r2": 100.0,
            "ps_mps_tg": 100.0,
            "recon": 1.0,
            "scan_date": "01/01/25",
            "scan_time": "12:00",
        },
        "series": {"prtcl": protocol, "se_desc": description},
        "exam": {"ex_no": exam_number},
        "image": {
            "specnuc": nucleus,
            "mr_flip": 90.0,
            "nex": 1.0,
            "tr": 2_000_000.0,
            "slthick": 5.0,
            "sctime": 60_000_000.0,
        },
        "mrconfig": {"fieldStrength": 30_000.0},
    }


# Deuterium at ~3T. Kept in one place so the .mat header, the ppm axis GESeries derives from it,
# and the peaks xmris simulates all agree on what "4.7 ppm" means.
_NUCLEUS = 2.0
_BANDWIDTH_HZ = 5000.0
_CENTRE_FREQ_X1E7 = 1.9612e8  # -> 80.0 MHz via the `centre_frequency` property
_CENTRE_FREQ_MHZ = _CENTRE_FREQ_X1E7 / 1e7
_CARRIER_PPM = 4.68  # matches GESeries.RawMRISeries.carrier_ppm for nucleus in {1, 2}


def _rng(*salt: object) -> np.random.Generator:
    return np.random.default_rng(abs(hash((_RNG_SEED, *salt))) % (2**32))


def _write_own_nii(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)


def _write_mat(path: Path, *, spec: np.ndarray, fov_mm: float, seed: int, **header_kwargs) -> None:
    recon = {
        "h": _mat_header(fov_mm=fov_mm, **header_kwargs),
        "bb": _rng(seed, "bb").standard_normal((4, 4)),
        "spec": spec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, recon)


def _write_mrs_mat(
    path: Path,
    *,
    series_id: int,
    averages: int,
    npts: int,
    seed: int,
    intensities: ArrayLike | None = None,
    fov_mm: float = 240.0,
    **header_kwargs,
) -> None:
    """Write the `.mat` spectrum and, from the same simulated peaks, the raw FID `.h5` cache.

    Every real MRS acquisition has both, so every synthetic one does too -- `simulate_fid` and
    `simulate_spectrum` share the same `(npts, seed, intensities, ...)` inputs, so the raw FID is
    the exact time-domain data `simulate_spectrum`'s FFT is computed from, not an independently
    random array.
    """
    spec = _spectra.simulate_spectrum(
        npts=npts,
        n_specs=averages,
        bandwidth=_BANDWIDTH_HZ,
        seed=seed,
        intensities=intensities,
    )
    fid = _spectra.simulate_fid(
        npts=npts,
        n_specs=averages,
        bandwidth=_BANDWIDTH_HZ,
        seed=seed,
        intensities=intensities,
    )
    _write_mat(
        path,
        spec=spec,
        fov_mm=fov_mm,
        seed=seed,
        nucleus=_NUCLEUS,
        bandwidth=_BANDWIDTH_HZ,
        centre_freq_x1e7=_CENTRE_FREQ_X1E7,
        **header_kwargs,
    )
    _write_raw_fids_h5(path.parent, series_id, fid)


def _write_mrsi_mat(
    path: Path,
    *,
    npts: int,
    grid: tuple[int, int, int],
    fov_mm: float,
    seed: int,
    intensity_map: np.ndarray | None = None,
    **header_kwargs,
) -> None:
    if intensity_map is None:
        intensity_map = np.full(grid, 0.5, dtype=np.float32)
    spec = _spectra.simulate_grid(
        npts=npts,
        grid=grid,
        bandwidth=_BANDWIDTH_HZ,
        centre_freq_mhz=_CENTRE_FREQ_MHZ,
        carrier_ppm=_CARRIER_PPM,
        intensity_map=intensity_map,
        seed=seed,
    )
    _write_mat(
        path,
        spec=spec,
        fov_mm=fov_mm,
        seed=seed,
        nucleus=_NUCLEUS,
        bandwidth=_BANDWIDTH_HZ,
        centre_freq_x1e7=_CENTRE_FREQ_X1E7,
        **header_kwargs,
    )


def _write_raw_fids_h5(series_folder: Path, series_id: int, fids: np.ndarray) -> None:
    """A dummy `ScanArchive*.h5` (only its existence matters) plus a pre-cached `fids` dataset.

    `load_raw_fids` reads the cache directly when present and never opens the archive -- see the
    diary entry.
    """
    series_folder.mkdir(parents=True, exist_ok=True)
    (series_folder / f"ScanArchive_{series_id}.h5").touch()
    with h5py.File(series_folder / f"Series{series_id}_raw_fids.h5", "w") as hf:
        hf.create_dataset("fids", data=fids)


def _write_raw_fid_files(series_folder: Path, series_id: int, npts: int, ntime: int = 150) -> None:
    """Random-noise raw FIDs for series with no `.mat`-backed spectrum to derive real ones from."""
    fids = _rng(series_folder, "fids").standard_normal((npts, ntime)) + 1j * _rng(
        series_folder, "fids-imag"
    ).standard_normal((npts, ntime))
    _write_raw_fids_h5(series_folder, series_id, fids)


def _build_hevo18(root: Path) -> None:
    data = root / "data"
    exam = data / "ExamHeVo18anon"
    grid = (16, 16, 16)

    # The real T1 template, cropped to the head -- see _spectra._TEMPLATE_ATTRIBUTION.
    _spectra.write_template_t1(data / "002_3D_Ax_T1_BRAVO" / "2_3D_Ax_T1_BRAVO.nii.gz")

    # Series 8's own NIfTI seeds create_MRSI_nii/create_MRSI_affine *and* doubles as the
    # per-grid-cell intensity map below -- real brain signal in, brain-shaped spectra out.
    own_affine, intensity_map = _spectra.template_grid_intensity(grid)
    _write_own_nii(
        data / "008_MRSI_pseudo_S64_X10_Y10_Z10_T1_C1" / "8_MRSI_pseudo.nii.gz",
        intensity_map * 200,
        own_affine,
    )
    (data / "014_localizer").mkdir(parents=True, exist_ok=True)  # DICOM-only gap filler

    _write_mrs_mat(
        exam / "Series6" / "ScanArchive_Series6.mat",
        series_id=6,
        averages=64,
        npts=2048,
        seed=6,
        protocol="MRS_unloc",
        description="006_MRS_unloc",
        exam_number=1801,
    )

    # Create increasing intensities for the washin series
    # Defines start values [HDO, Glucose, Glx, Baseline] and their stop values
    intensities = np.linspace(start=[1.0, 1.0, 1.0, 1.0], stop=[1.2, 2.0, 1.8, 1.0], num=250)

    _write_mrs_mat(
        exam / "Series7" / "ScanArchive_Series7.mat",
        series_id=7,
        averages=250,
        intensities=intensities,
        npts=2048,
        seed=7,
        protocol="MRS_washin",
        description="007_MRS_washin",
        exam_number=1801,
    )

    _write_mrsi_mat(
        exam / "Series8" / "ScanArchive_Series8.mat",
        npts=700,
        grid=grid,
        fov_mm=300.0,
        seed=8000,
        intensity_map=intensity_map,
        protocol="MRSI_pseudo",
        description="008_MRSI_pseudo_S64_X10_Y10_Z10_T1_C1",
        exam_number=1801,
    )

    for series_id, npts in ((9, 700), (11, 700), (12, 1678)):
        _write_raw_fid_files(exam / f"Series{series_id}", series_id, npts=npts)

    (exam / "Series13").mkdir(parents=True, exist_ok=True)  # exam data, no DICOM


def _build_hevo23(root: Path) -> None:
    data = root / "data"
    (data / "Series0010_BS_prescan").mkdir(parents=True, exist_ok=True)  # new-style naming
    _spectra.write_template_t1(data / "002_3D_Ax_T1_BRAVO" / "2_3D_Ax_T1_BRAVO_BODY.nii.gz")


def _build_lg_d19(root: Path) -> None:
    exam = root / "Exam4873anon"
    _write_mrsi_mat(
        exam / "Series7" / "ScanArchive_Series7.mat",
        npts=700,
        grid=(16, 16, 16),
        fov_mm=300.0,
        seed=7700,
        protocol="MRSI_pseudo",
        description="007_MRSI_pseudo",
        exam_number=4873,
    )
    (exam / "Series5").mkdir(parents=True, exist_ok=True)  # no .mat -- FileNotFoundError case
    (root / "010_axial_localizer").mkdir(parents=True, exist_ok=True)  # old-style naming


def _build_nist(root: Path) -> None:
    data = root / "data"
    exam = data / "ExamNISTanon"
    grid = (16, 16, 16)
    fov_mm = 300.0  # -> 14 mm in-plane voxels, comfortably resolves the 8-sphere ring
    z_extent_mm = 300.0
    rings = _spectra.DEFAULT_SPHERE_RINGS

    _spectra.sphere_phantom_image(
        data / "002_3D_Ax_T1_BRAVO" / "2_3D_Ax_T1_BRAVO.nii.gz",
        shape=(128, 128, 100),
        voxel_mm=1,
        rings=rings,
        seed=2,
    )

    intensity_map, voxel_xy = _spectra.sphere_grid_intensity(
        grid, fov_mm=fov_mm, z_extent_mm=z_extent_mm, rings=rings
    )
    own_affine = _spectra.grid_own_affine(grid, (voxel_xy, voxel_xy, z_extent_mm / grid[2]))
    _write_own_nii(
        data / "005_MRSI_pseudo_S64_X12_Y12_Z3_T1_C1" / "5_MRSI_pseudo.nii.gz",
        intensity_map * 200,
        own_affine,
    )

    _write_mrsi_mat(
        exam / "Series5" / "ScanArchive_Series5.mat",
        npts=700,
        grid=grid,
        fov_mm=fov_mm,
        seed=5500,
        intensity_map=intensity_map * 500,
        protocol="MRSI_pseudo",
        description="005_MRSI_pseudo_S64_X12_Y12_Z3_T1_C1",
        exam_number=408,
    )


def _write_legacy_mat(path: Path) -> None:
    """A genuine MATLAB v7.3 (HDF5) file -- `mat73`'s fallback path, not `scipy.io.loadmat`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as hf:
        dataset = hf.create_dataset("pyr", data=_rng(path, "pyr").standard_normal(64))
        dataset.attrs["MATLAB_class"] = np.bytes_(b"double")


@functools.cache
def build_fake_exam() -> Path:
    """Return the root of a synthetic multi-exam tree, generating it on first call.

    Contains `HeVo-18/`, `HeVo-23/`, `LG_D19/`, `20250408-NIST-Mag2/`, and `MRSexampleTE35.mat` --
    the same names and shapes docs pages and tests previously read from `tests/datasets/`, entirely
    fabricated (see the diary entry for what's simulated vs. downloaded). The tree lives under a
    unique `tempfile.mkdtemp` directory removed by an `atexit` hook when the process exits, so
    nothing accumulates across runs.
    """
    root = Path(tempfile.mkdtemp(prefix="mnutils_fake_exam_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    _build_hevo18(root / "HeVo-18")
    _build_hevo23(root / "HeVo-23")
    _build_lg_d19(root / "LG_D19")
    _build_nist(root / "20250408-NIST-Mag2")
    _write_legacy_mat(root / "MRSexampleTE35.mat")

    return root
