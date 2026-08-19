"""Synthetic GE exam fixtures for docs pages and tests -- no real scanner data required.

See docs/diary/2026-08-19-synthetic-exam-fixtures.md for why this exists and what it does and
doesn't fake. `build_fake_exam()` writes a fresh tree under a process-local temp directory (via
`tempfile.mkdtemp`) the first time it's called, caches the path for the rest of the process, and
registers an `atexit` cleanup so nothing survives past the process that created it.
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

_RNG_SEED = 20260819


def _mat_header(
    *,
    nucleus: float,
    protocol: str,
    description: str,
    exam_number: float,
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


def _rng(*salt: object) -> np.random.Generator:
    return np.random.default_rng(abs(hash((_RNG_SEED, *salt))) % (2**32))


def _write_mrs_mat(
    path: Path, *, averages: int, npts: int, fov_mm: float = 240.0, **header_kwargs
) -> None:
    spec = _rng(path, "spec").standard_normal((averages, npts)) + 1j * _rng(
        path, "spec-imag"
    ).standard_normal((averages, npts))
    recon = {
        "h": _mat_header(fov_mm=fov_mm, **header_kwargs),
        "bb": _rng(path, "bb").standard_normal((4, 4)),
        "spec": spec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, recon)


def _write_mrsi_mat(
    path: Path, *, npts: int, grid: tuple[int, int, int], fov_mm: float, **header_kwargs
) -> None:
    i, j, k = grid
    spec = _rng(path, "spec").standard_normal((npts, i, j, k)) + 1j * _rng(
        path, "spec-imag"
    ).standard_normal((npts, i, j, k))
    recon = {
        "h": _mat_header(fov_mm=fov_mm, **header_kwargs),
        "bb": _rng(path, "bb").standard_normal((4, 4)),
        "spec": spec,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(path, recon)


def _write_raw_fid_files(series_folder: Path, series_id: int, npts: int, ntime: int = 150) -> None:
    """A dummy `ScanArchive*.h5` (only its existence matters) plus a pre-cached `fids` dataset.

    `load_raw_fids` reads the cache directly when present and never opens the archive -- see the
    diary entry.
    """
    series_folder.mkdir(parents=True, exist_ok=True)
    (series_folder / f"ScanArchive_{series_id}.h5").touch()
    fids = _rng(series_folder, "fids").standard_normal((npts, ntime)) + 1j * _rng(
        series_folder, "fids-imag"
    ).standard_normal((npts, ntime))
    with h5py.File(series_folder / f"Series{series_id}_raw_fids.h5", "w") as hf:
        hf.create_dataset("fids", data=fids)


def _write_nifti(
    path: Path, *, shape: tuple[int, int, int], voxel_mm: float | tuple[float, float, float]
) -> None:
    if isinstance(voxel_mm, (int, float)):
        voxel_mm = (voxel_mm, voxel_mm, voxel_mm)
    data = (_rng(path, "nii").random(shape, dtype=np.float64) * 100).astype(np.float32)
    affine = np.diag([*voxel_mm, 1.0])
    affine[:3, 3] = -np.array(shape) * np.array(voxel_mm) / 2
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def _build_hevo18(root: Path) -> None:
    data = root / "data"
    exam = data / "ExamHeVo18anon"

    _write_nifti(
        # Large in-plane and enough through-plane range for voxel_overlay.md's crop/slice-index
        # demos to stay in-bounds after orientation/resampling.
        data / "002_3D_Ax_T1_BRAVO" / "2_3D_Ax_T1_BRAVO.nii.gz",
        shape=(448, 448, 64),
        voxel_mm=(0.5, 0.5, 3.0),
    )
    _write_nifti(
        data / "008_MRSI_pseudo_S64_X4_Y4_Z2_T1_C1" / "8_MRSI_pseudo.nii.gz",
        shape=(8, 8, 8),
        voxel_mm=15.0,
    )
    (data / "014_localizer").mkdir(parents=True, exist_ok=True)  # DICOM-only gap filler

    _write_mrs_mat(
        exam / "Series6" / "ScanArchive_Series6.mat",
        averages=8,
        npts=64,
        nucleus=2.0,
        protocol="MRS_unloc",
        description="006_MRS_unloc",
        exam_number=1801.0,
        bandwidth=5000.0,
        centre_freq_x1e7=8.0e8,
    )
    _write_raw_fid_files(exam / "Series6", 6, npts=64)

    _write_mrs_mat(
        exam / "Series7" / "ScanArchive_Series7.mat",
        averages=8,
        npts=64,
        nucleus=2.0,
        protocol="MRS_washin",
        description="007_MRS_washin",
        exam_number=1801.0,
        bandwidth=5000.0,
        centre_freq_x1e7=8.0e8,
    )

    _write_mrsi_mat(
        exam / "Series8" / "ScanArchive_Series8.mat",
        npts=64,
        grid=(10, 10, 10),
        fov_mm=240.0,
        nucleus=2.0,
        protocol="MRSI_pseudo",
        description="008_MRSI_pseudo_S64_X4_Y4_Z2_T1_C1",
        exam_number=1801.0,
        bandwidth=5000.0,
        centre_freq_x1e7=8.0e8,
    )

    for series_id, npts in ((9, 64), (11, 64), (12, 1678)):
        _write_raw_fid_files(exam / f"Series{series_id}", series_id, npts=npts)

    (exam / "Series13").mkdir(parents=True, exist_ok=True)  # exam data, no DICOM


def _build_hevo23(root: Path) -> None:
    data = root / "data"
    (data / "Series0010_BS_prescan").mkdir(parents=True, exist_ok=True)  # new-style naming
    _write_nifti(
        data / "002_3D_Ax_T1_BRAVO" / "2_3D_Ax_T1_BRAVO_BODY.nii.gz",
        shape=(64, 64, 48),
        voxel_mm=2.0,
    )


def _build_lg_d19(root: Path) -> None:
    exam = root / "Exam4873anon"
    _write_mrsi_mat(
        exam / "Series7" / "ScanArchive_Series7.mat",
        npts=700,
        grid=(1, 1, 1),
        fov_mm=240.0,
        nucleus=2.0,
        protocol="MRSI_pseudo",
        description="007_MRSI_pseudo",
        exam_number=4873.0,
        bandwidth=5000.0,
        centre_freq_x1e7=8.0e8,
    )
    (exam / "Series5").mkdir(parents=True, exist_ok=True)  # no .mat -- FileNotFoundError case
    (root / "010_axial_localizer").mkdir(parents=True, exist_ok=True)  # old-style naming


def _build_nist(root: Path) -> None:
    data = root / "data"
    exam = data / "ExamNISTanon"

    _write_nifti(
        data / "002_3D_Ax_T1_BRAVO" / "2_3D_Ax_T1_BRAVO.nii.gz", shape=(32, 32, 24), voxel_mm=2.0
    )
    _write_nifti(
        data / "005_MRSI_pseudo_S64_X4_Y4_Z2_T1_C1" / "5_MRSI_pseudo.nii.gz",
        shape=(4, 4, 2),
        voxel_mm=24.0,
    )
    _write_mrsi_mat(
        exam / "Series5" / "ScanArchive_Series5.mat",
        npts=64,
        grid=(4, 4, 2),
        fov_mm=96.0,
        nucleus=2.0,
        protocol="MRSI_pseudo",
        description="005_MRSI_pseudo_S64_X4_Y4_Z2_T1_C1",
        exam_number=408.0,
        bandwidth=5000.0,
        centre_freq_x1e7=8.0e8,
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
    fabricated. The tree lives under a unique `tempfile.mkdtemp` directory removed by an `atexit`
    hook when the process exits, so nothing accumulates across runs.
    """
    root = Path(tempfile.mkdtemp(prefix="mnutils_fake_exam_"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)

    _build_hevo18(root / "HeVo-18")
    _build_hevo23(root / "HeVo-23")
    _build_lg_d19(root / "LG_D19")
    _build_nist(root / "20250408-NIST-Mag2")
    _write_legacy_mat(root / "MRSexampleTE35.mat")

    return root
