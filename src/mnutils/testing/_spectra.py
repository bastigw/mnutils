"""Realistic-ish spectra and spatial phantoms for the synthetic exam fixtures.

Spectra come from `xmris.simulate_fid` (real AMARES-formulation peak simulation), not noise, with
three peaks always present -- HDO, Glucose, Glx -- at roughly their real chemical shifts. Spatial
realism comes from two sources: a downloaded real T1 template for brain-shaped anatomy (see
`_TEMPLATE_ATTRIBUTION`), and a synthetic ring-of-spheres phantom for the NIST-style dataset, where
each sphere gets a different, increasing signal intensity and that same intensity drives the
spectra simulated at its location.
"""

import functools
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import xarray as xr
import xmris
from loguru import logger
from numpy.typing import ArrayLike

_TEMPLATE_URL = "https://raw.githubusercontent.com/neurolabusc/niivue-images/main/chris_t1.nii.gz"
_TEMPLATE_ATTRIBUTION = (
    "T1 anatomical fixtures are resampled from Chris Rorden's 'chris_t1', part of the "
    "niivue-images sample set (https://github.com/neurolabusc/niivue-images), licensed "
    "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/) -- non-commercial use only."
)

# Deuterium (2H) chemical shifts track their 1H counterparts (chemical shift is a property of the
# electronic environment, not the nucleus). HDO dominates real 2H-MRSI spectra by roughly an order
# of magnitude over the metabolite peaks -- these ratios, and the per-peak linewidths, are
# illustrative, not measured. `Baseline` is a broad, low-amplitude component standing in for the
# macromolecular/lipid background every real spectrum sits on top of.
# name: (relative amplitude, chemical shift [ppm], linewidth [Hz])
DMI_TRUTH: dict[str, tuple[float, float, float]] = {
    "HDO": (1.0, 4.70, 12.0),
    "Glucose": (0.25, 3.80, 12.0),
    "Glx": (0.18, 2.40, 12.0),
    "Baseline": (1.0, 1.9, 70.0),
}
PEAKS_PPM: dict[str, float] = {name: ppm for name, (_, ppm, _) in DMI_TRUTH.items()}
PEAK_REL_AMPLITUDE: dict[str, float] = {name: amp for name, (amp, _, _) in DMI_TRUTH.items()}

FREQ_2H_MHz_3T = 19.61
CARRIER_PPM = 4.68
_BASE_AMPLITUDE = 40.0
_BASE_SNR_UNLOC = 50  # target SNR per simulated spectrum, unlocalized MRS
_BASE_SNR_MRSI = 30  # MRSI voxels are effectively single-transient -- no sqrt(N) boost


def simulate_fid(
    *,
    npts: int,
    bandwidth: float,
    centre_freq_mhz: float = FREQ_2H_MHz_3T,
    carrier_ppm: float = CARRIER_PPM,
    b0_ppm: float = 0.0,
    phase_deg: float = 0.0,
    base_snr: float = _BASE_SNR_UNLOC,
    intensities: ArrayLike | None = None,
    seed: int,
    n_specs: int = 1,
) -> xr.DataArray:
    """`n_specs` simulated FIDs (HDO/Glucose/Glx/Baseline), time-domain, via `xmris.simulate_fid`.

    Returns the raw `xr.DataArray` -- callers convert it to whatever they need: `xmris.to_spectrum`
    for the frequency domain, `.values`/`np.asarray` for a plain numpy array. This lets one
    simulated dataset back both a `.mat` spectrum and a raw-FID fixture (same peaks, same noise)
    instead of two independently-random ones. `intensities` scales each peak's amplitude, per
    spectrum -- either one `(n_peaks,)` vector broadcast across all `n_specs`, or a `(n_specs,
    n_peaks)` array for per-spectrum control (e.g. a washin series ramping Glucose/Glx over time).
    `b0_ppm`/`phase_deg` add a uniform chemical-shift offset / zero-order phase across all peaks,
    matching real per-repetition/per-voxel B0 and phase drift.
    """
    names = list(DMI_TRUTH)
    amps = np.asarray([_BASE_AMPLITUDE * DMI_TRUTH[name][0] for name in names], dtype=np.float32)
    chemical_shifts = np.asarray([DMI_TRUTH[name][1] + b0_ppm for name in names], dtype=np.float32)
    dampings = np.asarray([np.pi * DMI_TRUTH[name][2] for name in names], dtype=np.float32)
    phases = np.deg2rad(phase_deg)

    if intensities is None:
        intensities = np.ones(len(names))
    intensities = np.atleast_1d(intensities).astype(np.float32)

    # Check if the number of intensities matches the number of spectra to simulate
    if intensities.ndim == 1:
        intensities = np.broadcast_to(intensities, (n_specs, len(names)))
    elif intensities.ndim == 2:
        if intensities.shape[0] != n_specs or intensities.shape[1] != len(names):
            raise ValueError(
                f"intensities shape {intensities.shape} does not match n_specs={n_specs} and "
                f"number of peaks={len(names)}"
            )

    fid = xr.concat(
        [
            xmris.simulate_fid(
                amplitudes=amps * intensities[i],
                chemical_shifts=chemical_shifts,
                reference_frequency=centre_freq_mhz,
                carrier_ppm=carrier_ppm,
                spectral_width=bandwidth,
                n_points=npts,
                dampings=dampings,
                phases=phases,
                target_snr=base_snr,
                seed=seed + i,  # a shared seed across i would give every voxel identical noise
            )
            for i in range(n_specs)
        ],
        dim="voxel",
    ).assign_coords(voxel=np.arange(n_specs))
    fid.attrs = {"reference_frequency": centre_freq_mhz, "carrier_ppm": carrier_ppm}
    return fid


def simulate_grid(
    *,
    npts: int,
    grid: tuple[int, int, int],
    bandwidth: float,
    centre_freq_mhz: float = FREQ_2H_MHz_3T,
    carrier_ppm: float = CARRIER_PPM,
    intensity_map: np.ndarray,
    seed: int,
) -> np.ndarray:
    """`(npts, i, j, k)` grid of spectra, one per cell, scaled by `intensity_map` (same shape).

    Every cell gets its own unscaled spectrum from `simulate_fid`, then `intensity_map` scales
    each cell's signal post-FFT. MRSI voxels are effectively single-transient, so `_BASE_SNR_MRSI`
    (no `sqrt(N)` boost) sets the noise floor.
    """
    n_cells = int(np.prod(grid))

    # Simulate all FIDs, then convert to the frequency domain
    fid = simulate_fid(
        npts=npts,
        bandwidth=bandwidth,
        centre_freq_mhz=centre_freq_mhz,
        carrier_ppm=carrier_ppm,
        base_snr=_BASE_SNR_MRSI,
        seed=seed,
        n_specs=n_cells,
    )
    # xmris.to_spectrum returns (voxel, frequency) -- transpose to (frequency, voxel) before
    # reshaping, or the flat buffer scrambles frequency bins across voxels.
    specs_flat = np.asarray(xmris.to_spectrum(fid).values, dtype=complex).T

    # Reshape to (npts, i, j, k) and apply intensity map
    specs = specs_flat.reshape((npts, *grid))
    specs *= intensity_map.reshape((1, *grid))
    return specs


def grid_world_centers(
    grid: tuple[int, int, int], voxel_size: tuple[float, float, float]
) -> np.ndarray:
    """World-mm center of every grid cell, `(i, j, k, 3)`, under a `-shape*voxel/2` origin.

    This is the same convention `grid_own_affine`/`sphere_phantom_image` use for anatomical
    volumes, and matches `MRSISeries.create_MRSI_affine()` in x/y to within half a voxel when the
    MRSI series' own NIfTI is built with matching shape/voxel_size (see `grid_own_affine`) --
    comfortably inside a sphere's radius, so sphere placement lines up between the anatomical image
    and the MRSI grid without needing to replicate that method's exact half-voxel shift.
    """
    idx = np.indices(grid).astype(np.float64)  # (3, i, j, k)
    dims = np.array(grid, dtype=np.float64).reshape(3, 1, 1, 1)
    vox = np.array(voxel_size, dtype=np.float64).reshape(3, 1, 1, 1)
    world = vox * (idx - dims / 2)
    return np.moveaxis(world, 0, -1)


def grid_own_affine(
    grid: tuple[int, int, int], voxel_size: tuple[float, float, float]
) -> np.ndarray:
    """The `-shape*voxel/2`-centered affine for an image shaped exactly like `grid`."""
    affine = np.diag([*voxel_size, 1.0])
    affine[:3, 3] = -np.array(grid) * np.array(voxel_size) / 2
    return affine


# --- Sphere-ring phantom (NIST-style) -----------------------------------------------------------


@dataclass(frozen=True)
class SphereRing:
    """One ring of spheres in the x/y plane, offset along z, each with its own signal intensity.

    Add another metabolite range by appending a second `SphereRing` (different `peaks_ppm`/
    `peak_rel_amplitude`) to the list passed to `sphere_intensity`/`sphere_phantom_image` -- rings
    are independent and just take the max where they'd overlap.
    """

    name: str
    n_spheres: int = 8
    ring_radius_mm: float = 60.0
    z_offset_mm: float = 20.0
    sphere_radius_mm: float = 10.0
    intensity_range: tuple[float, float] = (0.3, 1.0)
    peaks_ppm: dict[str, float] = field(default_factory=lambda: dict(PEAKS_PPM))
    peak_rel_amplitude: dict[str, float] = field(default_factory=lambda: dict(PEAK_REL_AMPLITUDE))

    def spheres(self) -> list[tuple[tuple[float, float, float], float]]:
        """`[(center_xyz_mm, intensity), ...]`, intensity strictly increasing around the ring."""
        intensities = np.linspace(*self.intensity_range, self.n_spheres)
        out = []
        for k in range(self.n_spheres):
            theta = 2 * np.pi * k / self.n_spheres
            center = (
                self.ring_radius_mm * np.cos(theta),
                self.ring_radius_mm * np.sin(theta),
                self.z_offset_mm,
            )
            out.append((center, float(intensities[k])))
        return out


DEFAULT_SPHERE_RINGS: list[SphereRing] = [SphereRing(name="metabolite_ring")]


def sphere_intensity(world_xyz: np.ndarray, rings: list[SphereRing]) -> np.ndarray:
    """Per-position intensity in `[0, 1]`, `world_xyz` shaped `(..., 3)` -- 0 outside spheres."""
    intensity = np.zeros(world_xyz.shape[:-1], dtype=np.float32)
    for ring in rings:
        for center, amp in ring.spheres():
            dist = np.linalg.norm(world_xyz - np.array(center), axis=-1)
            intensity = np.maximum(intensity, np.where(dist <= ring.sphere_radius_mm, amp, 0.0))
    return intensity


def sphere_phantom_image(
    path: Path, *, shape: tuple[int, int, int], voxel_mm: float, rings: list[SphereRing], seed: int
) -> None:
    """Write a fine-resolution anatomical NIfTI showing the sphere ring as bright regions."""
    world = grid_world_centers(shape, (voxel_mm, voxel_mm, voxel_mm))
    intensity = sphere_intensity(world, rings)
    background = np.random.default_rng(seed).random(shape).astype(np.float32) * 5.0
    data = (background + intensity * 200.0).astype(np.float32)
    affine = grid_own_affine(shape, (voxel_mm, voxel_mm, voxel_mm))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def sphere_grid_intensity(
    grid: tuple[int, int, int], *, fov_mm: float, z_extent_mm: float, rings: list[SphereRing]
) -> tuple[np.ndarray, float]:
    """Intensity map over an MRSI grid plus the x/y voxel size (`fov_mm / grid[0]`) it implies."""
    voxel_xy = fov_mm / grid[0]
    voxel_z = z_extent_mm / grid[2]
    world = grid_world_centers(grid, (voxel_xy, voxel_xy, voxel_z))
    return sphere_intensity(world, rings), voxel_xy


# --- Real T1 template (brain-shaped anatomy) ----------------------------------------------------


def _template_cache_path() -> Path:
    return Path(tempfile.gettempdir()) / "mnutils_template_cache" / "chris_t1.nii.gz"


def _fetch_t1_template_path() -> Path:
    """Download the CC BY-NC 4.0 template once per machine, cached under the system temp dir."""
    cache = _template_cache_path()
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading T1 template fixture from {_TEMPLATE_URL} to {cache}")
        with urllib.request.urlopen(_TEMPLATE_URL, timeout=30) as response:  # noqa: S310
            cache.write_bytes(response.read())
    return cache


@functools.cache
def _cropped_template() -> tuple[np.ndarray, np.ndarray]:
    """The template's data cropped to its non-background bounding box, and the matching affine."""
    img = nib.load(_fetch_t1_template_path())
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    mask = data > (data.max() * 0.05)
    idx = np.array(np.where(mask))
    lo, hi = idx.min(axis=1), idx.max(axis=1) + 1
    cropped = data[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    affine = img.affine.copy()
    affine[:3, 3] = affine[:3, :3] @ lo + affine[:3, 3]
    return cropped, affine


def write_template_t1(path: Path) -> None:
    """Write the cropped, full-resolution template as an anatomical NIfTI."""
    data, affine = _cropped_template()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), path)


def template_grid_intensity(grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the template onto `grid`; returns `(own_affine, intensity_map in [0, 1])`.

    `own_affine` is what an MRSI series' own NIfTI (the one `create_MRSI_nii` derives its affine
    from) should carry so the low-res grid sits inside the real brain's footprint; `intensity_map`
    is that same downsampled data, normalized, and is meant to drive `simulate_grid`'s
    `intensity_map` directly -- real brain signal in, brain-shaped spectra out.
    """
    from scipy.ndimage import zoom

    data, affine = _cropped_template()
    voxel_native = np.diag(affine)[:3]
    physical_extent = np.array(data.shape) * voxel_native
    voxel_size = physical_extent / np.array(grid)

    # grid_mode=True samples voxel *centers* under the same uniform physical_extent/grid stride
    # own_affine assumes below -- the default corner-aligned mapping (native index 0 -> low-res
    # index 0, native's *last* index -> low-res's last index) stretches across a shorter span than
    # physical_extent implies, growing to a full blocky-voxel misalignment at the far edge.
    zoom_factors = np.array(grid) / np.array(data.shape)
    low_res = zoom(data, zoom_factors, order=1, grid_mode=True, mode="nearest")
    if low_res.shape != tuple(grid):
        padded = np.zeros(grid, dtype=np.float32)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(low_res.shape, grid, strict=True))
        padded[sl] = low_res[sl]
        low_res = padded

    # `simulate_grid` scales each cell's whole complex spectrum (signal *and* noise) by this map,
    # so a hard 0.0 outside the head would silence the noise floor there too -- a background voxel
    # would come back a perfectly flat, noiseless zero instead of a real acquisition's noise-only
    # spectrum. Floor it so background stays low-signal but still noisy.
    intensity_map = (low_res / (low_res.max() + 1e-6)).astype(np.float32)
    intensity_map = np.maximum(intensity_map, 0.0)
    own_affine = np.diag([*voxel_size, 1.0])
    own_affine[:3, 3] = affine[:3, 3]
    # MRSISeries.create_MRSI_affine() always adds an extra half-*blocky*-voxel shift in x/y on top
    # of its resize shift (tuned for real GE headers, where the "fine" reference affine needs it).
    # For grid_mode=True zoom, blocky voxel i's true center is:
    #   native_translation + blocky_voxel*(i + 0.5) - native_voxel/2
    # i.e. own_affine's translation should be native_translation - native_voxel/2 to land there once
    # create_MRSI_affine's own +blocky_voxel/2 shift is added back on top -- the blocky_voxel terms
    # cancel exactly, so the correction here is half a *native* voxel, not half the blocky one.
    own_affine[:2, 3] -= voxel_native[:2] / 2
    return own_affine, intensity_map


def template_tissue_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A binary head mask and a 3-label tissue segmentation of the cropped T1 template.

    Returns `(brain, seg, affine)`, all on the template's own full-resolution grid. `seg`
    labels are 1/2/3, split by intensity percentile inside `brain`.

    These are **not** anatomically faithful -- intensity banding is not tissue segmentation.
    They exist so partial-volume pages have a mask with realistic *shape*: convoluted borders,
    a solid interior, and an extent that only partly overlaps the MRSI grid. That geometry is
    what those pages test; which voxel is really grey matter is irrelevant to them.
    """
    data, affine = _cropped_template()
    brain = data > data.max() * 0.15

    seg = np.zeros(data.shape, dtype=np.int16)
    lo, hi = np.percentile(data[brain], [33.0, 66.0])
    seg[brain & (data <= lo)] = 1
    seg[brain & (data > lo) & (data <= hi)] = 2
    seg[brain & (data > hi)] = 3

    return brain, seg, affine
