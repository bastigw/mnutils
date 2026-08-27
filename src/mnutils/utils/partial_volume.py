"""Fractional occupancy of a high-resolution mask on a coarse (MRSI) voxel grid.

Answers one question: *what fraction of this spectroscopic voxel is covered by that
segmentation?* -- when the two images differ in field of view, position, voxel size and
obliquity, and are related only by their affines.

The mask is integrated over each target voxel by **forward binning**: every mask voxel is
pushed through ``T = inv(target_affine) @ mask_affine`` and accumulated into the target voxel
it lands in. Because the mask grid is typically far finer than the target grid (a 0.5-1 mm
segmentation against 10-15 mm MRSI voxels), the mask grid *is* the sample grid -- forward
binning reaches the dense-sampling limit at a cost that scales with the mask, not with a
sampling parameter, and only ever moves mass between bins.

Occupancy here is **nominal box overlap**: the target voxel is the parallelepiped its affine
defines. The MRSI spatial response function is deliberately not modelled -- see
``docs/diary/2026-08-24-mask-partial-volume.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from nibabel import spatialimages

from .. import GESeries

__all__ = ["mask_occupancy"]

#: Mask voxels are processed in slabs along the slowest axis so the intermediate coordinate
#: array stays bounded regardless of mask size. Chosen for ~100 MB of float64 coordinates.
_COORD_BUDGET_BYTES = 100 * 1024**2

type ImageLike = npt.NDArray[Any] | spatialimages.SpatialImage | GESeries.NiiBase


def _unpack(
    image: ImageLike,
    affine: npt.NDArray[np.float64] | None,
    shape: tuple[int, int, int] | None,
    name: str,
    *,
    need_data: bool,
) -> tuple[npt.NDArray[Any] | None, npt.NDArray[np.float64], tuple[int, int, int]]:
    """Return ``(data_or_None, affine, shape)`` for an array/NIfTI/NiiBase argument.

    `need_data` keeps the target grid's array off the heap: only its affine and shape are
    ever used, and forcing `get_fdata()` on it would load a volume nothing reads.
    """
    if isinstance(image, GESeries.NiiBase | spatialimages.SpatialImage):
        nii = image.nii if isinstance(image, GESeries.NiiBase) else image
        data = nii.get_fdata() if need_data else None
        img_affine, img_shape = nii.affine, nii.shape
    else:
        data = np.asarray(image)
        img_affine, img_shape = None, data.shape

    affine = img_affine if affine is None else affine
    shape = img_shape if shape is None else shape

    if affine is None:
        raise ValueError(
            f"No {name} affine available: pass `{name}` as a NIfTI image or GESeries object, "
            f"or supply `{name}_affine` explicitly."
        )
    if shape is None or len(shape) != 3:
        raise ValueError(f"The {name} grid must be 3-D; got shape {shape!r}.")

    return data, np.asarray(affine, dtype=np.float64), tuple(int(n) for n in shape[:3])


def _resolve_labels(
    mask: npt.NDArray[Any],
    labels: Sequence[int] | Mapping[int, str] | None,
) -> tuple[list[int] | None, list[str]]:
    """Decide binary/probabilistic vs multi-label, returning ``(label_values, label_names)``.

    ``label_values`` is None for the single-map case, where mask values are accumulated
    directly -- which covers both a binary mask and a probability map (an FSL FAST pve
    volume, say), since a binary mask is just a probability map of zeros and ones.

    Detection is by *value*, not dtype: ``nibabel``'s ``get_fdata()`` returns float64 no
    matter what the file stores, so a label volume read from disk arrives here as floats and
    a dtype test would silently treat every segmentation as a probability map.
    """
    if labels is not None:
        if isinstance(labels, Mapping):
            return [int(v) for v in labels], [str(n) for n in labels.values()]
        return [int(v) for v in labels], [str(int(v)) for v in labels]

    present = np.unique(mask)
    present = present[present != 0]

    # One non-zero value is a binary mask; fractional values are a probability map. Only
    # several distinct whole numbers mean discrete labels.
    if len(present) <= 1 or not np.all(present == np.round(present)):
        return None, ["mask"]

    logger.debug(f"Detected multi-label mask with labels {present.tolist()}")
    return [int(v) for v in present], [str(int(v)) for v in present]


def _world_coords(
    affine: npt.NDArray[np.float64], shape: tuple[int, int, int]
) -> dict[str, tuple[tuple[str, str, str], npt.NDArray[np.float64]]]:
    """World-mm centre of every target voxel, as three ``(i, j, k)``-shaped coordinates.

    These are 3-D, not one array per axis: for an oblique affine world *x* depends on all
    three indices, so per-axis 1-D coordinates would be quietly wrong for exactly the
    acquisitions this module exists to handle.
    """
    idx = np.indices(shape, dtype=np.float64)
    world = np.einsum("ab,bijk->aijk", affine[:3, :3], idx) + affine[:3, 3, None, None, None]
    return {name: (("i", "j", "k"), world[a]) for a, name in enumerate("xyz")}


def mask_occupancy(
    mask: ImageLike,
    target: ImageLike,
    *,
    mask_affine: npt.NDArray[np.float64] | None = None,
    target_affine: npt.NDArray[np.float64] | None = None,
    target_shape: tuple[int, int, int] | None = None,
    labels: Sequence[int] | Mapping[int, str] | None = None,
    min_coverage: float = 0.9,
    _force_general_path: bool = False,
) -> xr.Dataset:
    """Compute what fraction of each target voxel is covered by `mask`.

    Parameters
    ----------
    mask : ndarray, spatialimages.SpatialImage, or GESeries.NiiBase
        The high-resolution mask. A boolean or float array is treated as a binary mask or
        probability map and produces one occupancy map; an integer array holding more than
        one non-zero value is treated as a label volume and produces one map per label.
        If a plain array, `mask_affine` must be given.
    target : ndarray, spatialimages.SpatialImage, or GESeries.NiiBase
        The coarse grid to map onto, typically ``MRSISeries.RAW_exp``. Only its affine and
        shape are used; its data is never read.
    mask_affine : ndarray, optional
        The mask's affine. Required if `mask` is a plain array.
    target_affine : ndarray, optional
        The target affine. Overrides the one carried by `target`.
    target_shape : tuple[int, int, int], optional
        The target grid shape. Overrides the one carried by `target`.
    labels : sequence[int] or mapping[int, str], optional
        Explicit labels to compute, overriding auto-detection. A mapping additionally names
        them: ``{1: "gm", 2: "wm"}`` yields a `label` coordinate of ``["gm", "wm"]``.
    min_coverage : float, optional
        Occupancy is set to NaN in target voxels whose `coverage` falls below this fraction,
        since an occupancy computed over a sliver of a voxel is not comparable to one
        computed over a whole voxel. Defaults to 0.9; pass 0.0 to disable the masking.
    _force_general_path : bool, optional
        Internal test hook: skip the axis-aligned fast path even when `mask` -> `target` is
        diagonal, so both code paths can be checked against each other. Not part of the
        public API.

    Returns
    -------
    xarray.Dataset
        ``occupancy`` with dims ``(label, i, j, k)`` and ``coverage`` with dims
        ``(i, j, k)``, both in [0, 1]. Voxel-centre world coordinates are attached as
        3-D non-dimension coordinates ``x``, ``y``, ``z``; the affines and settings are in
        ``.attrs``.

    Notes
    -----
    `occupancy` is masked by `min_coverage` but `coverage` never is, so a voxel that reads
    NaN can always be explained by looking at the same voxel in `coverage`. Without both,
    "no mask here" and "no anatomy here" are indistinguishable.

    Examples
    --------
    >>> pv = mask_occupancy(seg_nii, mrsi.RAW_exp)  # doctest: +SKIP
    >>> pv.occupancy.sel(label="mask")  # doctest: +SKIP
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage must lie in [0, 1]; got {min_coverage}.")

    mask_data, mask_affine, mask_shape = _unpack(mask, mask_affine, None, "mask", need_data=True)
    _, target_affine, target_shape = _unpack(
        target, target_affine, target_shape, "target", need_data=False
    )
    assert mask_data is not None  # need_data=True always populates it

    label_values, label_names = _resolve_labels(mask_data, labels)

    # Mask index -> target index. One matrix absorbs the differing FOV, voxel size, position
    # and obliquity; nothing downstream needs to know how the two grids differ.
    tgt_from_mask = np.linalg.inv(target_affine) @ mask_affine
    # |det| is the mask-voxel volume expressed in target-voxel volumes, so its reciprocal is
    # how many mask voxels a fully covered target voxel holds -- the denominator for every
    # fraction below. It stays correct for anisotropic and oblique voxels, where a per-axis
    # ratio would not.
    mask_voxels_per_target_voxel = 1.0 / abs(np.linalg.det(tgt_from_mask))

    logger.debug(f"Mask {mask_shape} -> target {target_shape}")
    logger.debug(f"Mask-to-target affine:\n{tgt_from_mask}")
    logger.debug(f"Mask voxels per target voxel: {mask_voxels_per_target_voxel:.2f}")

    # The overlap split assumes a mask voxel reaches at most two target bins per axis.
    half = target_halfwidths(tgt_from_mask)
    if np.any(half >= 0.5):
        coarse = [f"{ax}: {2 * h:.2f}" for ax, h in zip("xyz", half, strict=True) if h >= 0.5]
        raise ValueError(
            "mask_occupancy expects a mask finer than the target grid, but one mask voxel "
            f"spans more than a whole target voxel along {', '.join(coarse)} (target voxels "
            "per mask voxel). Resample the mask onto a finer grid first, or swap the "
            "arguments if the two images were passed the wrong way round."
        )

    linear = tgt_from_mask[:3, :3]
    off_diagonal = linear - np.diag(np.diag(linear))
    axis_aligned = not _force_general_path and np.allclose(
        off_diagonal, 0.0, atol=1e-9 * max(np.abs(linear).max(), 1.0)
    )

    if axis_aligned:
        counts, sums = _accumulate_separable(mask_data, tgt_from_mask, target_shape, label_values)
    else:
        n_target = int(np.prod(target_shape))
        counts = np.zeros(n_target, dtype=np.float64)
        sums = np.zeros((len(label_names), n_target), dtype=np.float64)

        for sl in _slabs(mask_shape):
            flat, weights, values = _splat_slab(
                mask_data[sl], sl, tgt_from_mask, mask_shape, target_shape
            )
            if flat.size == 0:
                continue
            counts += np.bincount(flat, weights=weights, minlength=n_target)
            if label_values is None:
                sums[0] += np.bincount(flat, weights=weights * values, minlength=n_target)
            else:
                for row, value in enumerate(label_values):
                    sums[row] += np.bincount(
                        flat, weights=weights * (values == value), minlength=n_target
                    )
        counts = counts.reshape(target_shape)
        sums = sums.reshape((len(label_names), *target_shape))

    coverage = counts / mask_voxels_per_target_voxel
    occupancy = sums / mask_voxels_per_target_voxel

    # Binning is a discrete approximation of a volume ratio, so a fully covered voxel can
    # land a hair above 1.0. Clip rather than let a fraction read 1.0002.
    coverage = np.clip(coverage, 0.0, 1.0)
    occupancy = np.clip(occupancy, 0.0, 1.0)

    if min_coverage > 0.0:
        occupancy = np.where(coverage < min_coverage, np.nan, occupancy)

    return xr.Dataset(
        data_vars={
            "occupancy": (("label", "i", "j", "k"), occupancy.astype(np.float32)),
            "coverage": (("i", "j", "k"), coverage.astype(np.float32)),
        },
        coords={
            "label": label_names,
            "i": np.arange(target_shape[0]),
            "j": np.arange(target_shape[1]),
            "k": np.arange(target_shape[2]),
            **_world_coords(target_affine, target_shape),
        },
        attrs={
            # Flattened row-major, not nested: netCDF attributes must be 1-D, and a nested
            # 4x4 makes the whole Dataset unserialisable. Restore with .reshape(4, 4).
            "mask_affine": mask_affine.ravel().tolist(),
            "target_affine": target_affine.ravel().tolist(),
            "mask_shape": list(mask_shape),
            "min_coverage": float(min_coverage),
            "mask_voxels_per_target_voxel": float(mask_voxels_per_target_voxel),
            "method": "forward_binning_overlap_weighted",
        },
    )


def _slabs(mask_shape: tuple[int, int, int]) -> list[slice]:
    """Slice the mask's slowest axis into chunks whose coordinate arrays fit the budget."""
    per_slice = int(np.prod(mask_shape[1:])) * 3 * np.dtype(np.float64).itemsize
    step = max(1, min(mask_shape[0], _COORD_BUDGET_BYTES // max(per_slice, 1)))
    return [
        slice(start, min(start + step, mask_shape[0])) for start in range(0, mask_shape[0], step)
    ]


def target_halfwidths(tgt_from_mask: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Half-extent of one mask voxel along each target axis, in target-index units.

    A mask voxel spans +/-0.5 in each of its own index directions, so its image under the
    linear part of `tgt_from_mask` is a parallelepiped whose half-extent along target axis
    `a` is ``0.5 * sum_b |T[a, b]|``. For axis-aligned grids that reduces to half the voxel
    size ratio; the sum is what keeps it right when the grids are oblique.
    """
    return 0.5 * np.abs(tgt_from_mask[:3, :3]).sum(axis=1)


def _axis_weight_matrix(
    diag: float, offset: float, half: float, mask_len: int, target_len: int
) -> npt.NDArray[np.float64]:
    """Overlap weight of every mask index along one axis against every target bin it touches.

    Valid only when `mask` -> `target` is diagonal, so target axis `a`'s coordinate depends on
    mask index `a` alone: ``coord = diag * idx + offset``. Splitting each mask voxel's box
    (`half`-wide) between at most two target bins per axis is then the same overlap formula
    `_splat_slab` uses, just computed once per axis instead of once per mask voxel triple.
    """
    idx = np.arange(mask_len, dtype=np.float64)
    coord = diag * idx + offset
    lower, upper = coord - half, coord + half
    first = np.floor(lower + 0.5)

    weights = np.zeros((mask_len, target_len), dtype=np.float64)
    for step in (0.0, 1.0):
        n = first + step
        overlap = np.minimum(upper, n + 0.5) - np.maximum(lower, n - 0.5)
        w = np.clip(overlap, 0.0, None) / (2.0 * half)
        valid = (w > 0.0) & (n >= 0) & (n < target_len)
        rows = idx[valid].astype(np.intp)
        cols = n[valid].astype(np.intp)
        np.add.at(weights, (rows, cols), w[valid])
    return weights


def _accumulate_separable(
    mask_data: npt.NDArray[Any],
    tgt_from_mask: npt.NDArray[np.float64],
    target_shape: tuple[int, int, int],
    label_values: list[int] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fast path for an axis-aligned `mask` -> `target` transform.

    An outer/tensor product of three small per-axis weight matrices, instead of `_splat_slab`'s
    general 8-way splat. A separable weight product over a diagonal transform *is* an outer
    product -- not an approximation of `_splat_slab`, an exact reformulation of it (see the diary
    entry for the measured agreement). Returns `(counts, sums)` already reshaped to
    `target_shape` / `(n_labels, *target_shape)`, matching what the general path's `np.bincount`
    accumulation produces after its own reshape.
    """
    mask_shape = mask_data.shape
    half = target_halfwidths(tgt_from_mask)
    weights = [
        _axis_weight_matrix(
            tgt_from_mask[a, a], tgt_from_mask[a, 3], half[a], mask_shape[a], target_shape[a]
        )
        for a in range(3)
    ]

    counts = np.einsum("i,j,k->ijk", *(w.sum(axis=0) for w in weights))

    def contract(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.tensordot(values, weights[0], axes=([0], [0]))  # (mj, mk, ti)
        out = np.tensordot(out, weights[1], axes=([0], [0]))  # (mk, ti, tj)
        out = np.tensordot(out, weights[2], axes=([0], [0]))  # (ti, tj, tk)
        return out

    if label_values is None:
        sums = contract(np.asarray(mask_data, dtype=np.float64))[None, ...]
    else:
        sums = np.stack(
            [contract((mask_data == value).astype(np.float64)) for value in label_values]
        )

    return counts, sums


def _splat_slab(
    values: npt.NDArray[Any],
    sl: slice,
    tgt_from_mask: npt.NDArray[np.float64],
    mask_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split one slab of mask voxels across the target voxels they overlap.

    Each mask voxel is treated as a box of half-width `target_halfwidths` around its mapped
    centre and divided between target voxels in proportion to the overlap, separably per
    axis. Assigning it whole to its nearest bin instead makes the count in a target voxel
    quantise to whole mask voxels per axis: at a ratio of 11.75 mask voxels per target voxel
    a *fully covered* voxel's coverage lands anywhere in 0.886-1.054, which no threshold on
    `min_coverage` can survive. Splitting removes that -- the weights per mask voxel sum to
    exactly 1, so coverage varies smoothly instead of stepping.

    Returns flat target indices, the weight of each contribution, and the mask value that
    produced it, so the caller can accumulate values or per-label indicators from the same
    triple.
    """
    idx = np.indices((sl.stop - sl.start, *mask_shape[1:]), dtype=np.float64)
    idx[0] += sl.start

    # A NIfTI index names a voxel *centre*, so target voxel `n` spans [n - 0.5, n + 0.5) and
    # the bin of a mapped coordinate is floor(coord + 0.5). Using floor(coord) instead
    # displaces every result by half a target voxel -- the defect this module replaced.
    coords = (
        np.einsum("ab,bijk->aijk", tgt_from_mask[:3, :3], idx)
        + tgt_from_mask[:3, 3, None, None, None]
    ).reshape(3, -1)
    flat_values = np.asarray(values, dtype=np.float64).reshape(-1)

    half = target_halfwidths(tgt_from_mask)
    lower, upper = coords - half[:, None], coords + half[:, None]
    first = np.floor(lower + 0.5)  # first target bin each mask voxel touches, per axis

    # With the mask finer than the target a voxel spans at most two bins per axis, so the
    # eight (first, first + 1) combinations below cover every contribution. A coarser mask
    # would need more, and is out of scope for this function.
    per_axis: list[list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]] = []
    for axis in range(3):
        options = []
        for step in (0.0, 1.0):
            n = first[axis] + step
            overlap = np.minimum(upper[axis], n + 0.5) - np.maximum(lower[axis], n - 0.5)
            options.append((n, np.clip(overlap, 0.0, None) / (2.0 * half[axis])))
        per_axis.append(options)

    idx_out: list[npt.NDArray[np.intp]] = []
    w_out: list[npt.NDArray[np.float64]] = []
    v_out: list[npt.NDArray[np.float64]] = []
    for ni, wi in per_axis[0]:
        for nj, wj in per_axis[1]:
            for nk, wk in per_axis[2]:
                weight = wi * wj * wk
                bins = np.stack((ni, nj, nk)).astype(np.intp)
                keep = weight > 0.0
                for axis in range(3):
                    keep &= (bins[axis] >= 0) & (bins[axis] < target_shape[axis])
                if not keep.any():
                    continue
                kept = bins[:, keep]
                idx_out.append(
                    np.ravel_multi_index((kept[0], kept[1], kept[2]), target_shape).astype(np.intp)
                )
                w_out.append(weight[keep])
                v_out.append(flat_values[keep])

    if not idx_out:
        empty_i: npt.NDArray[np.intp] = np.empty(0, dtype=np.intp)
        empty_f: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        return empty_i, empty_f, empty_f

    return np.concatenate(idx_out), np.concatenate(w_out), np.concatenate(v_out)
