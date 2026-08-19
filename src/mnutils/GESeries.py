from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import cached_property
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyAMARES
import xarray as xr
import xmris  # noqa: F401, necessary for xarray accessor to work
from loguru import logger
from nibabel import nifti1, spatialimages

from . import fitting, plotting
from .utils import data_loaders, file_helpers, nifti, spectra


class NiiBase:
    """Wraps a NIfTI image with orientation-aware loading, display, and I/O helpers."""

    def __init__(
        self,
        nii: spatialimages.SpatialImage | Path,
        orientation: tuple[str, str, str] = ("L", "P", "S"),
    ) -> None:
        if isinstance(nii, Path):
            self.nii_path = nii
            nii = nifti1.load(nii)
        else:
            self.nii_path = None
        self.nii: spatialimages.SpatialImage = nii
        self.orientation = orientation

    @property
    def affine(self) -> np.ndarray:
        """The affine matrix of the NIfTI image."""
        if self.nii.affine is not None:
            return self.nii.affine
        else:
            raise ValueError("Affine matrix of the NIfTI image is None.")

    def resample_self_to(
        self, target_nii: spatialimages.SpatialImage | NiiBase, **kwargs
    ) -> NiiBase:
        """Return a new NiiBase with this image resampled onto target_nii's grid."""
        # Create new object of with resampled data
        if isinstance(target_nii, NiiBase):
            target_nii = target_nii.nii
        resampled_nii = nifti.resample_nifti(self.nii, target_nii, **kwargs)
        return NiiBase(resampled_nii, orientation=self.orientation)

    def images(
        self,
        orientation: tuple[str, str, str] | None = None,
        display_plane: nifti.DISPLAY_PLANES | None = None,
        caching: bool = True,
        **kwargs,
    ) -> npt.NDArray[np.float64]:
        """Return the image data array, oriented per orientation or display_plane."""
        if orientation is None and display_plane is None:
            orientation = self.orientation
        return nifti.orient_nifti(
            self.nii,
            orientation=orientation,
            display_plane=display_plane,
            caching=caching,
            **kwargs,
        )

    def with_new_data(
        self,
        new_data: npt.NDArray | xr.DataArray,
        new_affine: npt.NDArray | None = None,
    ) -> NiiBase:
        """Return a new NiiBase wrapping new_data, keeping this image's header/affine."""
        if new_affine is None and new_data.shape != self.nii.shape:
            # A change in shape is okay if the voxel sizes are different. Trust the
            # user that a correct affine is provided
            raise ValueError(
                f"New data shape {new_data.shape} does not match original "
                f"NIfTI shape {self.nii.shape}."
            )

        if new_affine is not None:
            affine = new_affine
        else:
            if self.nii.affine is None:
                raise ValueError(
                    "Original NIfTI affine is None, cannot create new NIfTI with new data."
                    "\nPass new_affine argument to specify affine for new NIfTI."
                )
            else:
                affine = self.nii.affine

        # Image data is to be oriented the following
        orientated_data = nib.orientations.apply_orientation(
            new_data, nib.orientations.axcodes2ornt(("P", "L", "S"))
        )

        new_nii = self.nii.__class__(
            orientated_data,  # pyright: ignore[reportArgumentType]
            affine=affine.copy(),
            header=self.nii.header,
            extra=self.nii.extra,
            file_map=self.nii.file_map,
        )

        new_nii.set_data_dtype(new_data.dtype)

        return NiiBase(new_nii, orientation=self.orientation)

    def apply_mask(self, mask: npt.NDArray | NiiBase) -> npt.NDArray:
        """Return the image data with voxels outside mask set to NaN."""
        images = self.images()
        if isinstance(mask, NiiBase):
            mask = mask.images()
        if mask.shape != images.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {images.shape}.")
        masked_images = np.where(mask, images, np.nan)
        return masked_images

    def display(
        self,
        orientation: tuple[str, str, str] | None = None,
        display_plane: nifti.DISPLAY_PLANES | None = None,
        mask: npt.NDArray | NiiBase | None = None,
        **kwargs,
    ) -> None:
        """Display the image, optionally masked, via plotting.images.display_images.

        Displays an interactive image-grid widget and returns nothing.
        """
        # Check if orientation or display_plane is provided in kwargs and orient images
        logger.debug(
            f"Displaying NIfTI with orientation {orientation} and display plane {display_plane}."
        )
        images = self.images(orientation=orientation, display_plane=display_plane)
        # If mask is provided, check if it has same shape as images and apply mask to images
        if mask is not None:
            try:
                images = self.apply_mask(mask)
            except ValueError as e:
                logger.error(f"Error applying mask: {e}. Displaying unmasked images.")

        plotting.images.display_images(images, **kwargs)

    def overlay_on_T1(
        self,
        t1: NiiBase | spatialimages.SpatialImage | npt.NDArray,
        resample_kwargs: dict | None = None,
        orientation: tuple[str, str, str] | None = None,
        display_plane: nifti.DISPLAY_PLANES | None = None,
        mask: npt.NDArray | NiiBase | None = None,
        **kwargs,
    ) -> None:
        """Overlay this image on a T1 image, resampling and masking as needed."""
        if resample_kwargs is None:
            resample_kwargs = {}
        if orientation is None and display_plane is None:
            orientation = self.orientation
        if isinstance(t1, np.ndarray):
            if mask is not None and isinstance(mask, NiiBase):
                mask = mask.images(orientation=orientation, display_plane=display_plane).astype(
                    bool
                )
            plotting.images.overlay_image_data_on_T1(
                t1,
                self.images(orientation=orientation, display_plane=display_plane),
                mask=mask,
                **kwargs,
            )
            return
        if isinstance(t1, NiiBase):
            t1 = t1.nii
        if isinstance(t1, (NiiBase, spatialimages.SpatialImage)):
            plotting.images.overlay_nifti_data_on_T1(
                t1,
                self.nii,
                orientation=orientation,
                display_plane=display_plane,
                resample_kwargs=resample_kwargs,
                mask=mask,
                **kwargs,
            )
        else:
            raise TypeError("t1 must be either a NIfTI image or a numpy array.")

    def get_brain_mask(
        self,
        create: bool = False,
        mask_keyword: str = "_bet.nii",
        **extract_kwargs,
    ) -> NiiBase:
        """Find (or optionally create) the brain mask NIfTI next to this image.

        Looks for a brain mask NIfTI (matching ``mask_keyword``) in the same
        folder as ``self.nii_path``. If none is found and ``create`` is True, a
        mask is generated with HD-BET via :func:`utils.images.extract_brain` and
        loaded. The mask is returned as its own :class:`NiiBase`.

        Parameters
        ----------
        create : bool, optional
            If True, run HD-BET brain extraction when no mask is found.
        mask_keyword : str, optional
            Substring identifying the brain mask NIfTI file.
        **extract_kwargs
            Forwarded to ``utils.images.extract_brain``.

        Returns
        -------
        NiiBase
            The brain mask wrapped as a NiiBase.

        Raises
        ------
        ValueError
            If this NIfTI was not loaded from a path (no folder to search).
        FileNotFoundError
            If no mask is found and ``create`` is False.
        """
        if self.nii_path is None:
            raise ValueError("Cannot locate a brain mask: this NIfTI was not loaded from a path.")
        folder = self.nii_path.parent
        mask_path = file_helpers.get_nifti_file(
            folder,
            filter_out_keywords=[],
            filter_in_keywords=[mask_keyword],
        )
        if mask_path is None:
            if not create:
                raise FileNotFoundError(
                    f"No brain mask (matching '{mask_keyword}') found in {folder}. "
                    "Pass create=True to generate one with HD-BET."
                )
            # Local import avoids the images <-> GESeries circular import and
            # defers the heavy torch/HD-BET import until a mask must be created.
            from .utils import images

            logger.info(f"No brain mask found in {folder}. Creating one with HD-BET.")
            extract_kwargs.setdefault("save_mask", True)
            extract_kwargs.setdefault("extract_brain", False)
            mask_path = images.extract_brain(self.nii_path, **extract_kwargs)

        return NiiBase(mask_path, orientation=self.orientation)

    @cached_property
    def brain_mask(self) -> NiiBase:
        """The brain mask as a NiiBase, loaded if present else created on access."""
        return self.get_brain_mask(create=True)

    def save(self, output_path: Path) -> None:
        """Save the NIfTI image to output_path as a Nifti1Image."""
        if self.nii_path is not None:
            logger.warning(
                f"NIfTI was originally loaded from {self.nii_path}. "
                f"Saving to a new location {output_path}."
            )
        # Create directory if it does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_helpers.move_files_with_glob(output_path.parent, f"*{output_path.stem}*")

        # To correctly store them in often required Nifti1 format, we need to
        # convert them to Nifti1Image before saving
        nifti1.Nifti1Image(
            self.nii.dataobj,
            self.nii.affine,
            self.nii.header,
            self.nii.extra,
            self.nii.file_map,
        ).to_filename(output_path)
        logger.info(f"Saved NIfTI to {output_path}.")


class MRISeries(NiiBase):
    """A reconstructed MRI series, loaded as a NIfTI from a scan's DATA_FOLDER."""

    def __init__(
        self,
        DATA_FOLDER: Path,
        SERIES_ID: int,
        orientation: tuple[str, str, str] = ("L", "P", "S"),
    ) -> None:
        self.DATA_FOLDER = DATA_FOLDER
        self.OUTPUT_FOLDER = (DATA_FOLDER / ".." / "output").resolve()
        self.SERIES_ID = SERIES_ID
        try:
            nii_path: Path = file_helpers.get_niftis_from_series(
                self.DATA_FOLDER, self.SERIES_ID, convert_dicoms=True
            )
            super().__init__(nii_path, orientation)
        except (ValueError, FileNotFoundError) as e:
            logger.warning(
                f"Could not create or find NIfTIs for series {self.SERIES_ID}. "
                f"Proceeding without initializing NiiBase.\nError: {e}"
            )


class RawMRISeries(MRISeries):
    """An MRI series with the associated raw reconstructed .mat data loaded."""

    def __init__(self, DATA_FOLDER: Path, SERIES_ID: int) -> None:
        super().__init__(DATA_FOLDER, SERIES_ID)
        self.mat_path: Path = file_helpers.get_mat_data_from_series(
            self.DATA_FOLDER, self.SERIES_ID
        )
        self.recon: dict = data_loaders.load_mat_file(self.mat_path)
        self.header: dict = self.recon.get("h", {})
        self.bb = self.recon.get("bb", np.array([]))
        self.bbabs = self.recon.get("bbabs", np.abs(self.bb))

    @cached_property
    def fids(self) -> xr.DataArray:
        """The raw FIDs, unchopped and conjugated, as a DataArray over id and time."""
        fids, _ = data_loaders.load_raw_fids(self.DATA_FOLDER, self.SERIES_ID)
        fids = fids.copy()
        if self.fids_are_chopped:
            fids[1::2, :] *= -1
        fids = np.conj(fids)
        return xr.DataArray(
            fids,
            dims=["id", "time"],
            coords={"time": np.arange(fids.shape[1]) * self.dwelltime},
        )

    @property
    def fids_are_chopped(self) -> bool:
        """Whether the FIDs are chopped or not."""
        value = self._get_header_value(["rdb_hdr", "data_collect_type"])
        if isinstance(value, float):
            is_chopped = int(value) % 2 == 0
            return is_chopped
        else:
            logger.error(
                "Value for data_collect_type in header does not have correct type. Returning False."
            )
            return False

    @property
    def pfile_number(self) -> int:
        """The P-file number of the scan."""
        value = self._get_header_value(["rdb_hdr", "run_int"])
        if isinstance(value, float):
            return int(value)
        else:
            return -1

    @property
    def protocol_name(self) -> str:
        """The protocol name of the scan."""
        value = self._get_header_value(["series", "prtcl"])
        if isinstance(value, str):
            return str(value)
        else:
            return ""

    @property
    def series_name(self) -> str:
        """The series name of the scan."""
        value = self._get_header_value(["series", "se_desc"])
        if isinstance(value, str):
            return str(value)
        else:
            return ""

    @property
    def exam_number(self) -> int:
        """The exam number of the scan."""
        value = self._get_header_value(["exam", "ex_no"])
        if isinstance(value, float):
            return int(value)
        else:
            return -1

    @property
    def scan_datetime(self) -> datetime:
        """The scan datetime."""
        date = self._get_header_value(["rdb_hdr", "scan_date"])
        time = self._get_header_value(["rdb_hdr", "scan_time"])
        if isinstance(date, str) and isinstance(time, str):
            datetime_str = f"{date} {time}"
            return datetime.strptime(datetime_str, "%m/%d/1%y %H:%M")
        else:
            return datetime.max

    @property
    def scan_date(self) -> date:
        """The scan date."""
        return self.scan_datetime.date()

    @property
    def scan_time(self) -> time:
        """The scan time."""
        return self.scan_datetime.time()

    @property
    def transients(self) -> float:
        """The number of transients."""
        value = self._get_header_value(["rdb_hdr", "nframes"])
        if isinstance(value, (int, float)):
            return float(value)
        else:
            return np.nan

    @property
    def bandwidth(self) -> float:
        """The bandwidth of the scan (Hz)."""
        value = self._get_header_value(["rdb_hdr", "user0"])
        if isinstance(value, (int, float)):
            return float(value)
        else:
            return np.nan

    @property
    def dwelltime(self) -> float:
        """The dwell time of the scan (s, 1/Hz)."""
        return 1 / self.bandwidth

    @property
    def deadtime(self) -> float:
        """The dead time is assumed to be a single echo time in seconds."""
        return self.echo_time / 1e6  # Convert from microseconds to seconds

    @property
    def centre_frequency(self) -> float:
        """The centre frequency of the scan (MHz)."""
        value = self._get_header_value(["rdb_hdr", "ps_mps_freq"])
        if isinstance(value, (int, float)):
            return float(value) / 1e7  # For some reason stored in Hz x10
        else:
            return np.nan

    @property
    def nucleus(self) -> int:
        """The nucleus of the scan as integer (e.g. 1 for 1H, 13 for 13C)."""
        value = self._get_header_value(["image", "specnuc"])
        # Try casting to int if not possible return 0
        try:
            return int(value)  # pyright: ignore[reportArgumentType]
        except (ValueError, TypeError):
            logger.error(f"Failed to convert nucleus value '{value}' to integer.")
            return 0

    @property
    def carrier_ppm(self) -> float:
        """The carrier ppm of the scan centre frequency (ppm). Based on nucleus."""
        if self.nucleus in [1, 2]:
            return 4.68
        else:
            return 0

    @property
    def echo_time(self) -> float:
        """The echo time (TE) of the scan (us)."""
        value = self._get_header_value(["rdb_hdr", "te"])
        if isinstance(value, (int, float)):
            return value
        else:
            return np.nan

    @property
    def field_of_view(self) -> float:
        """The field of view (FOV) of the scan (mm)."""
        value = self._get_header_value(["rdb_hdr", "fov"])
        if isinstance(value, (int, float)):
            return float(value)
        else:
            return np.nan

    @property
    def flip_angle(self) -> float:
        """The flip angle of the scan (in degrees)."""
        value = self._get_header_value(["image", "mr_flip"])
        if isinstance(value, (int, float)):
            return float(value)
        else:
            return np.nan

    @property
    def inversion_time(self) -> float:
        """The inversion time (TI) of the scan (us)."""
        value = self._get_header_value(["rdb_hdr", "user25"])
        if isinstance(value, float):
            return float(value)
        else:
            return np.nan

    @property
    def number_of_excitations(self) -> int:
        """The number of excitations (NEX) of the scan."""
        value = self._get_header_value(["image", "nex"])
        if isinstance(value, float):
            return int(value)
        else:
            return -1

    @property
    def repetition_time(self) -> float:
        """The repetition time (TR) of the scan (ms)."""
        value = self._get_header_value(["image", "tr"])
        if isinstance(value, float):
            return float(value) / 1e3
        else:
            return np.nan

    @property
    def rf_pulse_type(self) -> int:
        """The RF pulse type used in the scan."""
        value = self._get_header_value(["rdb_hdr", "user14"])
        if isinstance(value, float):
            return int(value)
        else:
            return -1

    @property
    def slice_thickness(self) -> float:
        """The slice thickness of the scan (mm)."""
        value = self._get_header_value(["image", "slthick"])
        if isinstance(value, float):
            return float(value)
        else:
            return np.nan

    @property
    def scan_duration(self) -> timedelta:
        """The scan duration."""
        value = self._get_header_value(["image", "sctime"])
        if isinstance(value, float):
            return timedelta(microseconds=value)
        else:
            raise ValueError("Scan duration not found in header.")

    @property
    def reconstruction_used(self) -> int:
        """The reconstruction method used."""
        value = self._get_header_value(["rdb_hdr", "recon"])
        if isinstance(value, float):
            return int(value)
        else:
            return -1

    @property
    def waveform_type_scan_mode(self) -> float:
        """The waveform type or scan mode."""
        value = self._get_header_value(["rdb_hdr", "user3"])
        if isinstance(value, float):
            return float(value)
        else:
            return np.nan

    @property
    def receive_gain_1(self) -> float:
        """The first receive gain."""
        value = self._get_header_value(["rdb_hdr", "ps_mps_r1"])
        if isinstance(value, float):
            return float(value)
        else:
            return np.nan

    @property
    def receive_gain_2(self) -> float:
        """The second receive gain."""
        value = self._get_header_value(["rdb_hdr", "ps_mps_r2"])
        if isinstance(value, float):
            return float(value)
        else:
            return np.nan

    @property
    def transmit_gain(self) -> int:
        """The transmit gain (TG)."""
        value = self._get_header_value(["rdb_hdr", "ps_mps_tg"])
        if isinstance(value, float):
            return int(value)
        else:
            return -1

    @property
    def field_strength(self) -> float:
        """The field strength of the scan (T)."""
        value = self._get_header_value(["mrconfig", "fieldStrength"])
        if isinstance(value, float):
            return float(value) / 10000
        else:
            return np.nan

    def _get_header_value(self, key_path: list[str]) -> None | str | np.double | npt.NDArray:
        """Traverse the header dictionary using a key path.

        Raises AttributeError if the path is invalid.
        """
        value = self.header
        try:
            for key in key_path:
                value = value[key]
            # Depending on the data type, return the appropriate value
            # If string
            if value is None:
                raise KeyError
            elif isinstance(value, str):
                return value
            elif isinstance(value, (np.ndarray, np.number, int, float)):
                value = np.asarray(value)
                if value.size == 1:
                    return np.double(value.item())
                else:
                    return value.flatten()
            # Any other type is treated as a missing/unsupported value.
            return None

        except (TypeError, KeyError):
            raise AttributeError(
                f"Mapped attribute not found. Expected path: {' -> '.join(key_path)}"
            )


class MRSISeries(RawMRISeries):
    """A raw MRSI series: spectroscopic imaging spectra, fitting, and voxel maps."""

    def __init__(
        self, DATA_FOLDER: Path, SERIES_ID: int, load_processed_data: bool = False
    ) -> None:
        super().__init__(DATA_FOLDER, SERIES_ID)
        # Expect spec and header to be present
        spec_data = self.recon.get("spec", np.ndarray([]))
        if not isinstance(spec_data, np.ndarray) or spec_data.size == 0:
            logger.error(
                "Expected 'spec' to be a non-empty numpy array. Not initializing object properly"
            )
        else:
            self.spec: xr.DataArray = xr.DataArray(
                spec_data,
                dims=["chemical_shift", "i", "j", "k"],
                coords={
                    "chemical_shift": spectra.calculate_ppm_axis(
                        spectral_width=self.bandwidth,
                        frequency=self.centre_frequency,
                        carrier_ppm=self.carrier_ppm,
                        npts=spec_data.shape[0],
                    ),
                    "i": np.arange(spec_data.shape[1]),
                    "j": np.arange(spec_data.shape[2]),
                    "k": np.arange(spec_data.shape[3]),
                },
                attrs={
                    "field_strength": self.field_strength,
                    "reference_frequency": self.centre_frequency,
                    "carrier_ppm": self.carrier_ppm,
                    "spectral_width": self.bandwidth,
                    "deadtime": self.deadtime,
                },
            )

            self.dims: tuple = self.spec.shape[1:4]
            self.npts: int = self.spec.shape[0]
            self.ppm: xr.DataArray = self.spec.coords["chemical_shift"]
            self.RAW_exp: NiiBase = self.create_MRSI_nii()
            # Update attributes of spec
            self.spec.attrs.update(
                {
                    "affine": self.RAW_exp.affine.tolist(),
                    "orientation": self.RAW_exp.orientation,
                }
            )

        self.fitted_metabolite_maps: dict = {}
        self.goodness_of_fit_maps: dict = {}
        if load_processed_data:
            self.load_processed_data()

    @cached_property
    def spec_flat(self) -> xr.DataArray:
        """The spectra with the voxel dimensions stacked into a single voxel dim."""
        return self.spec.stack(voxel=self.spec.dims[1:4])

    @cached_property
    def fids_flat(self) -> xr.DataArray:
        """The FIDs with the voxel dimensions stacked into a single voxel dim."""
        return self.fids.stack(voxel=self.fids.dims[1:4])

    @cached_property
    def fids(self) -> xr.DataArray:
        """The FIDs, derived from the spectra via FFT."""
        return self.spec.xmr.to_hz().xmr.to_fid()

    @cached_property
    def avg_fid(self) -> xr.DataArray:
        """Average fid across voxels with SNR above threshold."""
        return self.fids.where(self.get_SNR_mask()).mean(["i", "j", "k"], skipna=True)

    @cached_property
    def SNR_map(self) -> xr.DataArray:
        """The per-voxel SNR, computed from the FIDs via pyAMARES.fidSNR."""
        return xr.apply_ufunc(
            pyAMARES.fidSNR, self.fids, input_core_dims=[["time"]], vectorize=True
        )

    @property
    def voxel_size(self) -> float:
        """The in-plane voxel size (mm), derived from the field of view and matrix."""
        mtx = self.dims[0:2]
        # Assert that mtx is the same in x and y
        if mtx[0] != mtx[1]:
            logger.warning(
                f"Matrix size in x and y are not the same: {mtx[0]} vs {mtx[1]}. "
                "Using x size for voxel size calculation."
            )
        return self.field_of_view / mtx[0]

    @cached_property
    def default_SNR_mask(self) -> xr.DataArray:
        """The default SNR mask, computed with get_SNR_mask's default threshold."""
        return self.get_SNR_mask()

    def get_SNR_mask(self, threshold=1.5) -> xr.DataArray:
        """Return a boolean mask of voxels with SNR at or above threshold."""
        return self.SNR_map >= threshold

    def create_MRSI_affine(self) -> npt.NDArray:
        """Return an affine matrix rescaled to the MRSI voxel size and offset."""
        # Create new affine matrix based on original nifti affine and mat voxel size
        # This assumes the same orientation and slice thickness and spacing

        if self.nii.affine is None:
            raise ValueError("The affine matrix of the NIfTI image is None.")

        # Check if the original affine has uniform scaling in x and y
        if not np.isclose(self.nii.affine[0, 0], self.nii.affine[1, 1]):
            logger.warning(
                "Original affine matrix has non-uniform scaling in x and y: "
                f"{self.nii.affine[0, 0]} vs {self.nii.affine[1, 1]}. "
                "Using x scaling for new affine."
            )

        interpolated_voxel_sizes = self.nii.affine[0, 0]
        scaling_factor = self.voxel_size / interpolated_voxel_sizes

        new_affine = np.copy(self.nii.affine)
        new_affine[0, 0] *= scaling_factor
        new_affine[1, 1] *= scaling_factor
        # Round voxel sizes to 3 decimal places to avoid floating point issues
        new_affine[0:3, 0:3] = np.round(new_affine[0:3, 0:3], 3)

        # Offset x and y by centre offset
        # The shift is important as we want to make sure the overlay is in the correct position
        # The correct shift is the following
        shift = (np.diag(new_affine[:3]) - np.diag(self.nii.affine[:3])).round(3) / 2
        # However this creates the correct FOV but the overlay is not in the correct position
        # For a correct shift we need to add an additonal half voxel
        shift += np.array([new_affine[0, 0], new_affine[1, 1], 0]) / 2
        logger.trace(
            f"Calculated shift for new affine: {shift}. "
            f"Original affine translation: {self.nii.affine[:3, 3]}"
        )
        new_affine[:3, 3] += shift

        return new_affine

    def create_MRSI_nii(self) -> NiiBase:
        """Return the RAW magnitude image as a NiiBase, with the MRSI affine."""
        magnitude_data = np.sum(np.abs(self.spec), axis=0).astype(np.float32)
        return self.with_new_data(new_data=magnitude_data, new_affine=self.create_MRSI_affine())

    def create_map_nii(self, map_data: npt.NDArray) -> NiiBase:
        """Wrap a per-voxel map array as a NiiBase, with the MRSI affine."""
        # Make sure map_data has correct shape
        if map_data.shape != self.dims:
            raise ValueError(
                f"map_data shape {map_data.shape} does not match expected dimensions {self.dims}."
            )
        return self.with_new_data(new_data=map_data, new_affine=self.create_MRSI_affine())

    def get_voxel_spectrum(self, x: int, y: int, slice: int) -> xr.DataArray:
        """Return the spectrum at voxel (x, y, slice)."""
        logger.debug(
            f"Assuming voxel coordinates (x, y, slice) = ({x}, {y}, {slice}) "
            "are in correct order and within bounds."
        )
        return self.spec.sel(i=x, j=y, k=slice)

    def fit_single_voxel(
        self,
        x: int,
        y: int,
        slice: int,
        init_params: dict,
        fit_params: dict | None = None,
    ):
        """Fit the FID at voxel (x, y, slice) with AMARES."""
        if fit_params is None:
            fit_params = {}
        fid = self.fids.sel(i=x, j=y, k=slice)

        # Include series information in init_params
        passed_init_params = {
            "MHz": self.centre_frequency,
            "sw": self.bandwidth,
            "deadtime": self.deadtime,
        }
        passed_init_params.update(init_params)

        fit_result = fitting.AMARES.fit_single_fid(
            fid,
            passed_init_params,
            fit_params,
        )
        return fit_result

    def fit_all_voxels(
        self,
        init_params: dict,
        batch_fitting_params: dict | None = None,
        bad_fit_chisqr_threshold: float = -1.0,
        testing_mode: bool = False,
    ) -> npt.NDArray:
        """Fit all voxels above the SNR threshold with AMARES and store the maps.

        Parameters
        ----------
        init_params : dict
            Initial fit parameters, merged with series-derived MHz/sw/deadtime.
        batch_fitting_params : dict, optional
            Forwarded to ``fitting.AMARES.fit_multiple_fids``.
        bad_fit_chisqr_threshold : float, optional
            Chi-squared threshold above which a fit is flagged as bad. Defaults to -1.0
            (disabled).
        testing_mode : bool, optional
            If True, limit fitting to the first 100 voxels above the SNR threshold.

        Returns
        -------
        numpy.ndarray
            Flattened voxel indices of fits flagged as bad.
        """
        if batch_fitting_params is None:
            batch_fitting_params = {}
        SNR_mask = self.get_SNR_mask()
        # If in testing mode, limit to first 100 voxels with SNR above threshold
        if testing_mode:
            flat_SNR_mask = SNR_mask.stack(voxel=SNR_mask.dims)
            indices = np.where(flat_SNR_mask)[0][:100]
            SNR_mask = xr.full_like(flat_SNR_mask, False)
            SNR_mask[indices] = True
            SNR_mask = SNR_mask.unstack("voxel")

        # Include series information in init_params
        passed_init_params = {
            "MHz": self.centre_frequency,
            "sw": self.bandwidth,
            "deadtime": self.deadtime,
        }
        passed_init_params.update(init_params)

        results = fitting.AMARES.fit_multiple_fids(
            self.fids.where(SNR_mask).stack(voxel=SNR_mask.dims).dropna("voxel"),
            batch_fitting_params=batch_fitting_params,
            init_params=passed_init_params,
        )

        # Extract results back into original shape
        extracted_results = fitting.AMARES.extract_from_fit_results(
            results, bad_fit_chisqr_threshold
        )

        # Save extracted results into processing data folder with additional attributes
        additional_attributes: dict[str, str | int] = {
            "SERIES_ID": self.SERIES_ID,
            "exam_number": self.exam_number,
            "ACQUISITION_TIME": self.scan_datetime.isoformat(),
            "SAVED_ON": datetime.now().isoformat(),
        }
        additional_datasets = {
            "SNR_mask": SNR_mask.values,
        }

        fitting.AMARES.save_extracted_results(
            self.OUTPUT_FOLDER
            / "processing_data"
            / "dev"
            / f"Series{self.SERIES_ID:02d}_extracted_results.h5",
            extracted_results,
            additional_attributes,
            additional_datasets=additional_datasets,
            prepend_datetime=True,
        )

        combined_fit_results = extracted_results.combined_fit_results
        fitted_metabolites_temp = np.full((*self.dims, combined_fit_results.columns.size), np.nan)
        fitted_metabolites_temp[SNR_mask] = combined_fit_results.loc[
            (slice(None), "amplitude", "value"), :
        ].to_numpy()
        for i, metabolite in enumerate(combined_fit_results.columns):
            self.fitted_metabolite_maps[metabolite] = fitted_metabolites_temp[:, :, :, i]

        goodness_of_fit = extracted_results.goodness_of_fit
        gof_metrics_temp = np.full((*self.dims, goodness_of_fit.columns.size), np.nan)
        gof_metrics_temp[SNR_mask] = goodness_of_fit.to_numpy()
        for i, metric in enumerate(goodness_of_fit.columns):
            self.goodness_of_fit_maps[metric] = gof_metrics_temp[:, :, :, i]

        # The badfit_ids correspond to the flattened indices of the fitted voxels
        # Create mapping from badfit_ids to original voxel flattened indices
        if extracted_results.badfit_ids.size == 0:
            original_badfit_ids = np.array([], dtype=int)
        else:
            original_badfit_ids = np.where(SNR_mask.flatten())[0][extracted_results.badfit_ids]

        return original_badfit_ids

    def save_processed_data(self, output_folder: Path | None = None) -> Path:
        """Save fitted maps, FIDs, and identifying attributes to an HDF5 file.

        Parameters
        ----------
        output_folder : Path, optional
            Directory to save into. Defaults to ``OUTPUT_FOLDER / "processing_data"``.

        Returns
        -------
        Path
            Path to the saved (or, if a file already exists, the existing) HDF5 file.

        Raises
        ------
        OSError
            If the saved file fails the post-save integrity check.
        """
        attrs_to_save = [
            "SERIES_ID",  # Necessary to identify series later, do not remove
            "exam_number",  # Necessary to identify series later, do not remove
        ]

        data_to_save = [
            "fids",
            "avg_fid",
            "SNR_map",
            "default_SNR_mask",
        ]

        # Save results to output folder / processing_data
        if output_folder is None:
            output_folder = self.OUTPUT_FOLDER / "processing_data"
        output_folder.mkdir(exist_ok=True, parents=True)

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Series{self.SERIES_ID:02d}_processing_data.h5"

        # Move previous runs to old folder. Making it easier to determine which is the latest
        file_helpers.move_files_with_glob(output_folder, f"*{filename}*")

        filename = f"{date_str}_{filename}"
        full_path = output_folder / filename

        # If file exists, log warning and return
        if full_path.exists():
            logger.warning(
                f"File {full_path} already exists. Not overwriting. Returning existing file path."
            )
            return full_path

        with h5py.File(full_path, "a") as f:
            # Save fitted metabolite maps
            fitted_data_group = f.create_group("fitted_data")
            for metabolite, metabolite_map in self.fitted_metabolite_maps.items():
                logger.debug(f"Saving metabolite map for {metabolite}")
                fitted_data_group.create_dataset(
                    metabolite, data=metabolite_map, dtype=metabolite_map.dtype
                )

                # Save goodness of fit metrics
            gof_group = fitted_data_group.create_group("goodness_of_fit")
            for goodness_metric, metric_map in self.goodness_of_fit_maps.items():
                logger.debug(f"Saving goodness of fit metric map for {goodness_metric}")
                gof_group.create_dataset(goodness_metric, data=metric_map, dtype=metric_map.dtype)

            # Save acquisition time of series in HDF5 in readable format
            f.attrs["ACQUISITION_TIME"] = self.scan_datetime.strftime("%Y-%m-%d_%H:%M:%S")
            f.attrs["saved_on"] = datetime.now().isoformat()
            for attr in attrs_to_save:
                value = getattr(self, attr)
                logger.debug(f"Saving attribute {attr} with value {value}")
                f.attrs[attr] = value

            # Save additional attributes
            for attr in data_to_save:
                data = getattr(self, attr)
                logger.debug(f"Saving attribute {attr}")
                if hasattr(data, "dtype"):
                    f.create_dataset(attr, data=data, dtype=data.dtype)
                else:
                    f.create_dataset(attr, data=data)

        logger.info(f"Saved processed data to {'/'.join(full_path.parts[-4:])}")
        logger.debug("Checking saved data.")
        is_okay = self._check_saved_data(full_path)
        if not is_okay:
            logger.error("Saved data did not pass the integrity check! Renaming file!!")
            full_path.rename(full_path.with_suffix(".corrupt"))
            raise OSError(
                "Saved data did not pass the integrity check! "
                "Debug this to make sure data is saved correctly."
            )
        else:
            logger.debug("Saved data passed the integrity check.")

        return full_path

    def load_processed_data(
        self, output_folder: Path | None = None, force_load: bool = False
    ) -> bool:
        """Load the most recent processed-data HDF5 file for this series.

        Parameters
        ----------
        output_folder : Path, optional
            Directory to search in. Defaults to ``OUTPUT_FOLDER / "processing_data"``.
        force_load : bool, optional
            If True, load the data even if SERIES_ID, exam_number, or acquisition
            time do not match this series.

        Returns
        -------
        bool
            True if data was loaded successfully, False otherwise.
        """
        if output_folder is None:
            output_folder = self.OUTPUT_FOLDER / "processing_data"

        try:
            old_processed_data = file_helpers.get_latest_processing_file(
                output_folder, self.SERIES_ID
            )
        except FileNotFoundError:
            logger.warning(
                f"No processed data file found in directory {output_folder}. "
                "Returning without loading."
            )
            return False
        with h5py.File(old_processed_data, "r") as f:
            # Do some sanity checks, check Series id, exam_number and acquisition time
            try:
                same_series_id = f.attrs["SERIES_ID"] == self.SERIES_ID  # type: ignore
                same_exam_number = f.attrs["exam_number"] == self.exam_number  # type: ignore
                same_acq_time = (
                    datetime.strptime(
                        f.attrs["ACQUISITION_TIME"],  # type: ignore
                        "%Y-%m-%d_%H:%M:%S",
                    )
                    == self.scan_datetime
                )
                logger.debug(
                    f"Series ID match: {same_series_id}, "
                    f"Exam number match: {same_exam_number}, "
                    f"Acquisition time match: {same_acq_time}"
                )
                if not all([same_series_id, same_exam_number, same_acq_time]):
                    logger.warning("Mismatch found in saved attributes! Not loading data")
                    if force_load:
                        logger.warning("Force loading data despite mismatches.")
                    else:
                        raise ValueError("Cannot load data due to mismatches in attributes.")
            except KeyError as e:
                logger.error(
                    f"KeyError: {e}. The following keys must be present in the "
                    "file: 'SERIES_ID', 'exam_number', 'acquisition_time'."
                )
                logger.error(f"Tree of the HDF5 file:\n{file_helpers.print_hdf5_tree(f)}")
                return False

            # Load fitted metabolite maps
            self.fitted_metabolite_maps = {}
            self.goodness_of_fit_maps = {}
            try:
                fitted_data_group = f["fitted_data"]
                if isinstance(fitted_data_group, h5py.Group):
                    for key in fitted_data_group.keys():
                        data = fitted_data_group[key]
                        if isinstance(data, h5py.Dataset):
                            metabolite_map = data[:]
                            self.fitted_metabolite_maps[key] = metabolite_map
                            logger.debug(f"Loaded metabolite map for {key}")
                            logger.debug(
                                f"  dtype: {metabolite_map.dtype}, shape: {metabolite_map.shape}"
                            )
                            logger.debug(f"  sum: {np.nansum(metabolite_map)}")
                        if isinstance(data, h5py.Group):
                            logger.debug(f"Loaded group for {key}")
                            for subkey in data.keys():
                                subdata = data[subkey]
                                if isinstance(subdata, h5py.Dataset):
                                    subdata = subdata[:]
                                    self.goodness_of_fit_maps[subkey] = subdata
                                    logger.debug(f"  Loaded dataset for {subkey}")
                                    logger.debug(
                                        f"    dtype: {subdata.dtype}, shape: {subdata.shape}"
                                    )
                                    logger.debug(f"    sum: {np.nansum(subdata)}")
            except KeyError as e:
                logger.error(f"KeyError: {e}. The expected keys may not be present in the file.")
                logger.error(f"Tree of the HDF5 file:\n{file_helpers.print_hdf5_tree(f)}")

            # Check if fitted_metabolite_maps and goodness_of_fit_maps are loaded
            if not self.fitted_metabolite_maps:
                logger.warning(
                    "No fitted metabolite maps were loaded. Something went wrong! "
                    "Datasets probably not stored correctly in hdf5 file."
                )
            if not self.goodness_of_fit_maps:
                logger.warning(
                    "No goodness of fit maps were loaded. Something went wrong! "
                    "Datasets probably not stored correctly in hdf5 file."
                )
        return True

    def _check_saved_data(self, filename: Path) -> bool:
        """Check if the saved data file is valid.

        Validates that the file is an HDF5 file, contains maps in the 'fitted_data'
        group, and ensures the maps are nonzero in size and have 3D dimensions.

        Parameters
        ----------
        filename : Path
            The path to the file to check.

        Returns
        -------
        bool
            True if the file passes all checks, False otherwise.
        """
        if not filename.exists():
            logger.error(f"File {filename} does not exist.")
            return False

        if not h5py.is_hdf5(filename):
            logger.error(f"File {filename} is not a valid HDF5 file.")
            return False

        try:
            with h5py.File(filename, "r") as f:
                if "fitted_data" not in f:
                    logger.error(f"'fitted_data' group not found in {filename}.")
                    return False

                fitted_data_group = f["fitted_data"]
                if not isinstance(fitted_data_group, h5py.Group):
                    logger.error(f"'fitted_data' is not a group in {filename}.")
                    return False

                # Ensure 'fitted_data' group has at least one key
                if not any(
                    isinstance(fitted_data_group[key], h5py.Dataset)
                    for key in fitted_data_group.keys()
                ):
                    logger.error(f"'fitted_data' group in {filename} has no datasets.")
                    return False

                # Ensure 'goodness_of_fit' group has at least one key
                if "goodness_of_fit" in fitted_data_group:
                    gof_group = fitted_data_group["goodness_of_fit"]
                    if isinstance(gof_group, h5py.Group) and len(gof_group.keys()) == 0:
                        logger.error(f"'goodness_of_fit' group in {filename} has no keys.")
                        return False

                for key in fitted_data_group.keys():
                    dataset = fitted_data_group[key]
                    if isinstance(dataset, h5py.Dataset):
                        data = dataset[:]
                        if data.size == 0:
                            logger.error(
                                f"Dataset '{key}' in 'fitted_data' group is empty in {filename}."
                            )
                            return False
                        if data.ndim != 3:
                            logger.error(
                                f"Dataset '{key}' in 'fitted_data' group is not 3D in {filename}."
                            )
                            return False
                    elif isinstance(dataset, h5py.Group):
                        for subkey in dataset.keys():
                            subdataset = dataset[subkey]
                            if isinstance(subdataset, h5py.Dataset):
                                subdata = subdataset[:]
                                if subdata.size == 0:
                                    logger.error(
                                        f"Sub-dataset '{subkey}' in '{key}' group "
                                        f"is empty in {filename}."
                                    )
                                    return False
                                if subdata.ndim != 3:
                                    logger.error(
                                        f"Sub-dataset '{subkey}' in '{key}' group "
                                        f"is not 3D in {filename}."
                                    )
                                    return False
                    else:
                        logger.error(
                            f"Unexpected data type in 'fitted_data' group for "
                            f"key '{key}' in {filename}."
                        )
                        return False

                # Additional check: Ensure acquisition time is present and valid
                if "ACQUISITION_TIME" not in f.attrs:
                    logger.error(f"'ACQUISITION_TIME' attribute not found in {filename}.")
                    return False
                try:
                    acq_time_str = f.attrs["ACQUISITION_TIME"]
                    datetime.strptime(str(acq_time_str), "%Y-%m-%d_%H:%M:%S")
                    logger.debug(f"'ACQUISITION_TIME' in {filename} is valid: {acq_time_str}")
                except (TypeError, ValueError) as e:
                    logger.error(f"Invalid 'ACQUISITION_TIME' format in {filename}: {e}")
                    return False

        except (OSError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Error while checking file {filename}: {e}")
            return False

        logger.debug(f"File {'/'.join(filename.parts[-4:])} passed all checks.")
        return True

    def visualize_fitted_metabolite_map(
        self,
        metabolite_name: str,
        **kwargs,
    ):
        """Display the fitted metabolite map for metabolite_name."""
        plotting.images.display_images(
            self.fitted_metabolite_maps[metabolite_name],
            title=f"Fitted Metabolite Map: {metabolite_name}",
            colorbar=True,
            **kwargs,
        )

    def visualize_goodness_of_fit_map(
        self,
        metric_name: str,
        **kwargs,
    ):
        """Display the goodness-of-fit map for metric_name."""
        plotting.images.display_images(
            self.goodness_of_fit_maps[metric_name],
            title=f"Goodness of Fit Map: {metric_name}",
            colorbar=True,
            **kwargs,
        )

    def visualize_fitted_values(self):
        """Not yet implemented."""
        ...


class MRSSeries(RawMRISeries):
    """A single-voxel MRS series: FIDs, spectra, averaging, and fitting."""

    def __init__(self, DATA_FOLDER: Path, SERIES_ID: int) -> None:
        super().__init__(DATA_FOLDER, SERIES_ID)
        spec_data = self.recon.get("spec", np.ndarray([]))
        if not isinstance(spec_data, np.ndarray) or spec_data.size == 0:
            logger.error(
                "Expected 'spec' to be a non-empty numpy array. Not initializing object properly"
            )
        else:
            # This assumes no zero filling was applied yet
            self.npts: int = spec_data.shape[1]
            self.spec: xr.DataArray = xr.DataArray(
                spec_data,
                dims=["averages", "chemical_shift"],
                coords={
                    "averages": np.arange(spec_data.shape[0]),
                    "chemical_shift": spectra.calculate_ppm_axis(
                        spectral_width=self.bandwidth,
                        frequency=self.centre_frequency,
                        carrier_ppm=self.carrier_ppm,
                        npts=spec_data.shape[1],
                    ),
                },
                attrs={
                    "field_strength": self.field_strength,
                    "reference_frequency": self.centre_frequency,
                    "carrier_ppm": self.carrier_ppm,
                    "units": "a.u.",
                    "spectral_width": self.bandwidth,
                    "deadtime": self.deadtime,
                },
            )
            self.averages: int = self.spec.sizes["averages"]
            self.ppm: xr.DataArray = self.spec.coords["chemical_shift"]

    @cached_property
    def fids(self) -> xr.DataArray:
        """The FIDs from RawMRISeries, with the id coordinate renamed to averages."""
        # Return the fids from RawMRISeries but rename coordinate id to averages
        super().fids  # Ensure fids is cached in RawMRISeries before renaming
        fids = super().fids.rename({"id": "averages"})
        return fids.assign_attrs(**self.spec.attrs)

    @cached_property
    def avg_fid(self) -> xr.DataArray:
        """The FID averaged across all averages."""
        return self.fids.mean("averages").assign_attrs(**self.spec.attrs)

    @cached_property
    def avg_spec(self) -> xr.DataArray:
        """The spectrum averaged across all averages."""
        # WARNING: fid to spectrum has different scaling than avg spec!! this is
        # probably due to scaling in the mat data
        logger.trace(
            "Calculating average spectrum from averaged FIDs. Note that this may "
            "have different scaling than the average of the spectra due to "
            "scaling in the mat data."
        )
        return self.spec.mean("averages").assign_attrs(**self.spec.attrs)

    def group_size_by_duration(self, group_duration: float) -> int:
        """Calculate number of consecutive FIDs to average for a target duration.

        Parameters
        ----------
        group_duration : float
            Target duration in seconds for each group.

        Returns
        -------
        int
            Number of consecutive FIDs to average together.
        """
        scan_duration_sec = self.scan_duration.total_seconds()
        time_per_fid = scan_duration_sec / self.averages

        if scan_duration_sec % group_duration != 0:
            actual_group_duration = (group_duration * self.averages) / scan_duration_sec
            logger.warning(
                f"Scan duration {scan_duration_sec:.2f}s is not an exact multiple "
                f"of group duration {group_duration:.2f}s.\n Actual group "
                f"duration will be {actual_group_duration:.2f}s."
            )

        group_size = int(group_duration / time_per_fid)

        return group_size

    def average_fids(self, group_size: int | None = 8) -> xr.DataArray:
        """Average FIDs by grouping consecutive acquisitions.

        Parameters
        ----------
        group_size : int, optional
            Number of consecutive FIDs to average together. Defaults to 8.

        Returns
        -------
        xarray.DataArray
            Averaged FIDs, shape `(n_groups, n_timepoints)`.
        """
        if group_size is None:
            group_size = 8
        # Create bin edges: [0, group_size, 2*group_size, ..., self.averages]
        bins = np.arange(0, self.averages + group_size, group_size)

        # Group by bins and average across averages dimension
        return (
            self.fids.groupby_bins("averages", bins=bins)
            .mean("averages")
            .assign_attrs(**self.spec.attrs)
        )

    def average_spectra(self, group_size: int | None = 8) -> xr.DataArray:
        """Average spectra by grouping consecutive acquisitions.

        Parameters
        ----------
        group_size : int, optional
            Number of consecutive FIDs to average together. Defaults to 8.

        Returns
        -------
        xarray.DataArray
            Averaged spectra, shape `(n_groups, n_points)`.
        """
        # Average grouped FIDs
        averaged_fids = self.average_fids(group_size=group_size)

        # Apply FFT along time dimension
        return averaged_fids.xmr.to_spectrum()

    def average_spectra_by_duration(self, group_duration: int | None = 10) -> xr.DataArray:
        """Average spectra by grouping acquisitions within a time duration.

        Parameters
        ----------
        group_duration : int, optional
            Duration in seconds for each group. Defaults to 10.

        Returns
        -------
        xarray.DataArray
            Averaged spectra grouped by duration.
        """
        if group_duration is None:
            group_duration = 10
        group_size = self.group_size_by_duration(group_duration)
        return self.average_spectra(group_size=group_size)

    def average_fids_by_duration(self, group_duration: int | None = 10) -> xr.DataArray:
        """Average FIDs by grouping acquisitions within a time duration.

        Parameters
        ----------
        group_duration : int, optional
            Duration in seconds for each group. Defaults to 10.

        Returns
        -------
        xarray.DataArray
            Averaged FIDs grouped by duration.
        """
        if group_duration is None:
            group_duration = 10
        group_size = self.group_size_by_duration(group_duration)
        return self.average_fids(group_size=group_size)

    def phase_avg_spec(
        self,
        zero_order: float = 0.0,
        first_order: float = 0.0,
        autophase: bool = False,
        ppm_min: float = -np.inf,
        ppm_max: float = np.inf,
    ) -> xr.DataArray:
        """Phase (or autophase) the average spectrum, limited to a ppm range."""
        if autophase:
            phased_spec = self.spec.xmr.to_hz().xmr.autophase()
        else:
            phased_spec = self.spec.xmr.to_hz().xmr.phase(p0=zero_order, p1=first_order)

        return (
            phased_spec.mean("averages")
            .assign_attrs(**self.spec.attrs)
            .xmr.to_ppm()
            .sel(chemical_shift=slice(ppm_min, ppm_max))
        )

    def limit_spec_to_ppm_range(self, ppm_min: float = -2, ppm_max: float = 6) -> xr.DataArray:
        """Return the spectra limited to the chemical_shift range [ppm_min, ppm_max]."""
        return self.spec.sel(chemical_shift=slice(ppm_max, ppm_min))

    def fit_average_fid(
        self,
        init_params: dict,
        fit_params: dict | None = None,
        normalisation_factor: np.complexfloating | None = None,
        filter_by_ppm: tuple[float, float] | None = None,
    ):
        """Fit the averaged FID (avg_fid) with AMARES, normalizing it first."""
        if fit_params is None:
            fit_params = {}
        # Normalize avg_fid
        if normalisation_factor is None:
            scaling_factor = np.max(self.avg_fid) * self.number_of_excitations
            logger.warning(
                "Using averaged fid as normalisation factor. It is recommended to "
                "provide a normalisation_factor for consistent scaling across "
                "series. See debug for more details."
            )
        else:
            scaling_factor = normalisation_factor * self.number_of_excitations
        logger.debug(
            f"Scaling the FID by factor {scaling_factor:.4g} for fitting. "
            f"Number of excitations: {self.number_of_excitations}"
        )

        avg_fid = self.avg_fid / scaling_factor

        fit_result = fitting.AMARES.fit_single_fid(
            avg_fid,
            init_params,
            fit_params,
            raw_header=self.header,
            filter_by_ppm=filter_by_ppm,
        )
        return fit_result


class MRSWashinSeries(MRSSeries):
    """An MRS washin series: repeated spectra fitted and grouped over time."""

    def __init__(self, DATA_FOLDER: Path, SERIES_ID: int) -> None:
        super().__init__(DATA_FOLDER, SERIES_ID)
        self.fit_results: list = []

    @property
    def extracted_results(self):
        """The fit results extracted into a FitResults, or an empty one if unfit."""
        if not self.fit_results:
            logger.info(
                "No fit results present. Please run fit_grouped_by_duration() "
                "first. Returning empty results."
            )
            return fitting.AMARES.FitResults(np.asarray([]), pd.DataFrame(), pd.DataFrame())
        else:
            return self.extract_from_fit_results()

    def plot_washin(self, **kwargs):
        """Plot the duration-averaged spectra, one line per minute."""
        averaged_spec = self.average_spectra_by_duration(kwargs.pop("group_duration", None))
        minute_labels = [f"Min. {i + 1:.0f}" for i in range(averaged_spec.shape[0])]

        # averaged_spectra_per_minute
        return plotting.spectra.plot_spectra(
            data=averaged_spec.xmr.to_ppm(), labels=minute_labels, **kwargs
        )

    def fit_grouped_by_duration(
        self,
        init_params: dict,
        fit_params: dict | None = None,
        normalisation_factor: np.complexfloating | None = None,
        group_duration: int = 60,
    ) -> list:
        """Average FIDs by duration and fit each group's FID with AMARES."""
        if fit_params is None:
            fit_params = {}
        if normalisation_factor is None:
            logger.warning(
                "No normalisation_factor provided. This can lead to different "
                "scaling between different series."
            )

        averaged_fids = self.average_fids_by_duration(group_duration=group_duration)
        if normalisation_factor is None:
            scaling_factor = np.max(averaged_fids[0, :]) * self.number_of_excitations
            logger.warning(
                "Using first averaged fid as normalisation factor. It is "
                "recommended to provide a normalisation_factor for consistent "
                "scaling across series. See debug for more details."
            )
        else:
            scaling_factor = normalisation_factor * self.number_of_excitations
        logger.debug(
            f"Scaling the FIDs by factor {scaling_factor:.4g} for fitting. "
            f"Number of excitations: {self.number_of_excitations}"
        )

        fit_results = []
        for idx in range(averaged_fids.shape[0]):
            fid = averaged_fids[idx, :]
            # Normalize fid
            fid = fid / scaling_factor

            fit_result = fitting.AMARES.fit_single_fid(
                fid,
                init_params,
                fit_params,
                raw_header=self.header,
            )
            fit_results.append(fit_result)

        self.fit_results = fit_results
        return fit_results

    def extract_from_fit_results(
        self, index_key: str = "group_id", index_values: list | None = None
    ) -> fitting.AMARES.FitResults:
        """Restructure self.fit_results into a combined FitResults."""
        restructured_results = []
        for res in self.fit_results:
            data = (res.result_multiplets, res.out_obj)
            restructured_results.append(data)

        extracted_results = fitting.AMARES.extract_from_fit_results(
            restructured_results, index_key=index_key, index_values=index_values
        )
        return extracted_results
