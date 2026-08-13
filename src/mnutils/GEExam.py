from __future__ import annotations

from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger

from . import GESeries, plotting
from .utils import file_helpers

ANATOMICAL_SERIES_TO_SKIP = ["FGRE", "localizer", "GE_HOS", "localiser"]


class ExamBase:
    def __init__(
        self,
        BASE_FOLDER: str | Path,
        DATA_FOLDER: str | Path | None = None,
        OUTPUT_FOLDER: str | Path | None = None,
    ) -> None:
        self.BASE_FOLDER = Path(BASE_FOLDER)
        if DATA_FOLDER is None:
            self.DATA_FOLDER = self.BASE_FOLDER / "data"
        else:
            self.DATA_FOLDER = Path(DATA_FOLDER)
        if OUTPUT_FOLDER is None:
            self.OUTPUT_FOLDER = self.BASE_FOLDER / "output"
            # If it does not exist, create it
            self.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        else:
            self.OUTPUT_FOLDER = Path(OUTPUT_FOLDER)

        self.exam_overview = file_helpers.get_exam_overview(
            self.DATA_FOLDER, print_overview=True
        )
        self.series = self.exam_overview.index.tolist()
        self.series_dict = self._create_series_dict()

        self.all: dict[int, GESeries.MRISeries] = {}

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        folder_name = self.BASE_FOLDER.name
        return f"{class_name}({folder_name})"

    def __str__(self) -> str:
        width = 72
        line_sep = "\n" + "-" * width + "\n"
        topline = f"  {self.__repr__()}  ".center(width, "-") + "\n"
        return f"{topline}{self.exam_overview.to_string(max_colwidth=40)}{line_sep}"

    def __getitem__(self, key: int) -> GESeries.MRISeries:
        try:
            return self.all[key]
        except KeyError as ke:
            valid_series = list(self.all.keys())
            ke.add_note(
                f"Series ID {key} is not valid. Valid series IDs are: {valid_series}"
            )
            raise

    def __iter__(self) -> Iterator[tuple[int, GESeries.MRISeries]]:
        for series_id, series_obj in self.all.items():
            yield series_id, series_obj

    def _create_series_dict(self) -> dict[int, type[GESeries.MRISeries]]:
        series_class_dict = {}
        for series_num in self.series:
            folder_name = self.exam_overview.at[series_num, "Folder Name"]
            has_exam = self.exam_overview.at[series_num, "Has Exam"]
            has_dicom = self.exam_overview.at[series_num, "Has DICOM"]

            if has_exam and not has_dicom:
                logger.debug(
                    f"Series {series_num} does not have DICOM. Assuming it is an MRS series."
                )
                self.exam_overview.at[series_num, "Folder Name"] = "ASSUMING MRS"
                series_class_dict[series_num] = GESeries.MRSSeries
                continue

            # Make sure folder name is a string
            if isinstance(folder_name, str) and has_dicom:
                folder_name = folder_name.strip().lower()
            else:
                logger.warning(
                    f"Folder name for series {series_num} is not a string: {folder_name}. Skipping."
                )
                continue

            if not has_exam and not has_dicom:
                logger.debug(
                    f"Series {series_num} does not have exam or DICOM data. Skipping"
                )
                continue

            if has_dicom and not has_exam:
                logger.trace(f"Series {series_num} does not have raw data.")
                # Based on the folder name determine if it is anatomical data we want to load
                if any(
                    skip_name.lower() in folder_name
                    for skip_name in ANATOMICAL_SERIES_TO_SKIP
                ):
                    logger.debug(f"Series {series_num} is anatomical data to skip.")
                    continue
                series_class_dict[series_num] = GESeries.MRISeries
                continue

            if has_exam and has_dicom:
                logger.trace(
                    f"Series {series_num} has both exam and DICOM data. Checking folder name."
                )
                if "mrs_washin" in folder_name:
                    series_class_dict[series_num] = GESeries.MRSWashinSeries
                elif "mrsi_pseudo" in folder_name:
                    series_class_dict[series_num] = GESeries.MRSISeries
                elif "mrs_" in folder_name:
                    series_class_dict[series_num] = GESeries.MRSSeries
                elif "bs_prescan" in folder_name:
                    continue  # Skip the bs_prescan series
                else:
                    logger.warning(
                        f"Series {series_num} with folder name '{folder_name}' does not "
                        f"match any known series types. Skipping."
                    )

        return series_class_dict

    def load_all(self, overwrite: bool = False) -> None:
        for series_id, series_class in self.series_dict.items():
            if (
                overwrite
                or series_id not in self.all
                or not isinstance(self.all[series_id], series_class)
            ):
                self.all[series_id] = series_class(self.DATA_FOLDER, series_id)
            else:
                logger.debug(f"Series {series_id} already loaded, skipping.")

    def load_series(self, series_ids: int | list[int], overwrite: bool = False) -> None:
        if isinstance(series_ids, int):
            series_ids = [series_ids]
        for series_id in series_ids:
            if series_id not in self.series_dict:
                logger.warning(
                    f"Series {series_id} not found in series dictionary. Skipping."
                )
                continue
            series_class = self.series_dict[series_id]
            if (
                overwrite
                or series_id not in self.all
                or not isinstance(self.all[series_id], series_class)
            ):
                self.all[series_id] = series_class(self.DATA_FOLDER, series_id)
            else:
                logger.debug(f"Series {series_id} already loaded, skipping.")


class DMIExam(ExamBase):
    def __init__(
        self,
        BASE_FOLDER: str | Path,
        DATA_FOLDER: str | Path | None = None,
        OUTPUT_FOLDER: str | Path | None = None,
        ANATOMICAL_series: str | int | None = None,
    ) -> None:
        super().__init__(BASE_FOLDER, DATA_FOLDER, OUTPUT_FOLDER)

        if isinstance(ANATOMICAL_series, int):
            self.ANATOMICAL_series = ANATOMICAL_series
        else:
            if ANATOMICAL_series is not None:
                identifier = ANATOMICAL_series.strip().lower()
            else:
                identifier = "t1_bravo"
            self.ANATOMICAL_series = (
                self.exam_overview[
                    self.exam_overview["Folder Name"].str.contains(
                        identifier, na=False, case=False
                    )
                ]
                .index.to_numpy()
                .astype(int)[0]
            )

        self.MRS_series = (
            self.exam_overview[
                self.exam_overview["Folder Name"].str.contains("MRS_unloc", na=False)
            ]
            .index.to_numpy()
            .astype(int)
        )
        # By default append all series where Has Dicom in the exam overview table is False
        no_DICOM_series = (
            self.exam_overview[~self.exam_overview["Has DICOM"]]
            .index.to_numpy()
            .astype(int)
        )
        self.MRS_series = np.union1d(self.MRS_series, no_DICOM_series)

        self.MRSI_series = (
            self.exam_overview[
                self.exam_overview["Folder Name"].str.contains("MRSI_pseudo", na=False)
            ]
            .index.to_numpy()
            .astype(int)
        )

        self.anatomical: GESeries.MRISeries
        self.all_MRS: list[GESeries.MRSSeries] = []
        self.all_MRSI: list[GESeries.MRSISeries] = []

    def load_all_series(self, overwrite: bool = False) -> None:
        super().load_all(overwrite=overwrite)

        # Iterate through all series and assign to appropriate lists
        # To make sure that we do not have duplicates, reset all lists first
        self.washin = None  # type: ignore
        self.all_MRS = []
        self.all_MRSI = []

        for series_obj in self.all.values():
            # Order is important as MRSWashinSeries is a subclass of MRSSeries
            # and all raw classes are subclasses of MRISeries
            if isinstance(series_obj, GESeries.MRSSeries):
                self.all_MRS.append(series_obj)
            elif isinstance(series_obj, GESeries.MRSISeries):
                self.all_MRSI.append(series_obj)
            elif isinstance(series_obj, GESeries.MRISeries):
                if not self.ANATOMICAL_series == series_obj.SERIES_ID:
                    continue
                if hasattr(self, "anatomical") and self.anatomical is not None:
                    logger.warning(
                        f"Multiple anatomical datasets detected. "
                        f"Previous: Series ID {self.anatomical.SERIES_ID}, "
                        f"Name '{self.exam_overview.at[self.anatomical.SERIES_ID, 'Folder Name']}'. "
                        f"New: Series ID {series_obj.SERIES_ID}, "
                        f"Name '{self.exam_overview.at[series_obj.SERIES_ID, 'Folder Name']}'."
                    )
                self.anatomical = series_obj


class MS_DMIExam(DMIExam):
    def __init__(
        self,
        BASE_FOLDER: str | Path,
        DATA_FOLDER: str | Path | None = None,
        OUTPUT_FOLDER: str | Path | None = None,
        ANATOMICAL_series: str | int | None = None,
    ) -> None:
        super().__init__(BASE_FOLDER, DATA_FOLDER, OUTPUT_FOLDER, ANATOMICAL_series)
        logger.info("MS DMI Exam initialized.")

        self.MRSI: GESeries.MRSISeries
        self.pre_MRS: GESeries.MRSSeries
        self.post_MRS: GESeries.MRSSeries
        self.normalisation_factor: np.complexfloating

    def load_all_series(self, overwrite: bool = False) -> None:
        super().load_all_series(overwrite)

        if len(self.all_MRSI) != 1:
            logger.warning(
                f"Expected exactly one MRSI series, but found {len(self.all_MRSI)}. "
                f"Please check the exam overview and series loading."
            )
        else:
            self.MRSI = self.all_MRSI[0]

        mrs_candidates = self.all_MRS
        if len(mrs_candidates) > 2 and hasattr(self, "MRSI"):
            mrsi_id = self.MRSI.SERIES_ID
            before = [s for s in mrs_candidates if s.SERIES_ID < mrsi_id]
            after = [s for s in mrs_candidates if s.SERIES_ID > mrsi_id]
            if before and after:
                mrs_candidates = [
                    max(before, key=lambda s: s.SERIES_ID),
                    min(after, key=lambda s: s.SERIES_ID),
                ]
                logger.info(
                    "More than two MRS series found; selected series "
                    f"directly before ({mrs_candidates[0].SERIES_ID}) and after "
                    f"({mrs_candidates[1].SERIES_ID}) the MRSI series "
                    f"({mrsi_id})."
                )

        if len(mrs_candidates) != 2:
            logger.warning(
                f"Expected exactly two MRS series, but found {len(mrs_candidates)}. "
            )
        else:
            self.pre_MRS = mrs_candidates[0]
            self.post_MRS = mrs_candidates[1]
            self.normalisation_factor = (
                self.pre_MRS.avg_fid.__abs__().max().values.item() / 10
            )


class DMIinjExam(DMIExam):
    def __init__(
        self,
        BASE_FOLDER: str | Path,
        DATA_FOLDER: str | Path | None = None,
        OUTPUT_FOLDER: str | Path | None = None,
        ANATOMICAL_series: str | int | None = None,
    ) -> None:
        super().__init__(BASE_FOLDER, DATA_FOLDER, OUTPUT_FOLDER, ANATOMICAL_series)

        self.WASHIN_series = (
            self.exam_overview[
                self.exam_overview["Folder Name"].str.contains("MRS_washin", na=False)
            ]
            .index.to_numpy()
            .astype(int)[0]
        )

        # Define pre and post series based on wash-in series
        self.pre_MRS_series, self.post_MRS_series = self._split_series(
            self.MRS_series, self.WASHIN_series, "MRS"
        )
        self.pre_MRSI_series, self.post_MRSI_series = self._split_series(
            self.MRSI_series, self.WASHIN_series, "MRSI"
        )

        logger.info("DMI Intravenous Injection Exam initialized.")

        self.washin: GESeries.MRSWashinSeries
        self.pre_MRS: GESeries.MRSSeries
        self.post_MRS: list[GESeries.MRSSeries] = []
        self.pre_MRSI: GESeries.MRSISeries
        self.post_MRSI: list[GESeries.MRSISeries] = []
        self.last_MRS: GESeries.MRSSeries
        self.last_MRSI: GESeries.MRSISeries

        self.normalisation_factor: np.complexfloating = np.complex64(1.0)
        self.injection_time: datetime

    def load_all_series(self, overwrite: bool = False) -> None:
        super().load_all_series(overwrite=overwrite)

        for series_obj in self.all.values():
            if isinstance(series_obj, GESeries.MRSWashinSeries):
                self.washin = series_obj

        logger.debug("Splitting all series into pre- and post-injection series.")
        series_mapping = {
            "pre_MRS": self.pre_MRS_series,
            "post_MRS": self.post_MRS_series,
            "pre_MRSI": self.pre_MRSI_series,
            "post_MRSI": self.post_MRSI_series,
        }

        for attr_name, series in series_mapping.items():
            # If pre in the name than attr name is singular
            if attr_name.startswith("pre_"):
                pre = True
            else:
                pre = False

            # Convert to numpy array if it is a single integer for consistent processing
            series = np.atleast_1d(series)

            if series.size == 0 or np.all(series < 0):
                logger.debug(f"No series found for {attr_name}, skipping.")
                objs = []
            elif series.size == 1:
                logger.debug(f"Single series found for {attr_name}: {series.item()}.")
                if pre:
                    objs = self.all[series.item()]
                else:
                    objs = [self.all[series.item()]]
            else:
                logger.debug(f"Multiple series found for {attr_name}: {series}.")
                objs = list(itemgetter(*series)(self.all))
            setattr(self, attr_name, objs)

        # Set normalisation factor based on pre-injection MRS series if available
        self.normalisation_factor = np.max(self.pre_MRS.avg_fid)
        self.injection_time = self.washin.scan_datetime
        if len(self.post_MRS) > 0:
            self.last_MRS = self.post_MRS[-1]
        else:
            logger.warning("No post-injection MRS series found. Double check exam.")
        if len(self.post_MRSI) > 0:
            self.last_MRSI = self.post_MRSI[-1]
        else:
            logger.warning("No post-injection MRSI series found. Double check exam.")

    def _split_series(
        self, series: npt.NDArray, reference_series: int, series_type: str
    ) -> tuple[int, npt.NDArray]:
        logger.debug(
            f"Splitting {series_type} series into pre- and post-injection based on "
            f"reference series {reference_series} for series list: {series}."
        )
        pre_series_idx = np.where(series < reference_series)[0]

        if pre_series_idx.size == 0:
            logger.debug(
                f"No pre-{series_type} series found before the reference series."
            )
            pre_series = int(-1)
        elif pre_series_idx.size == 1:
            pre_series = int(series[pre_series_idx].item())
        else:
            logger.debug(
                f"Multiple pre-{series_type} series found before the reference series. "
                f"Selecting the last one as pre-injection series."
            )
            last_pre_idx = pre_series_idx[-1]
            pre_series = int(series[last_pre_idx])

        post_series_idx = np.where(series > reference_series)[0]
        if post_series_idx.size == 0:
            logger.warning(
                f"No post-{series_type} series found after the reference series."
            )
            post_series = np.array([], dtype=int)
        else:
            post_series = series[post_series_idx]

        return pre_series, post_series

    def plot_MRS_over_time(
        self, phase_params: dict | None = None, plot_params: dict | None = None
    ) -> None:
        plot_kwargs = plot_params if plot_params is not None else {}
        phase_kwargs = phase_params if phase_params is not None else {}

        inj_start = self.washin.scan_datetime
        inj_start_str = inj_start.strftime("%H:%M")
        title = f"Phased spectra over time\n(Injection at {inj_start_str})"

        labels = []
        phased_spectra = []
        for mrs_obj in self.all_MRS:
            # phase_avg_spec returns a single chemical_shift-indexed DataArray
            phased_spec = mrs_obj.phase_avg_spec(**phase_kwargs)
            phased_spectra.append(
                phased_spec.assign_coords(Spectra_ID=mrs_obj.SERIES_ID)
            )
            # Set labels
            scan_time_str = mrs_obj.scan_time.strftime("%H:%M")
            time_delta_min = (mrs_obj.scan_datetime - inj_start).total_seconds() / 60
            labels.append(f"{scan_time_str}\n({time_delta_min:.0f} min)")

        combined = xr.concat(phased_spectra, dim="Spectra_ID")
        melted_spectra_df = (
            combined.rename("Intensity")
            .to_dataframe()
            .reset_index()
            .rename(columns={"chemical_shift": "ppm"})
            .sort_values("Spectra_ID", ascending=True)
        )

        plotting.spectra.plot_spectra_over_time(
            melted_spectra_df, labels, title=title, **plot_kwargs
        )
