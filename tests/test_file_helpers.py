import json
import re
from pathlib import Path

import pytest

from mnutils.utils import file_helpers

# On Windows CI the pytest cwd sits on D: while tempfile hands out paths on C:.
# Path.relative_to(walk_up=True) raises ValueError across anchors, which used to
# blow up a debug log line and fail the whole run (see the v1.2.0 tag build).
# Faking a foreign-anchor cwd reproduces that on any platform.
_FOREIGN_CWD = "Z:\\somewhere\\else"


def test_relative_to_cwd_shortens_paths_under_cwd(tmp_path, monkeypatch):
    """A path below the cwd is reported relative to it."""
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "sub" / "file.h5"

    assert file_helpers._relative_to_cwd(target) == Path("sub/file.h5")


def test_relative_to_cwd_falls_back_on_foreign_anchor(tmp_path, monkeypatch):
    """A path with no common anchor is returned unchanged instead of raising."""
    monkeypatch.setattr(file_helpers.os, "getcwd", lambda: _FOREIGN_CWD)
    target = tmp_path / "file.h5"

    assert file_helpers._relative_to_cwd(target) == target


def test_move_files_with_glob_survives_foreign_anchor(tmp_path, monkeypatch):
    """Moving files still succeeds when the cwd shares no anchor with the target."""
    monkeypatch.setattr(file_helpers.os, "getcwd", lambda: _FOREIGN_CWD)
    (tmp_path / "result_a.h5").touch()
    (tmp_path / "result_b.h5").touch()

    file_helpers.move_files_with_glob(tmp_path, "*.h5")

    # move_files_with_glob prepends the creation date by default.
    moved = sorted(re.sub(r"^\d{8}_\d{6}_", "", p.name) for p in (tmp_path / "old").glob("*.h5"))
    assert moved == ["result_a.h5", "result_b.h5"]


def _write_series(
    folder: Path,
    name: str,
    series_id: int,
    description: str,
    image_type: str | None = None,
    orientation: list[float] | None = None,
) -> Path:
    """A NIfTI + dcm2niix-style sidecar pair; the NIfTI content is irrelevant to discovery."""
    folder.mkdir(parents=True, exist_ok=True)
    nii = folder / f"{name}.nii.gz"
    nii.touch()
    sidecar = {"SeriesNumber": series_id, "SeriesDescription": description}
    if image_type is not None:
        sidecar["ImageType"] = ["ORIGINAL", "PRIMARY", "OTHER", image_type]
    if orientation is not None:
        sidecar["ImageOrientationPatientDICOM"] = orientation
    (folder / f"{name}.json").write_text(json.dumps(sidecar))
    return nii


def _warnings_from(action) -> list[str]:
    """The WARNING-level log messages emitted while `action` runs."""
    from loguru import logger

    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        action()
    finally:
        logger.remove(handler)
    return messages


def test_index_ignores_folder_names(tmp_path):
    """A series is identified by its sidecar's SeriesNumber, whatever its folder is called."""
    _write_series(tmp_path / "anything at all", "scan", 7, "3D Ax T1 BRAVO")

    index = file_helpers.index_nifti_series(tmp_path)

    assert list(index) == [7]
    assert index[7].description == "3D Ax T1 BRAVO"
    assert file_helpers.get_niftis_from_series(tmp_path, 7) == index[7].nii_path


def test_index_skips_niftis_without_sidecar(tmp_path):
    """Derived files (`_bet`) have no sidecar and are never picked as the main image."""
    main = _write_series(tmp_path / "s2", "t1", 2, "T1")
    (tmp_path / "s2" / "t1_bet.nii.gz").touch()

    assert file_helpers.get_niftis_from_series(tmp_path, 2) == main


def test_index_picks_lowest_suffix_and_warns(tmp_path):
    """Several sidecar pairs per series: the shortest, then lowest, name wins."""
    from loguru import logger

    messages: list[str] = []
    handler = logger.add(messages.append, level="WARNING")
    try:
        _write_series(tmp_path / "s8", "FGRE_i00006", 8, "FGRE")
        first = _write_series(tmp_path / "s8", "FGRE_i00001", 8, "FGRE")
        _write_series(tmp_path / "s8", "FGRE_i00011", 8, "FGRE")

        assert file_helpers.get_niftis_from_series(tmp_path, 8) == first
    finally:
        logger.remove(handler)
    assert any("FGRE_i00001" in message for message in messages)


def test_magnitude_beats_derived_map_without_warning(tmp_path):
    """A field map next to its magnitude image is expected, not an ambiguity."""
    _write_series(tmp_path / "s3", "HOS_fieldmaphz", 3, "HOS", "FIELDMAPHZ")
    magnitude = _write_series(tmp_path / "s3", "HOS", 3, "HOS", "MAGNITUDE")

    messages = _warnings_from(lambda: file_helpers.index_nifti_series(tmp_path))
    series = file_helpers.index_nifti_series(tmp_path)[3]

    assert messages == []
    assert series.nii_path == magnitude
    assert series.kind == "derived"
    assert [p.name for p in series.extras] == ["HOS_fieldmaphz.nii.gz"]


def test_localizer_planes_pick_axial_without_warning(tmp_path):
    """One file per slice orientation is a multi-plane series; the axial plane is main."""
    sag = [0, 1, 0, 0, 0, -1]
    ax = [1, 0, 0, 0, 1, 0]
    cor = [1, 0, 0, 0, 0, -1]
    _write_series(tmp_path / "s1", "loc_i00001", 1, "FGRE", "MAGNITUDE", sag)
    axial = _write_series(tmp_path / "s1", "loc_i00006", 1, "FGRE", "MAGNITUDE", ax)
    _write_series(tmp_path / "s1", "loc_i00011", 1, "FGRE", "MAGNITUDE", cor)

    messages = _warnings_from(lambda: file_helpers.index_nifti_series(tmp_path))
    series = file_helpers.index_nifti_series(tmp_path)[1]

    assert messages == []
    assert series.nii_path == axial
    assert series.kind == "multiplane"
    assert len(series.extras) == 2


def test_index_is_cached_until_folder_changes(tmp_path):
    """The folder is scanned once; a new series invalidates the cached index."""
    _write_series(tmp_path / "s8", "a_i00001", 8, "X")
    _write_series(tmp_path / "s8", "a_i00006", 8, "X")

    first = _warnings_from(lambda: file_helpers.index_nifti_series(tmp_path))
    again = _warnings_from(lambda: file_helpers.index_nifti_series(tmp_path))
    assert len(first) == 1
    assert again == []

    _write_series(tmp_path / "s9", "b", 9, "Y")
    assert list(file_helpers.index_nifti_series(tmp_path)) == [8, 9]


def test_missing_series_error_names_the_dcm2niix_command(tmp_path):
    """No sidecar pair: the error says how to convert, because MNUtils never does."""
    _write_series(tmp_path / "s2", "t1", 2, "T1")

    with pytest.raises(FileNotFoundError, match="dcm2niix"):
        file_helpers.get_niftis_from_series(tmp_path, 3)
    with pytest.raises(FileNotFoundError, match="dcm2niix"):
        file_helpers.get_nifti_data(tmp_path / "empty")


def test_get_mat_archives_prefers_plain_then_shortest_suffix(tmp_path):
    """The raw archive beats derived files; with only derived files, the shorter suffix wins."""
    series = tmp_path / "Series13"
    series.mkdir()
    for name in ("ScanArchive_A_snr_map.mat", "ScanArchive_A_csi.mat"):
        (series / name).touch()
    assert file_helpers.get_mat_archives(series)[0].name == "ScanArchive_A_csi.mat"

    (series / "ScanArchive_A.mat").touch()
    assert file_helpers.get_mat_archives(series)[0].name == "ScanArchive_A.mat"
