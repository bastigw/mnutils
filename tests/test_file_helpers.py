import re
from pathlib import Path

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


def _exam_with_series(folder: Path, series_ids: list[int]) -> Path:
    """Create one DICOM folder per series ID, named the way an exported exam is."""
    for series_id in series_ids:
        (folder / f"{series_id:03d}_series").mkdir()
    return folder


# GE numbers protocol steps densely from 1, but reformats/resaves get their own
# far-away numbers (500/501, 650/651, 40003). Bounding the gap check by the
# largest series present therefore reported every unused integer below 100 as
# missing whenever such a series existed -- see issue #34, twice.
def test_missing_series_ignores_sparse_reformat_ids(tmp_path):
    """Series >= 100 are neither flagged as gaps nor widen the checked range."""
    _exam_with_series(tmp_path, [*range(1, 15), 113, 200, 300, 500, 501, 650, 651, 40003])

    found, missing = file_helpers.get_all_dicom_series_ids(tmp_path)

    assert missing == []
    assert found == [*range(1, 15), 113, 200, 300, 500, 501, 650, 651, 40003]


def test_missing_series_still_reports_gaps_in_the_dense_range(tmp_path):
    """A hole between protocol steps is real and stays reported, sparse IDs or not."""
    _exam_with_series(tmp_path, [1, 2, 4, 113])

    assert file_helpers.get_all_dicom_series_ids(tmp_path)[1] == [3]


def test_missing_series_checks_nothing_without_dense_series(tmp_path):
    """With every series >= 100 there is no dense range to check."""
    _exam_with_series(tmp_path, [113, 500])

    assert file_helpers.get_all_dicom_series_ids(tmp_path)[1] == []


def test_missing_series_handles_an_empty_folder(tmp_path):
    """An exam folder with no DICOM series returns empty lists instead of raising."""
    found, missing = file_helpers.get_all_dicom_series_ids(tmp_path)

    assert (found, missing) == ([], [])
