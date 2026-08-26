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
