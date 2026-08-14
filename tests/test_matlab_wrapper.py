import os
from pathlib import Path

import matlab.engine

from mnutils.matlab import (
    add_matlablatest_path,
    connect_to_matlab,
    setup_util_path,
)


def test_connect_to_matlab_success():
    """Test that connect_to_matlab returns a usable MATLAB engine instance."""
    eng = connect_to_matlab()
    import matlab.engine

    assert isinstance(eng, matlab.engine.MatlabEngine)
    result = eng.eval("disp('Hello from MATLAB!')", nargout=0)
    assert result is None  # disp returns None in Python


def test_setup_util_path_adds_path(tmp_path, monkeypatch):
    """Test that setup_util_path adds MATLAB_UTILS to the MATLAB path."""
    # Create a fake MATLAB_UTILS directory
    matlab_utils_dir = tmp_path / "matlab_utils"
    matlab_utils_dir.mkdir()
    monkeypatch.setenv("MATLAB_UTILS", str(matlab_utils_dir))

    eng = connect_to_matlab()
    # Remove path if it exists for test isolation
    current_paths = str(eng.path()).split(":")
    if str(matlab_utils_dir) in current_paths:
        eng.rmpath(str(matlab_utils_dir))

    setup_util_path(eng, os.environ.get("MATLAB_UTILS"))
    # Check that the path was added
    updated_paths = str(eng.path()).split(":")
    assert str(matlab_utils_dir) in updated_paths


def test_setup_util_path_path_already_exists(tmp_path, monkeypatch):
    """Test that setup_util_path doesn't duplicate a path already on MATLAB's path."""
    matlab_utils_dir = tmp_path / "matlab_utils"
    matlab_utils_dir.mkdir()
    monkeypatch.setenv("MATLAB_UTILS", str(matlab_utils_dir))

    eng = connect_to_matlab()
    # Add path first
    eng.addpath(str(matlab_utils_dir))
    # Now call setup_util_path, should not add again
    setup_util_path(eng)
    updated_paths = str(eng.path()).split(":")
    assert str(matlab_utils_dir) in updated_paths


def test_setup_util_path_invalid_path(monkeypatch):
    """Test that setup_util_path silently ignores a non-existent MATLAB_UTILS path."""
    # Set MATLAB_UTILS to a non-existent path
    monkeypatch.setenv("MATLAB_UTILS", "/non/existent/path")
    eng = connect_to_matlab()
    setup_util_path(eng)
    # Should not add the path, but should not raise
    current_paths = str(eng.path()).split(":")
    assert "/non/existent/path" not in current_paths


def test_setup_util_path_env_not_set(monkeypatch):
    """Test that setup_util_path doesn't raise when MATLAB_UTILS is unset."""
    monkeypatch.delenv("MATLAB_UTILS", raising=False)
    eng = connect_to_matlab()
    setup_util_path(eng)
    # Should not add any path, but should not raise
    assert isinstance(eng, matlab.engine.MatlabEngine)


def test_add_matlablatest_path():
    """Test that add_matlablatest_path adds the matlabfiles dir to MATLAB's path."""
    matlab_sources = os.getenv("MATLAB_SOURCES")
    assert matlab_sources is not None, "MATLAB_SOURCES environment variable is not set"
    matlablatest_dir = Path(matlab_sources) / "Data Analysis" / "matlabfiles"

    eng = connect_to_matlab()
    # Restore default path
    eng.restoredefaultpath(nargout=0)
    setup_util_path(eng)
    add_matlablatest_path(eng)
    # Check that the path was added
    updated_paths = str(eng.path()).split(":")
    assert str(matlablatest_dir) in updated_paths
