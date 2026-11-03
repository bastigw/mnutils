import importlib
import subprocess
import sys

import pytest

SETUP_MODULE = "mnutils.matlab.setup_matlab"
# None in sys.modules makes the import machinery raise ModuleNotFoundError, which is what
# `import matlab.engine` does where matlabengine isn't installed.
BLOCK_MATLAB = "import sys; sys.modules['matlab'] = None; "


@pytest.fixture
def without_matlabengine(monkeypatch):
    """Make `import matlab.engine` fail inside this process, restoring sys.modules afterwards."""
    for name in [m for m in sys.modules if m == "matlab" or m.startswith("matlab.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matlab", None)
    for name in ("mnutils.matlab", SETUP_MODULE):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_importing_mnutils_does_not_need_matlab():
    """A bare `import mnutils` must not touch matlabengine, even when it is unimportable."""
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_MATLAB + "import mnutils"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_matlabengine_raises_actionable_error(without_matlabengine):
    """The error names matlabengine, points at the README, and chains the original failure."""
    with pytest.raises(ImportError, match="matlabengine") as excinfo:
        importlib.import_module("mnutils.matlab")
    message = str(excinfo.value)
    assert "MATLAB" in message and "README" in message
    assert isinstance(excinfo.value.__cause__, ImportError)
