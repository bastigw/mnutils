import os
import re
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from mnutils.testing import build_fake_exam
from mnutils.utils import data_loaders

# load_mat_file is covered in docs/basics/mat_files.md (docs pages are the tests
# now, see uv run test). load_raw_fids stays here: it calls into MATLAB via
# mnutils.matlab, which this environment doesn't have -- see CLAUDE.md's
# "Gotchas" section and .claude/skills/docs-page/SKILL.md.


@pytest.fixture
def dataset_folder():
    """Return the root of the brain MRS/MRSI synthetic exam (see mnutils.testing)."""
    return build_fake_exam("brain_mrs_mrsi_exam")


@pytest.mark.skip(reason="Matlabengine not supported anymore")
def test_load_raw_fids_creates_correct_files(dataset_folder):
    """Test that load_raw_fids returns correctly-shaped complex FID arrays from the h5 cache."""
    data_folder = os.path.join(dataset_folder, "data")
    # Series 6/7 (MRS_unloc/MRS_washin) have real, `.mat`-derived FIDs shaped
    # (averages, npts); the rest are random-noise fixtures shaped (npts, ntime).
    shape_expected = {
        6: (64, 2048),
        7: (250, 2048),
        9: (700, 150),
        11: (700, 150),
        12: (1678, 150),
    }
    for series, shape in shape_expected.items():
        data, fid_h5_file = data_loaders.load_raw_fids(data_folder, series, force_override=False)
        assert isinstance(data, np.ndarray)
        assert np.iscomplexobj(data)
        assert data.shape == shape
        if series in (6, 7):  # MRS_unloc/MRS_washin -- real simulated data, not repeated noise
            assert not np.allclose(data[0], data[1])
        assert os.path.isfile(fid_h5_file)
        # `fid_h5_file` is part of the fixture `dataset_folder` built (and, since
        # `build_fake_exam`'s cache is now machine-wide, shared with every other test and
        # notebook that also asked for this dataset) -- not a byproduct of this test to remove.


def _add_double(group: h5py.Group, name: str, value: float) -> None:
    group.create_dataset(name, data=np.array([[value]])).attrs["MATLAB_class"] = np.bytes_(
        b"double"
    )


def _write_v73_archive(path: Path) -> None:
    """A v7.3 (HDF5) archive whose header has many datasets, only some of them allowlisted."""
    with h5py.File(path, "w") as hf:
        h = hf.create_group("h")
        _add_double(h, "rdb_hdr/te", 30000.0)
        _add_double(h, "rdb_hdr/spectral_width", 5000.0)
        _add_double(h, "image/tr", 2.0e6)
        _add_double(h, "exam/ex_no", 408.0)
        for i in range(50):
            _add_double(h, f"rdb_hdr/unused_{i}", float(i))
        desc = h.create_dataset(
            "series/se_desc",
            data=np.array([[ord(c)] for c in "MRS washin"], dtype=np.uint16),
        )
        desc.attrs["MATLAB_class"] = np.bytes_(b"char")
        _add_double(hf, "bb", 1.0)
        _add_double(hf, "hz", 3.0)  # a name starting with "h": mat73 substring-matches it to "h"


def test_read_mat_header_v73_matches_load_mat_file(tmp_path):
    """The h5py fast path returns exactly what the full mat73 load does, for allowlisted fields."""
    path = tmp_path / "ScanArchive_v73.mat"
    _write_v73_archive(path)

    header = data_loaders.read_mat_header(path)
    full = data_loaders.load_mat_file(path)["h"]

    assert header["series"]["se_desc"] == full["series"]["se_desc"] == "MRS washin"
    assert header["rdb_hdr"]["spectral_width"] == full["rdb_hdr"]["spectral_width"]
    assert header["rdb_hdr"]["te"] == full["rdb_hdr"]["te"]
    # Only allowlisted fields come back; absent ones (e.g. image/nex) are simply left out.
    assert "unused_0" not in header["rdb_hdr"]
    assert "nex" not in header["image"]


def test_read_mat_header_plain_matches_v73_shape(tmp_path):
    """A v5/v7 archive yields the same pruned nested dict, loading only ``h``."""
    path = tmp_path / "ScanArchive_v5.mat"
    scipy.io.savemat(
        path,
        {
            "h": {"rdb_hdr": {"te": 30000.0, "other": 1.0}, "series": {"se_desc": "MRS washin"}},
            "bb": np.zeros((4, 4)),
        },
    )
    assert data_loaders.read_mat_header(path) == {
        "rdb_hdr": {"te": 30000.0},
        "series": {"se_desc": "MRS washin"},
    }


def test_read_mat_header_custom_fields(tmp_path):
    """The overview asks for ``series/se_desc`` alone."""
    path = tmp_path / "ScanArchive_v73.mat"
    _write_v73_archive(path)
    assert data_loaders.read_mat_header(path, fields=["series/se_desc"]) == {
        "series": {"se_desc": "MRS washin"}
    }


def test_read_mat_header_validates_path(tmp_path):
    """Same input checks as `load_mat_file`."""
    with pytest.raises(FileNotFoundError):
        data_loaders.read_mat_header(tmp_path / "missing.mat")
    wrong = tmp_path / "header.txt"
    wrong.touch()
    with pytest.raises(ValueError):
        data_loaders.read_mat_header(wrong)


@pytest.mark.parametrize("writer", ["v73", "v5"])
def test_load_mat_file_variable_names(tmp_path, writer):
    """``variable_names`` skips the others, even ``h`` when ``hz`` is asked for (mat73 trap)."""
    path = tmp_path / "x.mat"
    if writer == "v73":
        _write_v73_archive(path)
    else:
        scipy.io.savemat(path, {"h": {"a": 1.0}, "bb": np.zeros((2, 2)), "hz": 3.0})
    loaded = data_loaders.load_mat_file(path, variable_names=["bb", "hz"])
    assert {"bb", "hz"} <= set(loaded)
    assert "h" not in loaded


@pytest.mark.parametrize("writer", ["v73", "v5"])
def test_load_mat_file_ignores_absent_variable_names(tmp_path, writer, capsys):
    """Asking only for absent variables loads nothing -- and no mat73 tree dump on stdout."""
    path = tmp_path / "x.mat"
    if writer == "v73":
        _write_v73_archive(path)
    else:
        scipy.io.savemat(path, {"h": {"a": 1.0}})
    loaded = data_loaders.load_mat_file(path, variable_names=["spec"])
    assert [key for key in loaded if not key.startswith("__")] == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("writer", ["v73", "v5"])
def test_list_mat_variables(tmp_path, writer):
    """Top-level names come back without loading anything."""
    path = tmp_path / "x.mat"
    if writer == "v73":
        _write_v73_archive(path)
    else:
        scipy.io.savemat(path, {"h": {"a": 1.0}, "bb": np.zeros((2, 2)), "hz": 3.0})
    assert set(data_loaders.list_mat_variables(path)) == {"h", "bb", "hz"}


def test_header_fields_cover_every_header_read():
    """Every header path GESeries reads, plus the spectral_width pyAMARES needs, is allowlisted."""
    source = (Path(data_loaders.__file__).parents[1] / "GESeries.py").read_text()
    used = {
        "/".join(re.findall(r'"([^"]+)"', match))
        for match in re.findall(r"_get_header_value\(\[([^\]]+)\]\)", source)
    }
    assert used, "found no _get_header_value calls -- has the accessor been renamed?"
    assert used | {"rdb_hdr/spectral_width"} == set(data_loaders.HEADER_FIELDS)


def test_overview_reads_description_from_v73_header(tmp_path):
    """``_read_series_description`` works on a v7.3 archive without loading the full file."""
    from mnutils.utils import file_helpers

    path = tmp_path / "ScanArchive_v73.mat"
    _write_v73_archive(path)
    assert file_helpers._read_series_description(path) == "MRS washin"
