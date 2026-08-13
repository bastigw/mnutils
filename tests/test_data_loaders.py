import os

import numpy as np
import pytest

from mnutils.utils import data_loaders

# load_mat_file is covered in docs/basics/mat_files.md (docs pages are the tests
# now, see uv run test). load_raw_fids stays here: it calls into MATLAB via
# mnutils.matlab, which this environment doesn't have -- see CLAUDE.md's
# "Gotchas" section and .claude/skills/docs-page/SKILL.md.


@pytest.fixture
def test_folder():
    test_path = os.path.realpath(__file__)
    folder = os.path.dirname(test_path)
    return folder


@pytest.fixture
def dataset_folder(test_folder):
    return os.path.join(test_folder, "datasets")


def test_load_raw_fids_creates_correct_files(dataset_folder):
    base_folder = os.path.join(dataset_folder, "HeVo-18")
    data_folder = os.path.join(base_folder, "data")
    npts_expected = {
        6: 64,
        9: 64,
        11: 64,
        12: 1678,
    }
    for series, npts in npts_expected.items():
        data, fid_h5_file = data_loaders.load_raw_fids(
            data_folder, series, force_override=False
        )
        assert isinstance(data, np.ndarray)
        assert np.iscomplexobj(data)
        assert data.shape[0] == npts
        assert data.shape[1] > 100  # Should have more than 100 points
        assert os.path.isfile(fid_h5_file)
        # Clean up created file
        os.remove(fid_h5_file)
