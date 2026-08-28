import os

import numpy as np
import pytest

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
