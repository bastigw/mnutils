import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import pytest
from loguru import logger

from mnutils import GESeries
from mnutils.utils import file_helpers, images

logger.remove()
logger.add(sys.stdout, level="DEBUG")


@pytest.fixture
def test_folder():
    """Return the directory containing this test file."""
    test_path = os.path.realpath(__file__)
    folder = os.path.dirname(test_path)
    return folder


@pytest.fixture
def dataset_folder(test_folder):
    """Return the path to the test datasets directory."""
    return os.path.join(test_folder, "datasets")


def test_hevo23_bet(dataset_folder):
    """Test that extract_brain produces the expected skull-stripped nifti output file."""
    base_folder = Path(dataset_folder) / "HeVo-23"
    data_folder = base_folder / "data"
    series = 2
    nifti_data = file_helpers.get_niftis_from_series(
        data_folder, series, convert_dicoms=True
    )
    logger.info(f"Extracting brain from {nifti_data}")
    bet_file = images.extract_brain(nifti_data, save_mask=True)
    # Check if the output file exists
    assert bet_file.exists()
    # Check if the output file has the expected name
    expected_name = "2_3D_Ax_T1_BRAVO_BODY_bet.nii.gz"
    assert bet_file.name == expected_name
    # Load nifti and display it
    bet_nii = GESeries.NiiBase(bet_file)
    bet_nii.display()
    plt.show()
