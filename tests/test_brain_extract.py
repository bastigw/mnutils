import sys

import matplotlib
import pytest

pytest.importorskip("torch", reason="requires the optional 'bet' extra (torch + HD-BET)")

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # noqa: E402
from loguru import logger  # noqa: E402

from mnutils import GESeries  # noqa: E402
from mnutils.testing import build_fake_exam  # noqa: E402
from mnutils.utils import file_helpers, images  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="DEBUG")


@pytest.fixture
def dataset_folder():
    """Return the root of the brain-extraction synthetic exam (see mnutils.testing)."""
    return build_fake_exam("brain_extraction_exam")


@pytest.mark.skip(reason="Dataset not available in remote. Need to refactor!")
def test_hevo23_bet(dataset_folder):
    """Test that extract_brain produces the expected skull-stripped nifti output file."""
    data_folder = dataset_folder / "data"
    series = 2
    nifti_data = file_helpers.get_niftis_from_series(data_folder, series, convert_dicoms=True)
    logger.info(f"Extracting brain from {nifti_data}")
    bet_file = images.extract_brain(nifti_data, save_mask=True, device="cpu")
    # Check if the output file exists
    assert bet_file.exists()
    # Check if the output file has the expected name
    expected_name = "2_3D_Ax_T1_BRAVO_BODY_bet.nii.gz"
    assert bet_file.name == expected_name
    # Load nifti and display it
    bet_nii = GESeries.NiiBase(bet_file)
    bet_nii.display()
    plt.show()
