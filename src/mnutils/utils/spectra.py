import numpy as np
import numpy.typing as npt
from loguru import logger


def calculate_hz_axis(
    spectral_width: float | None = None,
    npts: int | None = None,
    header: dict | None = None,
) -> npt.NDArray[np.float64]:
    _sw = spectral_width
    _npts = npts

    if header is not None:
        rdb_hdr = header.get("rdb_hdr", {})
        if isinstance(rdb_hdr, dict):
            if _sw is None:
                _sw = rdb_hdr.get("spectral_width", None)
            if _npts is None:
                _npts = rdb_hdr.get("user1", None)

    if _sw is None:
        raise ValueError(
            "Spectral width information is required to calculate frequency axis. Please provide spectral width or ensure it is present in the header."
        )
    if _npts is None:
        raise ValueError(
            "Number of points information is required to calculate frequency axis. Please provide number of points or ensure it is present in the header."
        )

    return np.linspace(-_sw / 2, _sw / 2, int(_npts))


def calculate_ppm_axis(
    spectral_width: float | None = None,
    frequency: float | None = None,
    carrier_ppm: float | None = None,
    nucleus: int | str | None = None,
    npts: int | None = None,
    header: dict | None = None,
) -> npt.NDArray[np.float64]:
    hz_axis = calculate_hz_axis(
        spectral_width=spectral_width,
        npts=npts,
        header=header,
    )

    _freq = None
    _carrier_ppm = None
    _nuc = None

    if frequency is not None:
        _freq = frequency

    if header is not None:
        rdb_hdr = header.get("rdb_hdr", {})
        if isinstance(rdb_hdr, dict):
            if _freq is None:
                _freq = rdb_hdr.get("ps_mps_freq", None)
                if _freq is not None:
                    _freq = np.double(_freq / 1e7)  # Convert to MHz
        image_hdr = header.get("image", {})
        if isinstance(image_hdr, dict):
            _nuc = image_hdr.get("specnuc", None)

    if _freq is None:
        raise ValueError(
            "Frequency information is required to calculate ppm axis. Please provide frequency or ensure it is present in the header."
        )

    if nucleus is not None:
        # Convert nucleus to integer if provided as string (e.g., '1H' -> 1) by extracting 1 to 3 digits from the string
        if isinstance(nucleus, str):
            import re

            match = re.search(r"\d{1,3}", nucleus)
            if match:
                _nuc = int(match.group(0))
            else:
                logger.warning(
                    f"Could not parse nucleus from string '{nucleus}'. Defaulting to 1H. This will offset ppm by 4.68!"
                )
                _nuc = 1
        else:
            _nuc = nucleus

    if _nuc is not None and carrier_ppm is not None:
        logger.warning(
            "Both nucleus and carrier_ppm provided. Carrier ppm will be used for ppm axis calculation."
        )
        _carrier_ppm = carrier_ppm
    elif carrier_ppm is not None:
        _carrier_ppm = carrier_ppm
    elif _nuc is not None:
        _carrier_ppm = 4.68 if _nuc in [1, 2] else 0.0
    else:
        logger.warning(
            "Neither nucleus nor carrier_ppm provided. Defaulting to 1H with carrier ppm of 4.68. This will offset ppm by 4.68!"
        )
        _carrier_ppm = 4.68

    ppm = (hz_axis / _freq) + _carrier_ppm
    return ppm.astype(np.float64)
