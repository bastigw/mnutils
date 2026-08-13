import nmrglue as ng
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


def phase_nmr_data(
    spectra: npt.NDArray[np.complexfloating],
    zero_order: float | None = None,
    first_order: float | None = None,
    autophase: bool = False,
    invert_if_negative: bool = True,
) -> npt.NDArray[np.floating]:
    """Phase NMR signals, such as FIDs or spectra.

    Parameters
    ----------
    spectra : ndarray of complexfloating
        The complex FIDs or spectra to phase.
    zero_order : float, optional
        Zero-order phase correction in degrees.
    first_order : float, optional
        First-order phase correction in degrees.
    autophase : bool, optional
        Whether to apply automatic phasing using ACME.
    invert_if_negative : bool, optional
        If `autophase`, invert the data if the sum is negative.

    Returns
    -------
    ndarray of floating
        The phased, real part of the NMR data.
    """

    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]

    phased_spec: npt.NDArray[np.floating]

    if zero_order is not None and first_order is not None:
        phased_spec = np.asarray(
            ng.proc_base.ps(spectra, zero_order, first_order), dtype=np.complex64
        ).real
    elif (zero_order is None) ^ (first_order is None):
        logger.warning(
            "Only one of zero_order or first_order phase provided. Skipping manual phasing."
        )
        phased_spec = spectra.real
    elif autophase:
        phased_complex, opt = ng.proc_autophase.autops(
            spectra,
            fn="acme",
            disp=False,
            return_phases=True,
        )
        logger.debug(
            f"Autophase applied with optimized phases: p0={opt[0]:.2f}, p1={opt[1]:.2f}"
        )
        phased_spec = np.asarray(
            phased_complex,
            dtype=np.complex64,
        ).real
        # Occasionally the autophase optimises such that the spectrum is inverted.
        if np.sum(phased_spec) < 0:
            if invert_if_negative:
                logger.debug(
                    "Autophase resulted in inverted spectrum. Inverting spectra."
                )
                phased_spec = -phased_spec
            else:
                logger.debug(
                    "Autophase resulted in inverted spectrum. Not inverting spectra as per user request."
                )
    else:
        logger.debug(
            "No phasing applied to average spectrum. Returning real part only."
        )
        phased_spec = spectra.real

    return phased_spec.astype(np.float64)


def process_and_trim_spectra(
    spectra: npt.NDArray[np.complexfloating],
    ppm: npt.NDArray[np.floating] = np.array([], dtype=np.float32),
    zero_order: float | None = None,
    first_order: float | None = None,
    autophase: bool = False,
    ppm_min: float = -np.inf,
    ppm_max: float = np.inf,
    invert_if_negative: bool = True,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Phase spectra and trim to a specific PPM range.

    Parameters
    ----------
    spectra : ndarray of complexfloating
        The complex spectra.
    ppm : ndarray of floating, optional
        The PPM axis corresponding to the spectra.
    zero_order : float, optional
        Zero-order phase correction.
    first_order : float, optional
        First-order phase correction.
    autophase : bool, optional
        Enable automatic phasing.
    ppm_min : float, optional
        Minimum PPM value to include.
    ppm_max : float, optional
        Maximum PPM value to include.
    invert_if_negative : bool, optional
        Invert spectrum if autophase results in a negative sum.

    Returns
    -------
    tuple[ndarray of floating, ndarray of floating]
        Phased and trimmed spectra (real part), and the trimmed ppm axis.
    """
    phased_spec = phase_nmr_data(
        spectra,
        zero_order=zero_order,
        first_order=first_order,
        autophase=autophase,
        invert_if_negative=invert_if_negative,
    )

    if ppm.size == 0:
        logger.debug("PPM axis not provided. Using full range for masking.")
        actual_points = phased_spec.shape[1]
        ppm_mask = np.ones(actual_points, dtype=bool)
        # Generate dummy ppm axis if none provided to satisfy return type
        ppm_to_return = np.arange(actual_points, dtype=np.float64)
    else:
        ppm_mask = (ppm >= ppm_min) & (ppm <= ppm_max)
        ppm_to_return = ppm

    return phased_spec[:, ppm_mask].astype(np.float64), ppm_to_return[ppm_mask].astype(
        np.float64
    )
