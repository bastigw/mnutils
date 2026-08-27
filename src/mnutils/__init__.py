"""Utilities for working with multi-nuclear MR data from GE scanners.

The public surface is re-exported here, so the common entry points are
reachable without knowing the internal module layout::

    import mnutils

    exam = mnutils.DMIExam("/data/exam")
    mnutils.plotting.images.display_images(exam.anatomical[0].data)

Package-wide defaults live in `mnutils.rcParams`, which works like
matplotlib's::

    mnutils.rcParams["image.cmap"] = "viridis"

`mnutils.matlab` is deliberately *not* imported here: it needs the optional
`matlabengine` dependency, so import it explicitly when you need it.
"""

from importlib.metadata import PackageNotFoundError, version

from . import GEExam, GESeries, fitting, plotting, testing, utils
from .GEExam import DMIExam, DMIinjExam, ExamBase, MS_DMIExam
from .GESeries import (
    MRISeries,
    MRSISeries,
    MRSSeries,
    MRSWashinSeries,
    NiiBase,
    RawMRISeries,
)
from .rcparams import RcParams, rc_context, rc_presets, rcdefaults, rcParams, rcParamsDefault

try:
    __version__ = version("mnutils")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = [
    # subpackages / modules
    "GEExam",
    "GESeries",
    "fitting",
    "plotting",
    "testing",
    "utils",
    # exam classes
    "ExamBase",
    "DMIExam",
    "MS_DMIExam",
    "DMIinjExam",
    # series classes
    "NiiBase",
    "MRISeries",
    "RawMRISeries",
    "MRSISeries",
    "MRSSeries",
    "MRSWashinSeries",
    # configuration
    "RcParams",
    "rcParams",
    "rcParamsDefault",
    "rc_context",
    "rc_presets",
    "rcdefaults",
    "__version__",
]
