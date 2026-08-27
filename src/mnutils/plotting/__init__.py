"""Plotting for images and spectra, plus figure-saving helpers.

Importing this package applies the MNUtils matplotlib defaults (sans-serif
font stack, bold axis labels, constrained layout).
"""

import matplotlib.pyplot as plt

from . import images, saving, spectra
from .saving import contexts, save_current_figure, save_figure

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.constrained_layout.use"] = True

__all__ = [
    # modules
    "images",
    "saving",
    "spectra",
    # figure saving
    "contexts",
    "save_figure",
    "save_current_figure",
]
