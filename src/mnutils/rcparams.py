"""Package-wide defaults, in one validated mapping.

Every knob that used to live in a module-level ``DEFAULT_*`` dict is a key in
`rcParams`, the way matplotlib's own ``rcParams`` works, and for the same
reason: a default that is worth having is worth being able to change once at
the top of a notebook instead of at every call site.

    import mnutils

    mnutils.rcParams["image.cmap"] = "viridis"          # for the whole session
    with mnutils.rc_context({"grid.panel_height": "30rem"}):
        ...                                             # ...or just for a block
    mnutils.rcdefaults()                                # back to the built-ins

Three rules make this safe to rely on:

- **Keys are validated on assignment.** `_VALIDATORS` defines both the legal
  key set and what each key accepts, so a typo raises immediately (with a
  near-match suggestion) rather than being silently ignored until someone
  wonders why the colormap never changed.
- **Nothing reads a default at import time.** Functions take ``None`` and
  resolve it in the body via `resolve_rc`, so assigning to `rcParams` after
  ``import mnutils`` actually takes effect. Binding a default in a signature
  (``cmap=rcParams["image.cmap"]``) freezes it at import and is the one
  mistake this module exists to prevent.
- **An explicit argument always wins.** Precedence is argument > `rc_context`
  > `rcParams` > the built-in default in `_DEFAULTS`.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Callable, Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any

import numpy as np
from matplotlib import colormaps

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

# A CSS length as the grid widget will accept it. Deliberately a whitelist and
# not a free string: these values are interpolated into a `style` attribute on
# generated HTML, so anything that isn't a plain length has no business getting
# through.
_CSS_LENGTH_RE = re.compile(r"^\d+(\.\d+)?(px|rem|em|vh|vw|%)$")

_RESAMPLE_MODES = frozenset(
    {
        "constant",
        "nearest",
        "reflect",
        "mirror",
        "wrap",
        "grid-constant",
        "grid-mirror",
        "grid-wrap",
    }
)

# Single-letter anatomical axis codes, as nibabel's `orientations` uses them.
_AXIS_CODES = frozenset("LRAPIS")


def _validate_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"expected a bool, got {value!r}")
    return value


def _validate_str(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected a str, got {value!r}")
    return value


def _validate_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected a number, got {value!r}") from None


def _validate_positive_int(value: Any) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected an int, got {value!r}") from None
    if out < 1:
        raise ValueError(f"expected a positive int, got {value!r}")
    return out


def _validate_cmap(value: Any) -> str:
    name = _validate_str(value)
    if name not in colormaps:
        raise ValueError(f"unknown colormap {name!r}; see `matplotlib.colormaps`")
    return name


def _validate_css_length(value: Any) -> str:
    """Accept ``"30rem"``/``"400px"``/``"60vh"``, or a bare number read as rem."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):g}rem"
    text = _validate_str(value)
    if not _CSS_LENGTH_RE.match(text):
        raise ValueError(
            f"expected a CSS length such as '30rem', '400px' or '60vh' "
            f"(or a bare number, read as rem), got {value!r}"
        )
    return text


def _validate_ticker_steps(value: Any) -> list[int]:
    """`matplotlib.ticker.MaxNLocator` steps: increasing ints in [1, 10] starting at 1."""
    try:
        steps = [int(step) for step in value]
    except TypeError:
        raise ValueError(f"expected a sequence of ints, got {value!r}") from None
    if (
        not steps
        or steps[0] != 1
        or any(b <= a for a, b in zip(steps, steps[1:]))
        or steps[-1] > 10
    ):
        raise ValueError(
            f"expected increasing ints starting at 1 and ending at most 10, got {value!r}"
        )
    return steps


def _validate_ticker_bins(value: Any) -> int | str:
    if value == "auto":
        return "auto"
    return _validate_positive_int(value)


def _validate_ticks(value: Any) -> list[float]:
    """Explicit tick positions; the empty list means "no ticks"."""
    try:
        return [float(tick) for tick in value]
    except TypeError:
        raise ValueError(f"expected a sequence of numbers, got {value!r}") from None


def _validate_limits(value: Any) -> tuple[float, float]:
    try:
        low, high = value
    except (TypeError, ValueError):
        raise ValueError(f"expected a (low, high) pair, got {value!r}") from None
    return (float(low), float(high))


def _validate_orientation(value: Any) -> tuple[str, str, str]:
    try:
        codes = tuple(str(code).upper() for code in value)
    except TypeError:
        raise ValueError(f"expected three axis codes, got {value!r}") from None
    if len(codes) != 3 or any(code not in _AXIS_CODES for code in codes):
        raise ValueError(
            f"expected three axis codes from {''.join(sorted(_AXIS_CODES))}, got {value!r}"
        )
    return codes  # type: ignore[return-value]


def _validate_resample_order(value: Any) -> int:
    try:
        order = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected an int, got {value!r}") from None
    if not 0 <= order <= 5:
        raise ValueError(f"expected a spline order in [0, 5], got {value!r}")
    return order


def _validate_resample_mode(value: Any) -> str:
    mode = _validate_str(value)
    if mode not in _RESAMPLE_MODES:
        raise ValueError(f"expected one of {sorted(_RESAMPLE_MODES)}, got {mode!r}")
    return mode


def _validate_bbox_inches(value: Any) -> str | None:
    if value is None:
        return None
    return _validate_str(value)


def _validate_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return _validate_float(value)


def _validate_linewidths(value: Any) -> float | list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    try:
        return [float(width) for width in value]
    except TypeError:
        raise ValueError(f"expected a number or a sequence of numbers, got {value!r}") from None


def _validate_colors(value: Any) -> str | list[str]:
    if isinstance(value, str):
        return value
    try:
        return [_validate_str(color) for color in value]
    except TypeError:
        raise ValueError(f"expected a color or a sequence of colors, got {value!r}") from None


# ---------------------------------------------------------------------------
# The key set
# ---------------------------------------------------------------------------

_VALIDATORS: dict[str, Callable[[Any], Any]] = {
    # Image panels -- `plotting.images`
    "image.cmap": _validate_cmap,
    "image.ticker_steps": _validate_ticker_steps,
    "image.ax.xlabel": _validate_str,
    "image.ax.ylabel": _validate_str,
    "image.ax.xticks": _validate_ticks,
    "image.ax.yticks": _validate_ticks,
    "image.mask.colors": _validate_colors,
    "image.mask.linewidths": _validate_linewidths,
    # The `display_images` grid widget. Lengths land in the widget's CSS
    # custom properties, so their units are CSS units.
    "grid.max_cols": _validate_positive_int,
    "grid.max_height": _validate_css_length,
    "grid.panel_height": _validate_css_length,
    "grid.panel_min_height": _validate_css_length,
    "grid.panel_min_width": _validate_css_length,
    # Spectra axes -- `plotting.spectra`
    "spectra.xlim": _validate_limits,
    "spectra.xlabel": _validate_str,
    "spectra.ylabel": _validate_str,
    "spectra.ticker_steps": _validate_ticker_steps,
    "spectra.xticker_bins": _validate_ticker_bins,
    "spectra.yticker_bins": _validate_ticker_bins,
    # NIfTI handling -- `utils.nifti`
    "nifti.orientation": _validate_orientation,
    "nifti.resample.order": _validate_resample_order,
    "nifti.resample.mode": _validate_resample_mode,
    "nifti.resample.cval": _validate_float,
    # Figure saving -- `plotting.saving`
    "save.dpi": _validate_positive_int,
    "save.format": _validate_str,
    "save.transparent": _validate_bool,
    "save.bbox_inches": _validate_bbox_inches,
    "save.pad_inches": _validate_float_or_none,
}

_DEFAULTS: dict[str, Any] = {
    "image.cmap": "magma",
    "image.ticker_steps": [1, 2, 4],
    "image.ax.xlabel": "",
    "image.ax.ylabel": "",
    "image.ax.xticks": [],
    "image.ax.yticks": [],
    "image.mask.colors": "green",
    "image.mask.linewidths": 3,
    "grid.max_cols": 5,
    "grid.max_height": "50vh",
    "grid.panel_height": "17rem",
    "grid.panel_min_height": "8rem",
    "grid.panel_min_width": "9rem",
    "spectra.xlim": (8.1, -2.1),
    "spectra.xlabel": "$^2$H chemical shift [ppm]",
    "spectra.ylabel": "signal [a.u.]",
    "spectra.ticker_steps": [1, 2, 5],
    "spectra.xticker_bins": "auto",
    "spectra.yticker_bins": "auto",
    "nifti.orientation": ("L", "P", "S"),
    "nifti.resample.order": 0,
    "nifti.resample.mode": "grid-constant",
    "nifti.resample.cval": np.nan,
    "save.dpi": 150,
    "save.format": "png",
    "save.transparent": False,
    "save.bbox_inches": "tight",
    "save.pad_inches": None,
}


class RcParams(MutableMapping):
    """A validated ``str -> value`` mapping of package defaults.

    Behaves like a dict, except that only keys in `_VALIDATORS` exist and every
    assignment goes through that key's validator. Keys cannot be added or
    removed -- `rcdefaults` restores values, it never rebuilds the key set.
    """

    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        if mapping is not None:
            self.update(mapping)

    def __setitem__(self, key: str, value: Any) -> None:
        """Validate `value` against `key`'s validator and store the result."""
        validator = _VALIDATORS.get(key)
        if validator is None:
            raise KeyError(_unknown_key_message(key))
        try:
            self._data[key] = validator(value)
        except ValueError as err:
            raise ValueError(f"invalid value for rcParams[{key!r}]: {err}") from None

    def __getitem__(self, key: str) -> Any:
        """Return the current value of `key`."""
        try:
            return self._data[key]
        except KeyError:
            raise KeyError(_unknown_key_message(key)) from None

    def __delitem__(self, key: str) -> None:
        """Refuse deletion -- the key set is fixed by `_VALIDATORS`."""
        raise TypeError(
            f"rcParams keys cannot be removed; use `mnutils.rcdefaults()` to restore "
            f"{key!r} to its built-in default"
        )

    def __iter__(self) -> Iterator[str]:
        """Iterate over the parameter names, alphabetically."""
        return iter(sorted(self._data))

    def __len__(self) -> int:
        """Return the number of parameters."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a dict-like, one-key-per-line rendering."""
        body = "\n".join(f"  {key!r}: {self._data[key]!r}," for key in self)
        return f"RcParams({{\n{body}\n}})"

    def copy(self) -> RcParams:
        """A detached `RcParams` holding the same values."""
        clone = RcParams()
        clone._data = dict(self._data)
        return clone

    def group(self, prefix: str) -> dict[str, Any]:
        """The keys under ``prefix``, with the prefix stripped.

        ``rcParams.group("image.ax")`` gives ``{"xlabel": ..., "xticks": ...}``
        -- the shape a `matplotlib.axes.Axes.set` call wants, which is why the
        axis defaults are stored as individual keys rather than as one opaque
        dict value.
        """
        head = f"{prefix}." if not prefix.endswith(".") else prefix
        return {
            key[len(head) :]: value for key, value in self._data.items() if key.startswith(head)
        }


def _unknown_key_message(key: str) -> str:
    close = difflib.get_close_matches(key, _VALIDATORS, n=1)
    hint = f"; did you mean {close[0]!r}?" if close else ""
    return f"{key!r} is not a valid rcParam{hint}"


rcParamsDefault = RcParams(_DEFAULTS)
"""The built-in defaults. Never mutated -- `rcdefaults` restores from here."""

rcParams = RcParams(_DEFAULTS)
"""The live parameter mapping every MNUtils function reads its defaults from."""


# Named bundles for `rc_context`, replacing the old per-context ``*_SAVE_PARAMS``
# dicts: one mechanism for "change some defaults for a while" instead of two.
rc_presets: dict[str, dict[str, Any]] = {
    "poster": {"save.dpi": 300},
    "talk": {"save.transparent": True},
    "paper": {"save.dpi": 300, "save.pad_inches": 0.1},
}


def rcdefaults() -> None:
    """Restore every rcParam to its built-in default."""
    rcParams.update(rcParamsDefault)


@contextmanager
def rc_context(rc: dict[str, Any]) -> Iterator[RcParams]:
    """Temporarily override rcParams, restoring the previous values on exit.

    Parameters
    ----------
    rc : dict
        Overrides keyed by rcParam name, e.g. one of `rc_presets`. Names
        contain dots, so there is no keyword-argument form -- a mapping is the
        only way to spell them.

    Yields
    ------
    RcParams
        The live mapping, so it can be inspected or further modified inside
        the block.

    Examples
    --------
    >>> with rc_context({"image.cmap": "gray"}):
    ...     pass
    """
    previous = rcParams.copy()
    try:
        rcParams.update(rc)
        yield rcParams
    finally:
        rcParams._data = previous._data


def resolve_rc(value: Any, key: str) -> Any:
    """Return `value`, or ``rcParams[key]`` when it is ``None``.

    The one-liner behind every ``param: T | None = None`` signature in the
    package. Resolving in the body rather than in the signature is what makes
    an assignment to `rcParams` after import take effect at all.
    """
    return rcParams[key] if value is None else value


__all__ = [
    "RcParams",
    "rcParams",
    "rcParamsDefault",
    "rc_context",
    "rc_presets",
    "rcdefaults",
    "resolve_rc",
]
