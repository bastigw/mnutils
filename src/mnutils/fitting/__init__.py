"""Spectral fitting, delegated to pyAMARES through a thin wrapper."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import AMARES

__all__ = ["AMARES"]


def __getattr__(name: str):
    if name == "AMARES":
        from importlib import import_module

        return import_module(f"{__name__}.AMARES")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
