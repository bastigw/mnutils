"""Prove that a bare ``pip install mnutils`` is importable and usable.

Every other CI job installs with ``uv sync --all-extras --dev``, so the dependency
set a real user receives is exercised nowhere. A submodule that only imports cleanly
because a dev/test dependency happens to pull something in transitively (or because
the optional ``bet`` extra -- ``hd-bet``/``torch``, per ``CLAUDE.md``'s gotchas --
stopped being import-lazy) would pass every other job and still break on a fresh
install.

Run this with an interpreter whose environment was built from ``[project]
dependencies`` **alone** -- no extras, no dev group. It asserts two things:

1. the declared ``requires-python`` actually admits the running interpreter;
2. every top-level submodule imports without needing the ``bet`` extra, and doing
   so does not pull in ``torch``/``HD_BET`` (which must stay import-lazy).

Exits non-zero with a diagnostic on the first failure.
"""

import importlib
import importlib.metadata
import sys

import matplotlib

# A CI runner is headless. Select a non-interactive backend before pyplot is
# imported anywhere, so a missing display fails as a real bug and never as Tk.
matplotlib.use("Agg")

SUBMODULES = [
    "mnutils",
    "mnutils.GEExam",
    "mnutils.GESeries",
    "mnutils.plotting",
    "mnutils.utils",
    "mnutils.fitting",
]


def check_requires_python() -> None:
    """Assert the running interpreter satisfies the distribution's own metadata.

    ``requires-python`` was once spelled ``<=3.13``, which PEP 440 reads as
    ``<= 3.13.0`` -- excluding every 3.13 patch release from 3.13.1 on. Installing
    from a local path does not enforce the marker, so CI never noticed; a user
    running ``pip install mnutils`` on any real 3.13 was refused before dependency
    resolution even began. This catches that class of typo permanently.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    declared = importlib.metadata.metadata("mnutils")["Requires-Python"]
    running = Version(".".join(str(n) for n in sys.version_info[:3]))
    if running not in SpecifierSet(declared):
        raise AssertionError(
            f"Python {running} does not satisfy the declared Requires-Python "
            f"({declared}), so `pip install mnutils` would refuse this interpreter.\n"
            f"To fix this, correct `requires-python` in pyproject.toml -- a highest "
            f'supported minor of X.Y is spelled ">=...,<X.(Y+1)", never "<=X.Y".'
        )
    print(f"requires-python  OK  ({declared} admits {running})")


def check_submodules_import() -> None:
    """Import every top-level submodule and confirm the `bet` extra stayed lazy."""
    for name in SUBMODULES:
        importlib.import_module(name)
    print(f"submodule imports OK  ({', '.join(SUBMODULES)})")

    leaked = [m for m in ("torch", "HD_BET") if m in sys.modules]
    assert not leaked, (
        f"{', '.join(leaked)} ended up in sys.modules from a bare import -- the "
        f"`bet` extra (hd-bet/torch) must stay import-lazy per CLAUDE.md's gotchas, "
        f"so a plain `uv sync` (no extras) stays usable"
    )
    print("bet extra        OK  (torch/HD_BET not imported eagerly)")


def main() -> int:
    """Run every check, reporting the first failure."""
    for check in (check_requires_python, check_submodules_import):
        try:
            check()
        except Exception as exc:  # noqa: BLE001  (a smoke test reports, it does not handle)
            sys.stdout.flush()  # keep the passing lines above the failure in CI logs
            print(f"\nFAIL in {check.__name__}: {exc}", file=sys.stderr)
            return 1
    print("\nA bare `pip install mnutils` is importable and usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
