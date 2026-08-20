# Contributing to MNUtils

Thanks for considering a contribution to **MNUtils** — utility functions for multi-nuclear MR data
from our GE scanners.

The full, always-current contribution guidelines live in the rendered documentation under
`docs/contribute/` (see `docs/contribute/index.md` — served locally with `uv run docs-mnutils` until a
public docs site exists).

## Quick start

```bash
git clone <repo-url>
cd MNUtils
uv sync --all-extras --dev   # uv replaces pip/virtualenv — see docs/contribute/setup.md
uv run pytest                # run the test suite (includes notebook tests via nbmake)
```

The package manager is [`uv`](https://docs.astral.sh/uv/); please never use `pip`.

## Questions, bugs, and ideas

Open an issue against this repository. For a substantial change, start with an issue or discussion
before a large PR so we can agree on the approach first.
