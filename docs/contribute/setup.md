(setup)=
# Recommended Setup

We use modern, Rust-based tooling to keep the development environment fast, reproducible, and free
of dependency conflicts.

(setup-uv)=
### 1. `uv` (Environment & Package Management)

`uv` replaces `pip`, `virtualenv`, and `poetry`. It manages our dependencies, locks versions, and
ensures perfectly isolated virtual environments.

* **Install `uv`:** Run `curl -LsSf https://astral.sh/uv/install.sh | sh` (Mac/Linux) or check the
  [official docs](https://docs.astral.sh/uv/) for Windows/Homebrew methods.
* **Bootstrap the project:** Run `uv sync --all-extras --dev` in the root directory. This reads
  `pyproject.toml`, resolves dependencies, and automatically creates the `.venv` folder.
* **Run commands:** Always prefix development commands with `uv run` to ensure they execute inside
  the isolated environment (e.g., `uv run pytest`).
* **Add dependencies:** Do not use `pip install`. Instead, run `uv add <package_name>`. This
  automatically updates `pyproject.toml` and the lockfile.

(setup-matlab)=
### 2. MATLAB Engine for Python

Some functions call into MATLAB. Install the `matlabengine` version matching your local MATLAB
installation — see the [MATLAB Engine for Python PyPI page](https://pypi.org/project/matlabengine/#history)
for the version table, then:

```bash
uv add "matlabengine==<version>"
```

(setup-ruff)=
### 3. `ruff` (Linting & Formatting)

* **Format code:** Run `uv run ruff format .`
* **Lint code:** Run `uv run ruff check .`
* **Auto-fix issues:** Run `uv run ruff check . --fix`

(setup-vscode)=
### 4. VS Code Configuration

Recommended extensions:

* [Python (Microsoft)](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* [Ruff (Astral Software)](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

**Settings (`.vscode/settings.json`):**

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": true,
      "source.organizeImports.ruff": true
    }
  }
}
```
