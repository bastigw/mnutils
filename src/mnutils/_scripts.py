"""Thin `uv run <name>` wrappers around the MyST/quartodoc docs toolchain.

Simpler than the equivalent script in the sibling `xmris` package: MNUtils has
no interactive-widget config classes to auto-document, so this is just
`quartodoc build` (which emits Quarto Markdown, `.qmd`) followed by a
translation pass into plain MyST Markdown (`.md`), then subprocess wrappers
around `myst build`/`myst start`.
"""

import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _find_project_root() -> Path:
    """Return the root of the MNUtils project, or exit if not found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "Error: could not find MNUtils project root (no pyproject.toml in any parent)"
    raise RuntimeError(msg)


def _get_docs_dir() -> Path:
    docs_dir = _find_project_root() / "docs"

    if not docs_dir.exists():
        print(f"Error: 'docs' directory not found at: {docs_dir!s}")
        sys.exit(1)

    return docs_dir


def _translate_qmd_to_myst(api_dir: Path) -> None:
    """Convert quartodoc's `.qmd` output into plain MyST `.md`.

    quartodoc targets Quarto by default; mystmd needs MyST Markdown instead.
    This rewrites Quarto's `{.doc-method #anchor}` attribute blocks into
    explicit `(anchor)=` MyST targets, drops leftover Quarto CSS blocks, and
    downgrades any local link whose target didn't survive rather than
    shipping a silently dead link. That downgrade is expected and routine
    for plain attributes -- quartodoc's summary table links every attribute
    by name, but (unlike methods) an attribute never gets its own anchored
    section to link to, documented or not -- so it's counted, not printed
    per occurrence; see the summary line `docs_api()` prints at the end.
    """
    downgraded = 0
    for qmd_file in api_dir.rglob("*.qmd"):
        content = qmd_file.read_text(encoding="utf-8")

        # Fix internal file extensions to point to standard Markdown.
        content = content.replace(".qmd", ".md")

        # Converts: ### Heading {.doc-method #anchor}  ->  (anchor)= \n ### Heading
        content = re.sub(
            r"^(.*?)\s*\{[^\}]*?#([\w\.\-]+)[^\}]*\}\s*$",
            r"(\2)=\n\1",
            content,
            flags=re.MULTILINE,
        )
        # Strip any leftover pure CSS blocks that didn't have an ID (e.g. {.doc-signature}).
        content = re.sub(r"\s*\{\.[^\}]+\}", "", content)

        valid_targets = set(re.findall(r"^\(([\w\.\-]+)\)=", content, flags=re.MULTILINE))

        def link_replacer(match: re.Match) -> str:
            nonlocal downgraded
            text, anchor = match.group(1), match.group(2)
            if anchor in valid_targets:
                return match.group(0)
            downgraded += 1
            return f"`{text}`"

        content = re.sub(r"\[([^\]]+)\]\(#([\w\.\-]+)\)", link_replacer, content)

        md_file = qmd_file.with_suffix(".md")
        md_file.write_text(content, encoding="utf-8")
        qmd_file.unlink()

    # Give the generated landing page a stable MyST target -- quartodoc emits
    # a bare `# Function reference` with no anchor, so without this every
    # link into the API chapter would resolve against a URL instead.
    index_file = api_dir / "index.md"
    if index_file.exists():
        index_text = index_file.read_text(encoding="utf-8")
        if not index_text.startswith("(api-home)="):
            index_file.write_text(f"(api-home)=\n{index_text.lstrip()}", encoding="utf-8")

    if downgraded:
        print(
            f"docs-api: downgraded {downgraded} dead attribute link(s) to plain text "
            "(expected -- attributes never get their own anchor)"
        )


def docs_api() -> None:
    """Regenerate the quartodoc API reference stubs under docs/api/."""
    docs_dir = _get_docs_dir()
    api_dir = docs_dir / "api"

    if api_dir.exists():
        shutil.rmtree(api_dir)
    api_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["quartodoc", "build", "--config", "quartodoc.yml"], cwd=docs_dir, check=True)
    _translate_qmd_to_myst(api_dir)


def docs_notebooks() -> None:
    """Serve a live preview of the docs, executing notebooks. Blocking."""
    docs_dir = _get_docs_dir()
    subprocess.run(["myst", "start", "--execute"], cwd=docs_dir, check=True)


def docs_all() -> None:
    """Regenerate API stubs, then serve a live preview. Blocking."""
    docs_api()
    docs_notebooks()


# Docs chapters whose pages are executed as tests. One directory per sidebar
# chapter, mirrored into tests/autogen_notebooks/<chapter>/. Keep in sync with
# GENRES in .claude/skills/docs-page/check_docs.py -- that file decides which
# house rules a chapter's pages are held to, this one decides which get run.
# `matlab` is deliberately excluded: its pages need a live MATLAB install and
# a matching `matlabengine` version (see CLAUDE.md's Gotchas), which no CI or
# contributor machine can be assumed to have -- it isn't run by default.
TEST_CHAPTERS = ("basics", "data-model", "plotting", "fitting", "nifti")


def _convert_to_notebook(md_file: Path, out_dir: Path) -> None:
    subprocess.run(
        [
            "jupytext",
            "--to",
            "notebook",
            "--output",
            str(out_dir / f"{md_file.stem}.ipynb"),
            str(md_file),
        ],
        check=True,
    )


def generate_test_notebooks() -> None:
    """Convert every executable page under TEST_CHAPTERS into a test notebook.

    Mirrors xmris's docs-are-the-tests pattern: a page only joins the suite
    once it carries a jupytext kernelspec (frontmatter-less pages, like a bare
    chapter `index.md`, are skipped rather than failing with no kernel to
    start). Output goes under tests/autogen_notebooks/, which is gitignored --
    regenerate it before running pytest, never commit it.

    Each page is its own `jupytext` subprocess (~0.3-0.5s of interpreter
    startup apiece) with nothing shared between them, so they run in a thread
    pool -- the GIL doesn't matter here since every thread spends its time
    blocked in `subprocess.run`, not executing Python.
    """
    project_root = _find_project_root()
    docs_dir = project_root / "docs"
    out_root = project_root / "tests" / "autogen_notebooks"

    if out_root.exists():
        shutil.rmtree(out_root)

    jobs: list[tuple[Path, Path]] = []
    for chapter in TEST_CHAPTERS:
        chapter_dir = docs_dir / chapter
        if not chapter_dir.is_dir():
            continue
        out_dir = out_root / chapter
        for md_file in sorted(chapter_dir.glob("*.md")):
            if _is_executable_page(md_file):
                jobs.append((md_file, out_dir))

    for out_dir in {out_dir for _, out_dir in jobs}:
        out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda job: _convert_to_notebook(*job), jobs))


def _is_executable_page(md_file: Path) -> bool:
    """A page joins the test suite only if it carries a jupytext kernelspec."""
    head = md_file.read_text(encoding="utf-8")[:400]
    return head.startswith("---") and "kernelspec:" in head


def run_tests() -> None:
    """Regenerate the notebook tests from docs, then run the full pytest suite."""
    generate_test_notebooks()
    subprocess.run(["pytest"], check=True)
