(contribute-docs)=
# Write a docs page

Every hand-authored page in these docs is a MyST notebook — jupytext frontmatter plus a kernelspec
— so any of them can run live `code-cell`s with real output and plots. There are two genres, split
by reader and by where their cells execute:

- **Tutorials** (`docs/basics/`, `data-model/`, `plotting/`, `fitting/`, `nifti/`, `matlab/`)
  demonstrate a task step by step, and *are* the test suite once they carry a jupytext kernelspec.
- **Guides** (`docs/contribute/`) are procedural pages like this one.

The four house-style rules — motivated narrative, one home per concept, every article stands
alone, and the MyST palette carries the argument — are the single source of truth in
[`CLAUDE.md` § Documentation style](https://github.com/bastigw/mnutils/blob/main/CLAUDE.md).

(contribute-docs-skill)=
## Working with Claude Code

The **`docs-page`** skill owns the cell structure, the hidden-assert convention, and the TOC step,
and it ships a stdlib-only checker (`check_docs.py`) that catches what the build stays silent
about — a missing target, a dead `.ipynb` link, a drifted kernel name. Run it on the page you are
editing:

```bash
uv run python .claude/skills/docs-page/check_docs.py docs/<path>/<page>.md
```

Its errors are render-breaking. Its warnings deliberately do not gate anything: they are real
drift, but too judgment-dependent to block a merge on. The checklist:

```{literalinclude} ../../.claude/skills/docs-page/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from .claude/skills/docs-page/SKILL.md
:class: skill-quote
```
