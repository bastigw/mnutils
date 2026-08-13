# Explainer — `docs/concepts/` (not yet created)

The genre that answers **"why is it like this?"** — the permanent home of a concept, written as a
motivated narrative rather than a reference dump, once a concept in MNUtils outgrows a tutorial
aside or a diary entry.

No `docs/concepts/` chapter exists yet — MNUtils doesn't currently have material dense enough to
need it (no accessor pattern, no controlled vocabulary to explain). This template is here so the
first explainer has a shape to follow rather than inventing one from scratch; when you write the
first page under `docs/concepts/`, also add the chapter to `docs/myst.yml`'s TOC and to `GENRES`
in `check_docs.py`.

## Claims must be live cells, not static blocks

An explainer carries frontmatter and a kernelspec, so its code executes on every docs build. That
makes the choice of fence a choice about rot:

- **`code-cell`** for anything asserting how MNUtils *actually behaves*. If it stops being true,
  the build (or test) fails. Prefer this.
- **Static ```` ```python ````** only for code that must *not* run: an API that does not exist yet,
  or a deliberate ❌ anti-pattern.

```{warning}
Executed is not the same as *tested*. A cell that merely runs proves the API exists and does not
raise; proving a *value* still needs an `assert` in a `remove-cell`.
```

## Skeleton

````markdown
---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (mnutils)
  language: python
  name: python3
---

(my-concept)=
# The Concept, Named Plainly

<Two or three sentences establishing the ground the reader already stands on. Then the
tension — ideally as the reader's own question, in a blockquote:>

> **<The reader's actual objection.>**

<One line committing to follow that question. It shapes the whole page.>

(my-concept-problem)=
## The problem

<The tension made concrete. Show the failure, do not describe it — a live cell that raises
is worth a paragraph of prose.>

(my-concept-goal)=
## The goal

:::{important}
<The whole design in one sentence. If it takes two, the page is at the wrong altitude.>
:::

(my-concept-shape)=
## How it works

<The body. A contract table or decision-tree mermaid wherever it beats prose.>

:::{dropdown} Why not <the alternative>?
<What it would have bought, what it cost, why the cost won. Link the issue for the full
deliberation.>
:::
````

## What makes these pages work

- **Follow one question the whole way down.** A cold "Why X?" heading makes a sound decision read
  as an assertion to accept.
- **Name the tempting wrong answer, then kill it.**
- **Quote error messages verbatim.** Grep the actual string out of `src/`; do not paraphrase.
- **Put the deep rationale in a `:::{dropdown}`.** Off the main line, where it informs without
  derailing.
- **Every article stands alone.** Readers arrive from search and deep links. Declare a hard
  prerequisite in a `seealso` at the top; keep the orienting recap when you thin the page.
- **Guardrails get a `{warning}`,** including what fails loudly and why that is the desired
  behavior.

## Register

TOC group would be **Concepts** in `docs/myst.yml`, ideally near the tutorial chapter it explains.
Cross-link the tutorial that demonstrates the concept, and the diary entry that decided it, with
relative `.md` paths.
