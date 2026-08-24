(contribute-home)=
# Contributing to `MNUtils`

Welcome! Rather than one long checklist that applies unevenly, the rules here are organised by
**the kind of change you are making**. Find your row and follow the page it points to:

| You are adding… | Start here |
|---|---|
| A docs page — tutorial or guide | [Write a docs page](#contribute-docs) |
| The record of a significant decision | [Write a dev-diary entry](#contribute-dev-diary) |
| A line in the release notes | [Write a changelog entry](#contribute-changelog) |
| A new version on PyPI | [Publish a release](#contribute-publishing) |

Each of those pages carries a **live checklist**, rendered straight from the Claude Code skill that
automates that kind of change — so whether you work by hand or with Claude, you follow the same,
always-current rules.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart LR
    C1["Docs page"] --> P["docs-page"]
    C2["Decision record"] --> V["dev-diary"]
    C3["Release note"] --> L["changelog"]
    C4["New version"] --> R["release"]
    P --> H["Documentation style"]
    V --> H
    R --> L
```

:::{note}
**For Claude Code users:** the four skills under
[`.claude/skills/`](https://github.com/bastigw/mnutils/tree/main/.claude/skills)
fire on the matching change above — except `release`, which is user-triggered only, since every
step of it pushes something irreversible. None of them carries rules of its own: each routes to the
one canonical doc that owns it, obeying the same "one home per concept" rule these docs preach.
:::

(contribute-home-first)=
## Before your first change

1. [**Set up your environment**](#setup) — clone the repo, run `uv sync --all-extras --dev`, and
   confirm `uv run pytest` is green.
2. **Make your change.** A significant decision starts as a
   [dev-diary draft](#contribute-dev-diary) that gets reviewed before the code is written.
3. **Open a merge/pull request.** MNUtils has no enforced branch-protection rules yet — don't
   assume required CI checks exist until they actually do (see `CLAUDE.md` § Commits).
4. **Drive it green, then hand off** for review.
