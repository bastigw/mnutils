(contribute-changelog)=
# Write a changelog entry

The [changelog](#changelog) answers one question, for one reader: *I just ran
`pip install -U mnutils` — what is different?* Nothing else on this site answers it. The
[dev diary](#diary-about) records **why** a decision was made; that is reasoning, and the changelog
is consequences.

It lives at `docs/changelog.md`. The `CHANGELOG.md` in the repo root is a pointer to this page and
nothing else — adding an entry there just creates a second copy that will disagree with the first.

There is no generator, and that is deliberate. `git log` is raw material: a squash-merge subject
says what the *diff* did, and a bullet has to say what the *user* got. Turning one into the other
is the whole job.

(contribute-changelog-shape)=
## The shape

One `## vX.Y.Z` heading per version, newest first, each with a single `(changelog-vX-Y-Z)=` target
above it. Inside a version, bullets are grouped under bold runs — **Breaking** · **Added** ·
**Changed** · **Fixed** · **Documentation** · **Maintenance** — with any empty group dropped. There
are deliberately no `###` headings: a release owns exactly one anchor, so inserting a version
churns no other page's deep links.

Every bullet is one sentence naming the **public symbol** (`display_images`, `GESeries`), followed
by its trail in a fixed order: issues → pull requests → docs page → diary entry. Issues and PRs are
full `https://github.com/bastigw/mnutils/…` URLs, because readers arrive from PyPI where a relative
link is dead. Docs and diary links are MyST targets, because `myst build` checks those for you and
checks nothing about a URL.

:::{warning}
An unresolved MyST target is reported as a **warning**, and `--strict` only promotes *errors* to a
non-zero exit — so a changelog full of dead cross-links builds green. Grep the build log for
`No target for internal reference`; the exit code will not tell you.
:::

:::{note}
The changelog is the one page on this site exempt from the motivated-narrative house rule. It is a
reference genre: no driving question, no admonitions, no dropdowns. A reader scans it. Do not carry
that shape out to any other page.
:::

(contribute-changelog-skill)=
## Working with Claude Code

The **`changelog`** skill's checklist:

```{literalinclude} ../../.claude/skills/changelog/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from .claude/skills/changelog/SKILL.md
:class: skill-quote
```
