---
name: changelog
description: Write or update an MNUtils changelog entry — the user-facing record of what shipped in a release, in docs/changelog.md. Use at release time (the `release` skill calls it while the full matrix runs), or standalone to correct, re-curate or top up an existing version section.
---

# Write an MNUtils changelog entry

`docs/changelog.md` answers one question, for one reader: *I just ran `pip install -U mnutils` —
what is different?* Nothing else on the site answers it. The [dev diary](#diary-about) records
**why** a decision was made; that is about reasoning, and this page is about consequences.

There is no generator. `git log` is raw material, not a draft — the squash-merge subject says what
the *diff* did, and a changelog bullet has to say what the *user* got. Turning one into the other
is the whole job, and it is why this is a skill rather than a tool.

The root `CHANGELOG.md` is a **pointer file**, not a second copy. Never add an entry there.

```{note}
**The genre carve-out.** House style lives in `CLAUDE.md` § "Documentation style" and is not
restated here — with one exception, and it is a big one: **a changelog is a reference genre, not a
motivated narrative.** No driving question, no felt tension, no admonition palette, no
`{dropdown}`. A reader scans it; they do not read it. This is the only page on the site that works
this way, so do not carry the shape *out* of here — and `docs-page`'s templates do not apply *in*
here.
```

## 1. Establish the range

```bash
git describe --tags --abbrev=0 --match 'v*'          # the last released tag, e.g. v1.2.0
git log <tag>..HEAD --reverse --no-merges --pretty='%h %s'
```

`--match 'v*'` is not optional: the repo carries non-version tags (`pre-xmris-rebuild`), and a bare
`git describe` will happily hand you one and silently widen the range by a whole rebuild.

Every line in that output is accounted for by the end — as a bullet, or as a deliberate drop (§3).
That accounting is what replaces a generator's guarantee that nothing was silently missed.

Adding to a section that already exists (a release that slipped, a correction after the fact)? Use
the range since the last commit the section already covers, and merge into the existing groups
rather than appending a second block.

## 2. Harvest the trail

Every bullet carries links, and most of them are mechanical. Collect them per commit:

| Link | Where it comes from |
|---|---|
| **PR** | the `(#N)` suffix squash-merge leaves on the subject |
| **Issue(s)** | `gh pr view <N> --json body` — a PR body that names what it closes. Link only the issues you actually confirmed; a plausible-looking number is worse than none |
| **Docs page** | `git show --stat <sha>` → a new or substantially rewritten `docs/**/*.md`; read its page target from the file head (`grep -m1 -oE '^\([a-z0-9-]+\)=' <file>`) |
| **Diary entry** | the same `--stat` → a touched `docs/diary/*.md`; use its `(diary-…)=` target |

`gh pr view <N> --json body` is also how you write the *sentence* when the subject does not explain
the user impact. The subject is a headline; read the body for anything you cannot describe in user
terms from the subject alone.

## 3. Decide what earns a bullet

The test is **user-visible consequence**, not commit type. A `chore:` that changes what
`pip install mnutils` pulls in earns a bullet; a `feat:` that only adds an internal helper does not.

| Material | Treatment |
|---|---|
| `build(deps)` / `build(deps-dev)` | **collapse** — one **Maintenance** bullet naming the count, no per-bump links |
| `ci:` / `chore:` with no user-visible effect | **drop** |
| `refactor:` behind an unchanged public surface | **drop**, unless it changed a default, an export, or an error |
| `docs:` | a bullet only when it changes what a reader should read or do — and then the new page *is* the link |
| version bumps, merge-back noise, reverted pairs | **drop** |

Write the drops down as you go. If a line is neither a bullet nor a stated drop, the accounting in
§1 has not been done.

## 4. Write it

Group under bold-run labels, in this order, omitting any that is empty:

**Breaking** · **Added** · **Changed** · **Fixed** · **Documentation** · **Maintenance**

`**Breaking**` leads whenever it applies, and it is the only group a reader must not scroll past.
MNUtils is past 1.0, so a bullet landing there is also a claim about the version number — if
`**Breaking**` has content and the bump is not major, one of the two is wrong. Raise it rather
than quietly writing the bullet.

One bullet, one sentence, then the trail:

```markdown
- `display_images` renders a self-contained HTML slice viewer. —
  [#12](https://github.com/bastigw/mnutils/pull/12) ·
  [Displaying image arrays of any shape](#plotting-images) ·
  [diary](#diary-anywidget-slice-viewer)
```

- **Trail order is fixed: issues → PRs → docs page → diary.** Issues and PRs are bare `#N` with
  full `https://github.com/bastigw/mnutils/…` URLs, since a reader may arrive from PyPI. Pages are
  linked by their **title**; a diary entry by the word *diary*.
- **Docs and diary links use MyST targets** (`[text](#plotting-images)`), never URLs or file paths.
  Targets resolve page-globally and the build reports every one it cannot find, so that half of the
  trail is machine-checkable — a URL is checked by nothing. Read §6 for how, and for the trap: an
  unresolved target is a **warning**, so the exit code will not tell you.
- **Name the public symbol**, never the internal module: `display_images`, `GESeries`, `GEExam` —
  not `plotting/images.py`. The reader has never seen `src/`.
- Past tense or plain present, one sentence, no lead-in ("This release adds…"). Say what it does
  now.
- Several PRs delivering one capability get **one** bullet with several PR links, not one each.

## 5. Section shape and targets

```markdown
(changelog-v1-3-0)=
## v1.3.0 — unreleased
```

- **One H2 per version, newest first, exactly one `(target)=` per version.** Deliberately no H3s:
  the group labels are bold runs, so a release has one anchor — the one anyone would deep-link —
  and inserting a version churns no other anchor.
- The target is the version with dots as hyphens: `v1.3.0` → `(changelog-v1-3-0)=`.
- Heading reads `## vX.Y.Z — unreleased` until the release, then `## vX.Y.Z — YYYY-MM-DD`. The
  `release` skill greps for that word before it tags.
- `## Earlier releases` stays pinned at the bottom — everything up to `v1.1.2` predates the page.

**Never explain the page on the page.** The intro is one line of signposting to the diary, and it
is already written — do not extend it, and never describe the format ("every entry links the PR
that…"), the audience, or the changelog's own history. The reader can see the format; they are
here for the bullets. The same applies to the one-line summary under a version heading: it names
what shipped, not what the changelog is.

Read `templates/entry.md` for the shape to copy.

## 6. Verify

```bash
python3 .claude/skills/docs-page/check_docs.py docs/changelog.md     # 0 errors
cd docs && uv run myst build --html --strict 2>&1 | tee /tmp/mnutils-build.log
grep "No target for internal reference" /tmp/mnutils-build.log          # must print nothing
```

`myst build` is one-shot (~10 s warm). **Never `uv run docs-mnutils`** — it starts a blocking
preview server and never exits. It also leaves the shell in `docs/`; use absolute paths afterwards.

```{warning}
**The grep is not optional.** A broken `#target` is reported as a *warning*, and `--strict` only
promotes **errors** to a non-zero exit — so a changelog full of dead cross-links builds green. The
exit code catches a malformed directive; only the grep catches a mistyped target.
```

That covers the docs and diary half of every trail. PR numbers no build can check: spot-check a few
against `gh pr view <N> --json title`. (`deploy.yml` runs `--check-links`, which does reach the
GitHub URLs — but only after merge, which is too late to be your check.)

Then read the rendered page. Every bullet has to make sense to someone who has never seen the diff
— that is the only test that matters, and it is not automatable.

## Checklist

<!-- excerpt:start -->
- [ ] Range established from the last `v*` tag; **every commit in it is a bullet or a stated drop**
- [ ] Bullets describe user-visible consequence, naming the public symbol — never a module path
- [ ] Trail on every bullet, in order: issues → PRs → docs page → diary, the last two as MyST targets
- [ ] `build(deps)` collapsed to one **Maintenance** bullet; `ci:`/internal `chore:`/`refactor:` dropped
- [ ] Groups in order, `**Breaking**` first where it applies — and the version bump agrees with it
- [ ] One H2 and one `(changelog-vX-Y-Z)=` per version, newest first; root `CHANGELOG.md` untouched
- [ ] `check_docs.py` clean, `myst build --strict` exits 0, **and** the build log is free of
      `No target for internal reference` — the exit code alone does not prove the targets resolve
- [ ] Rendered page read start to finish; no bullet requires the diff to understand
<!-- excerpt:end -->
