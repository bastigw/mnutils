(contribute-publishing)=
# Publish a release

MNUtils ships to [PyPI](https://pypi.org/project/mnutils/) through GitHub Actions. Nobody runs
`uv publish` by hand — `.github/workflows/ci-publish.yml` does it, authenticated by PyPI Trusted
Publishing (OIDC), so there is no API token to hold or leak. **Pushing a `vX.Y.Z` tag is the
publish.**

That makes a release two pushes with very different consequences, and the workflow exists to keep
them apart.

(contribute-publishing-two-triggers)=
## Two triggers, one workflow

`ci-publish.yml` listens for both a `release/**` branch and a `v*` tag. The branch runs the full
matrix and stops; the tag runs the same matrix and then publishes.

```{mermaid}
%%{init: {'flowchart': {'htmlLabels': false}}}%%
flowchart TD
    B["push release/vX.Y.Z<br/>(version NOT bumped)"] --> M1["Full matrix<br/>3 OS × Py 3.12–3.14"]
    M1 -->|green| BUMP["Bump version + changelog<br/>→ pull request → main"]
    BUMP --> T["tag vX.Y.Z on the merged commit"]
    T --> M2["Full matrix again"]
    M2 -->|green| P["uv build --no-sources<br/>uv publish → PyPI"]
    P --> R["GitHub Release, written by hand"]
```

The release branch is pushed **unbumped**, and that is the point. Running the matrix before the
version changes means a failure costs you a fix, not a burnt version number — otherwise every red
run leaves behind a bump commit you have to redo. A PyPI version, once uploaded, can never be
reused, not even after deleting it.

(contribute-publishing-tag-target)=
## Tag the merged commit, never the branch

Pull requests here are squash-merged, so the commits on `release/vX.Y.Z` are not the commits that
end up on `main`. A tag cut on the branch would point at an object that never enters `main`'s
history, and `git describe` on `main` would never see the release. Check out `main`, pull, confirm
the tip is the bump commit, and tag there.

Two checks belong immediately before the tag, because nothing downstream repeats them:

```bash
uv version                                    # MUST print X.Y.Z
grep -n "^## vX.Y.Z" docs/changelog.md        # MUST exist, and MUST NOT say "unreleased"
```

`ci-publish.yml` deliberately knows nothing about the [changelog](#contribute-changelog). A check
that fired after the tag would leave you deleting and re-pushing one — so the guard lives here, in
the last moment it is still cheap.

:::{note}
MNUtils has no enforced branch protection yet (`CLAUDE.md` § Commits), so nothing mechanically
stops a direct push to `main`. Land the bump through a pull request anyway: that is the only place
the five fast checks — docs style, lint, bare install, tests on 3.12 and 3.14 — run against the
merge result.
:::

(contribute-publishing-release-notes)=
## The GitHub Release is written by hand

CI stops at PyPI. Creating the GitHub Release is a manual `gh release create` for two reasons: it
keeps `contents: write` off the job that holds publish rights, and it means the announcement only
ever exists for a version that actually shipped. Write its notes from the changelog section,
rewriting the MyST targets (`[Displaying image arrays of any shape](#plotting-images)`) into full
`https://bastigw.github.io/mnutils/…` URLs — GitHub cannot resolve a MyST anchor.

(contribute-publishing-skill)=
## Working with Claude Code

The **`release`** skill runs this page as an operational checklist. It is user-triggered only —
it never fires on its own, because every step of it pushes something irreversible:

```{literalinclude} ../../.claude/skills/release/SKILL.md
:language: markdown
:start-after: <!-- excerpt:start -->
:end-before: <!-- excerpt:end -->
:caption: Quote from .claude/skills/release/SKILL.md
:class: skill-quote
```
