(diary-test-suite-speedup)=
# 150 seconds to run the docs is 150 seconds nobody waits out twice

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-27 · #48</span>

`uv run test-mnutils` took 150s. Profiling (`--durations=0`, `-X importtime`, cProfile) showed none
of it was one villain: a 2.0s `import mnutils` paid fresh by every notebook kernel, a
synthetic-exam fixture (see [the fixtures entry](#diary-synthetic-exam-fixtures)) rebuilt from
scratch in 9 of 11 kernels because its cache was per-*process*, and one page —
`nifti/partial_volume` — that was 50.9s on its own because `mask_occupancy` is called 12 times over
an 8.95M-voxel mask with a splat that pays for the fully general oblique case even when the grid
is axis-aligned.

:::{important}
Four independent fixes, landed in dependency order: lazy-import the two eager, rarely-needed
imports (pyAMARES, IPython) out of module scope; move the fixture cache from per-process
`functools.cache` to a content-hash-keyed directory on disk shared by every kernel/worker; add a
separable fast path to `mask_occupancy` for axis-aligned grids, guarded by an off-diagonal check
with the general path kept as fallback; then parallelize notebooks with `pytest-xdist` now that the
shared cache is safe under concurrent builders. Suite time: **150s → ~48-51s** (repeated locally).
:::

(diary-test-suite-speedup-shape)=
## Why this order

```{mermaid}
flowchart LR
    A["1. lazy imports"] --> B["2. fixture cache<br><span style='font-family:monospace'>~126s</span>"]
    B --> C["4. mask_occupancy<br>fast path<br><span style='font-family:monospace'>~101s</span>"]
    C --> D["3. xdist (-n 4)<br><span style='font-family:monospace'>~48-51s</span>"]
```

xdist has to come last: notebooks are independent processes, but only because nothing they share is
mutated concurrently. Step 2 changes that — once the fixture cache is a single directory every
kernel reads, two workers building the same dataset at once need to not stomp each other, and any
code that treats a fixture file as its own private, disposable copy needs to stop.

```python
# the shape of the fixture cache after this lands
from mnutils.testing import build_fake_exam

# same call site; internally now keyed by sha256(synthetic.py + _spectra.py)[:12]
# instead of functools.cache, so a second kernel reuses the first kernel's build
data = build_fake_exam("brain_mrs_mrsi_exam") / "data"
```

A build is race-safe two ways: an `mkdir`-based lock (atomic — `mkdir` on an existing path raises)
means only one process ever builds a given dataset, and the winner writes into a `.tmp` sibling and
`os.replace`s it into place, so a reader never observes a half-written tree.

:::{dropdown} Why not just skip the slow pages?
Issue #47 measured this and rejected it: with the fixes above the suite lands at ~48-51s, so a
skip flag buys little and costs documentation coverage. If a flag is ever wanted, the honest shape
is a `slow` marker scoped to `partial_volume` alone, off by default.
:::

:::{dropdown} Why `-n 4` instead of `-n auto`?
`-n auto` sizes to the CPU count — 22 logical cores here, xdist capping to 11 workers for 16 items.
Each worker's notebook launches its own `ipykernel` subprocess, and `ipykernel`'s TCP port
selection isn't atomic: under 11 kernels starting within the same second, two can pick the same
"free" port before either binds, crashing with `zmq.error.ZMQError: Address already in use`. `-n 4`
keeps enough overlap to matter (~48-51s vs. ~101s serial) without enough concurrent kernel startups
to hit the race in practice -- unlike the fixture-cache race below, this one lives in `ipykernel`
itself, not in this repo's code, so there's no fix to land here beyond bounding concurrency.
:::

:::{dropdown} Why a separable path instead of just speeding up the general splat?
The general path handles arbitrary affines (oblique acquisitions, real data) and stays exactly
correct — cell 66 of `partial_volume.md` deliberately exercises a 20°-rotated case, and that
coverage survives untouched. The separable path is only reachable when `tgt_from_mask[:3, :3]` is
diagonal (checked via `np.allclose` against the off-diagonal terms); everything else falls back to
`_splat_slab` unchanged. It's an exact reformulation (an outer/tensor product of three small
per-axis weight matrices), not an approximation: `docs/nifti/partial_volume.md`'s
"Does the axis-aligned fast path agree with the general path?" section reruns the same call through
both via a private `_force_general_path` test hook and asserts they agree to `atol=1e-6` — in
practice the two are bit-identical on the fixture's masks.
:::

## What changed from the plan

The plan assumed the fixture cache only needed to be *written* race-safely (lock + atomic rename).
It also needed to be *read-only* once built — something no code had tested yet, and the first
`-n auto` run found the gap immediately: `tests/test_data_loaders.py` read a shared fixture's
`Series7_raw_fids.h5`, then `os.remove()`d it as "cleanup," a leftover from when
`build_fake_exam`'s cache was per-process and that file really was the test's own disposable copy.
Under the new shared cache it deleted the file out from under whichever other kernel touched that
dataset next — `docs/data-model/geseries.md`'s `plot_washin()` cell, running concurrently in
another `xdist` worker, hit the resulting cache miss and fell through to `mnutils.matlab`'s real
MATLAB-engine path, which this environment doesn't have. The fix was deleting the `os.remove()`
call, not adding more locking: nothing in the fixture cache should ever be mutated by a reader once
`build_fake_exam` returns it, and that test was the one place doing so.

:::{seealso}
[The synthetic exam fixtures entry](#diary-synthetic-exam-fixtures) covers what the fixtures
fabricate and why; this entry only covers making the suite that runs them fast.
:::
