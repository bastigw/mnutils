(diary-mask-partial-volume)=
# A mask voxel is a box, not a point

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-24</span>

A segmentation arrives at 512 × 512 × 100, half a millimetre in plane. The spectrum it has to
annotate arrives at 16 × 16 × 16, ten to fifteen millimetres a side, on a different field of view
and usually oblique to the anatomy. In between sits the number you want: *what fraction of this
spectroscopic voxel is tumour?* `downsample_to_coverage` claimed to answer it — but pointed at an
analytically sharp edge, it placed the boundary half a voxel from where the affines say it is, on
all three axes, `z` opposite to `x` and `y`. At these voxel sizes half a voxel is seven and a half
millimetres.

:::{important}
Occupancy is computed by pushing every mask voxel through the affines into the MRSI grid and
splitting it across the target voxels it overlaps — not by sampling each MRSI voxel and hoping the
sample grid was dense enough.
:::

(diary-mask-partial-volume-direction)=
## Which way to integrate

Both directions are defensible; the resolution ratio picks between them, and it is worth knowing
the rule rather than the answer:

```{mermaid}
flowchart TD
  Q{"Which grid is finer?"}
  Q -->|"source much finer<br>(here: about 12-16x per axis)"| F["Forward — scatter.<br>Every source voxel contributes once.<br>Cost scales with the source."]
  Q -->|"comparable, or source coarser"| B["Backward — gather.<br>Supersample each target voxel.<br>Cost scales with samples chosen."]
  F --> R["Exact repartition of mass:<br>nothing created, nothing lost"]
  B --> S["Accuracy is a tuning parameter"]
```

Forward binning is the `n → ∞` limit of supersampling at a fraction of the cost, because the mask
grid *is* the sample grid. That buys an exact property rather than a tolerance: the operation only
moves mass between bins, so mask volume in equals occupied volume out — for the part of the mask
that lands on the target grid at all. On the synthetic brain that identity holds to float32
precision for a sphere well inside both fields of view, while the brain mask itself loses 0.16 %
off the rim, where the MRSI grid stops before the T1 does.

(diary-mask-partial-volume-two-defects)=
## Two ways to be off by half a voxel

The first was the one the old routine had, and it has a one-line cause worth stating because it is
easy to reintroduce: a NIfTI index names a voxel **centre**, so voxel `i` spans `[i − 0.5, i + 0.5)`
and mapped coordinates must be binned with `floor(idx + 0.5)`, never `floor(idx)`.

The second only appeared once the first was fixed, and it is the reason this entry is not called
*"every mask voxel gets exactly one vote"* any more. Dropping each mask voxel whole into its
nearest bin is the obvious implementation, and it fails for a reason that has nothing to do with
registration: the ratios here are 11.75, 16.0 and 11.625 mask voxels per target voxel, so a target
voxel catches *either 11 or 12* of them per axis depending on where the lattice falls. The count
quantises, and the fraction built from it wobbles by roughly ±8 %.

:::{warning}
That wobble is not cosmetic. On the synthetic brain, **300 of 3375 target voxels that are provably
100 % inside the T1** came back below `min_coverage=0.9` and were silently NaN'd. Coverage on
those voxels spanned 0.886 to 1.054 when the correct answer is exactly 1.
:::

Treating each mask voxel as the box it actually is — half-extent `0.5 * Σ_b |T[a, b]|` along target
axis `a`, a row sum rather than a diagonal so obliquity is included — and splitting it between bins
in proportion to overlap removes this entirely. The weights of one mask voxel sum to 1 by
construction, so the repartition property survives, and coverage on those same 3375 voxels becomes
exactly 1.0.

```python
pv = mask_occupancy(seg_nii, mrsi.RAW_exp, min_coverage=0.9)

# (i, j, k), NaN where coverage fell short
pv.occupancy.sel(label="tumour")

# (i, j, k), never masked
pv.coverage
```

Two variables, because two different questions hide inside one number:

| Variable | Answers | Masked by `min_coverage` |
|---|---|---|
| `occupancy` | "How much of this voxel is the label?" | yes — NaN below the threshold |
| `coverage` | "How much of this voxel did the anatomy even reach?" | no — always readable |

Collapsing them loses the distinction between *no tumour here* and *no data here*, which at the
edge of an MRSI slab is most of the interesting voxels.

(diary-mask-partial-volume-details)=
## Three things that only showed up in code

**Labels are detected by value, not dtype.** `nibabel`'s `get_fdata()` returns float64 whatever the
file stores, so a dtype test would classify every segmentation read from disk as a probability map.
The rule is instead: one distinct non-zero value is a binary mask, fractional values are a
probability map, several distinct whole numbers are labels.

**Affines in `.attrs` are flattened.** netCDF attributes must be one-dimensional, and a nested 4 × 4
made the whole Dataset impossible to save — the round trip is now a test on the docs page.

**Labels sum to `coverage` only if the background counts as a label.** Left to auto-detect,
`mask_occupancy` finds the non-zero labels and their sum falls short of coverage by exactly the
background fraction. The clean partition identity needs `labels={0: "background", ...}`.

:::{dropdown} Why not vectorise the supersampler we already had?
That is the plan [issue #30](https://github.com/bastigw/mnutils/issues/30) records, and it hinges
on reproducing the current output bit-exactly — which would preserve the mis-centring. Its
`s = ceil(diag(tgt2src))` also reads only the diagonal, understating the scale precisely when the
grids are oblique.
:::

:::{dropdown} Why not weight by the MRSI point-spread function?
A 16³ Hamming-apodised grid has no box-shaped voxels; signal genuinely bleeds across boundaries.
But PSF-weighted occupancy is a different quantity, needs the acquisition's apodisation, and goes
negative in the sidelobes. Box overlap is geometric and checkable against an overlay. The kernel
stays swappable.
:::

(diary-mask-partial-volume-changed)=
## What changed from the plan

- The plan treated overlap-weighted splitting as a later refinement worth roughly 8× the cost for
  an error "already below registration accuracy". That estimate assumed 0.5 mm mask voxels; the
  real pairing is 0.88 mm anatomy against 10–14 mm MRSI voxels, only ~12 voxels per axis, where
  the quantisation is large enough to break the default `min_coverage` threshold outright.
  Splitting is not a refinement here, it is the thing that makes the feature work.
- The plan asserted mass conservation as an unqualified identity and wrote the tests around it. It
  holds exactly, but only for mask that lands on the target grid — a distinction invisible until a
  mask whose extent matches the target's was actually run through it.
