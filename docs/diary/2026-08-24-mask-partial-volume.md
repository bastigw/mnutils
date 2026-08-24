(diary-mask-partial-volume)=
# Every mask voxel gets exactly one vote

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-08-24</span>

A segmentation arrives at 512 × 512 × 100, half a millimetre in plane. The spectrum it has to
annotate arrives at 16 × 16 × 16, fifteen millimetres a side, on a different field of view and
usually oblique to the anatomy. In between sits the number you want: *what fraction of this
spectroscopic voxel is tumour?* `downsample_to_coverage` claimed to answer it — but pointed at an
analytically sharp edge, it placed the boundary half a voxel from where the affines say it is, on
all three axes, `z` opposite to `x` and `y`. Here half a voxel is seven and a half millimetres.

:::{important}
Occupancy is computed by pushing every mask voxel through the affines into the MRSI grid and
counting where it lands — not by sampling each MRSI voxel and hoping the sample grid was dense
enough.
:::

(diary-mask-partial-volume-direction)=
## Which way to integrate

Both directions are defensible; the resolution ratio picks between them, and it is worth knowing
the rule rather than the answer:

```{mermaid}
flowchart TD
  Q{"Which grid is finer?"}
  Q -->|"source much finer<br>(here: about 30x in plane)"| F["Forward — scatter.<br>Every source voxel votes once.<br>Cost scales with the source."]
  Q -->|"comparable, or source coarser"| B["Backward — gather.<br>Supersample each target voxel.<br>Cost scales with samples chosen."]
  F --> R["Exact repartition of mass:<br>nothing created, nothing lost"]
  B --> S["Accuracy is a tuning parameter"]
```

Forward binning is the `n → ∞` limit of supersampling at a fraction of the cost, because the mask
grid *is* the sample grid. That buys an exact property rather than a tolerance: the operation only
moves mass between bins, so mask volume in equals occupied volume out. That identity, not an error
bound, is what the tests assert.

The half-voxel bug has a one-line cause, worth stating because it is easy to reintroduce: a NIfTI
index names a voxel **centre**, so voxel `i` spans `[i − 0.5, i + 0.5)` and mapped coordinates
must be binned with `floor(idx + 0.5)`, never `floor(idx)`.

```python
pv = mask_occupancy(seg_nii, mrsi_series, min_coverage=0.9)

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

:::{attention} Assumptions to verify
- That mass conservation holds *exactly* in floating point, not merely to a tolerance. The tests
  are written around it, and an oblique affine is where it would first fray.
- That `1 / |det(T)|` is the right denominator for anisotropic mask voxels — the volume ratio the
  affines imply, not yet checked against a hand-counted case.
- That the half-voxel shift is the *only* registration error in the old routine. Measured on
  axis-aligned grids only; an oblique test could expose a second.
- That xarray's non-dimension 3-D world coordinates survive a round trip to disk. Nothing here has
  serialised one yet.
- That `min_coverage=0.9` is sensible. It is a judgement call, not a calibrated one.
:::
