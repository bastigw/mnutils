(diary-deprecate-amares-wrapper)=
# Two AMARES front ends, one engine: delete ours

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-11-03</span>

`mnutils.fitting.AMARES` is ~830 lines wrapping pyAMARES. `xmris` now ships its own front end,
`xmris.fitting.fit_amares`, over the same engine — batch, parallel, domain-preserving, with a
`fit_status` flag. So there are two sets of defaults (`method`, `num_workers`, `g_global`,
`ppm_offset` all differ) and two result formats, and one of them is MNUtils-side upkeep for logic
`CLAUDE.md` says belongs upstream. Keep both and they drift apart; the question is whether to
soften the exit or just take it.

:::{important}
`GESeries` moves onto `xmris.fit_amares` and returns its `xr.Dataset`; `mnutils.fitting.AMARES` is
deleted outright in the same change — no deprecation cycle, no shims.
:::

(diary-deprecate-amares-wrapper-shape)=
## Shape: migrate the callers, then delete

```{mermaid}
flowchart LR
    A["1. gap analysis<br>(AMARES.py vs xmris)"] --> B["2. upstream gaps<br>to xmris"]
    B --> C["3. GESeries fits<br>via fit_amares"]
    C --> D["4. delete AMARES.py<br>and its docs/tests"]
```

Anything xmris lacks (ppm filtering, the fit plot) is pushed upstream rather than carried here, so
the gap analysis decides whether step 4 can happen in this change or waits on an xmris release.

| Old (`fitting.AMARES`) | New |
|---|---|
| `fit_single_fid`, `fit_multiple_fids` | `fit_amares` (single voxel = length-1 dim) |
| `extract_from_fit_results`, `FitResults` | the returned `xr.Dataset` |
| `priorknowledge_to_multiindex` and inverse | `build_prior_knowledge` |
| `save/load_extracted_results` (HDF5) | `to_netcdf` / `from_netcdf`; old HDF5 files are not readable |
| `GESeries.snr` via `pyAMARES.fidSNR` | xmris SNR |

```python
# the call site we want: no mnutils.fitting import at all
series = GESeries.load(...)
ds = series.fit_all_voxels(prior_knowledge)     # an xr.Dataset
ds["amplitude"].sel(metabolite="PCr")
ds.where(ds.fit_status)                          # failed fits masked, not silently kept
```

:::{dropdown} Why not keep our wrapper and just fix the defaults?
It is the cheapest option, but it leaves two front ends to keep in step with every pyAMARES release,
which is the cost this change exists to remove.
:::

:::{dropdown} Why not a deprecation cycle first?
It would protect users with scripts calling `fit_single_fid` or holding old HDF5 results. There are
none: the maintainer is the only user, and will migrate their own scripts. A warning cycle would
cost a release and keep the old code alive for nobody. New users never see the old API.
:::

:::{attention} Assumptions to verify
- `fit_amares` reproduces our numbers on the same FID once the differing defaults are matched.
- Every function in the table has an xmris equivalent, or the gap can be filled upstream.
- `GESeries` voxel-result maps can be replaced by the Dataset without breaking `plotting`.
- The removal is breaking, so the release is a minor bump at least; the version is undecided.
:::
