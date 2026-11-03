(diary-deprecate-amares-wrapper)=
# Two AMARES front ends, one engine: delete ours

<span style="color: gray; font-size: 0.9em;">Last edited: 2026-11-03 · #80</span>

`mnutils.fitting.AMARES` was ~830 lines wrapping pyAMARES. `xmris` ships its own front end,
`xmris.fitting.fit_amares`, over the same engine — batch, parallel, domain-preserving, with a
`fit_status` flag. Two wrappers meant two sets of defaults (`method`, `num_workers`, `g_global`,
`ppm_offset` all differed) and two result formats, and one of them was MNUtils-side upkeep for logic
`CLAUDE.md` says belongs upstream. Keep both and they drift apart; the question was whether to
soften the exit or just take it.

:::{important}
`GESeries` fits through `xmris.fit_amares` and returns its `xr.Dataset`; `mnutils.fitting` is deleted
outright — no deprecation cycle, no shims.
:::

(diary-deprecate-amares-wrapper-shape)=
## Shape: the series hand over FIDs, xmris does the rest

A series already carries `reference_frequency` and `carrier_ppm` on its FIDs, so the only thing a
caller supplies is `prior_knowledge`. The rest of the old surface either has a direct xmris
equivalent or went away with it.

| Old (`fitting.AMARES`) | Now |
|---|---|
| `fit_single_fid`, `fit_multiple_fids` | `fit_amares` via `fit_average_fid`, `fit_single_voxel`, `fit_all_voxels`, `fit_grouped_by_duration` |
| `extract_from_fit_results`, `FitResults` | the returned `xr.Dataset` (`fit_results` on the series) |
| `priorknowledge_to_multiindex` and inverse | `build_prior_knowledge` |
| `save/load_extracted_results` (HDF5) | none; old HDF5 files are not readable. `save_processed_data` still writes the voxel maps |
| `plot_amares_fitting`, table helpers | none; to come back upstream in xmris if wanted |
| `filter_by_ppm`, `bad_fit_chisqr_threshold` | none |

```python
mrsi = MRSISeries(DATA_FOLDER, 8)
ds = mrsi.fit_all_voxels(prior_knowledge)  # an xr.Dataset on the full (i, j, k) grid
ds["amplitude"].sel(metabolite="PCr")
ds["fit_status"]  # 0 fitted, 1 no signal, 2 failed; unfitted voxels are NaN
```

`fitted_metabolite_maps` and `goodness_of_fit_maps` survive as plain arrays derived from the
Dataset, because plotting and `save_processed_data` read them. The goodness-of-fit maps are now
`residual_rms` and `fit_status`, since xmris reports no chi-squared.

:::{warning}
`fit_all_voxels` defaults to `num_workers=-1`. xmris defaults to 1 (in-process), which took ~80 s
for 100 voxels against ~8 s on a pool. A grid is where a pool pays, so the series overrides it;
single-spectrum fits keep the xmris default.
:::

:::{dropdown} Why not keep our wrapper and just fix the defaults?
It is the cheapest option, but it leaves two front ends to keep in step with every pyAMARES release,
which is the cost this change exists to remove.
:::

:::{dropdown} Why not a deprecation cycle first?
It would protect users with scripts calling `fit_single_fid` or holding old HDF5 results. There are
none: the maintainer is the only user, and migrates their own scripts. A warning cycle would cost a
release and keep the old code alive for nobody. New users never see the old API.
:::

(diary-deprecate-amares-wrapper-changed)=
## What changed from the plan

- The plan assumed the fitting docs page could not be an executed tutorial because fits needed a
  MATLAB Engine. `fids` is now derived from the spectra, so `docs/fitting/index.md` runs on the
  synthetic exam and is the test of the new path.
- `SNR_map` still uses `pyAMARES.fidSNR`: xmris has no pre-fit SNR helper, so that one pyAMARES
  import stays (lazily, inside the property).
- Dividing an `xr.DataArray` by the normalisation factor drops its attributes, so the series
  re-attach them before fitting; without `reference_frequency` xmris cannot find `mhz`.
