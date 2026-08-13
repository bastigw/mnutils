(fitting-index)=
# Fitting

Overview page for `fitting/AMARES.py`, MNUtils's wrapper around
[`xmris`](https://github.com/andrewendlinger/xmris)/pyAMARES fitting. Content to be written.

:::{note}
Every fitting path (`MRSSeries.fit_average_fid`, `MRSISeries.fit_all_voxels`/`fit_single_voxel`)
goes through `RawMRISeries.fids`, which calls `mnutils.utils.data_loaders.load_raw_fids` — a
raw-FID reload that requires a real MATLAB Engine connection (see `CLAUDE.md` § Gotchas). Unlike
the rest of the docs chapters, this one can't be written as an always-executed tutorial without
either requiring MATLAB for every docs build/reader, or the fitting code changing to not need it.
`tests/test_amares_fitting.ipynb` has a real, hand-verified walkthrough (unlocalised + localised
AMARES fits on `HeVo-18`) that's kept as a plain notebook for exactly this reason — see
`CLAUDE.md` § Gotchas and the "some tests can't be converted" decision it documents.
:::
