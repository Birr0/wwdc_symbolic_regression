# Inputs and acquisition

`artifacts/analysis_inputs.json` records the byte sizes and SHA-256 hashes of
the exact local embedding shards and analysis catalogues used in the audit.
Paths are relative to a data root. `artifacts/fronts_manifest.json` covers the
18 candidate CSVs included with this repository. Run manifests record the
actual consumed inputs; checksums identify a snapshot rather than guaranteeing
the astrophysical correctness of that source.

## Released model products and spectrum splits

The upstream project is [Birr0/wwdc_spectra](https://github.com/Birr0/wwdc_spectra),
pinned here to `0895b4ed9dbe6815cbe641fd1f936c6d53197aa5`. Its training entry
point is `src/wwdc_spectra/training/train.py`; inference is
`src/wwdc_spectra/inference/embed.py`. See its README and `src/conf/` for
training configuration. This project uses its `SDSS`, `PretrainedSpender`,
`LightningFlowMatching`, and ODE solver implementation directly.

The exact paper checkpoint is `7655991_0.ckpt`, SHA-256
`92d9aaf21e8523c1c45ee7b61c39ec53ea6235c308907e5a84475f1b7ff19604`.
Its YAML companion has SHA-256
`e1e7c16ac99eb87fe9914554233fe9f7ebab04e9e93247af6bd916da8353bb3a`.
Place them under `sdss_II/spender_I_flow_v2/spender_I_flow_v2/ckpts/`.
Inference uses ten latent dimensions, hidden dimension 1024, midpoint integration
with 20 steps (0.05 step size), and conditional/unconditional guidance 1/0.

The spectrum dataset must be the archived `sdss/sdss_II_catalog` DatasetDict,
with 422,481 / 52,810 / 52,811 parent rows. The pinned upstream fallback loader
uses `test_size=0.1` for its second random split, which does not reconstruct
these 80:10:10 archives. The wrapper requires the saved dataset and never invokes
that fallback. A seed and a fresh raw download alone do not reproduce the
original membership. Original spectra come from the
[Multimodal Universe SDSS pipeline](https://github.com/MultimodalUniverse/MultimodalUniverse/tree/main/scripts/sdss).

Public download locations for the exact checkpoint, its YAML, archived split
dataset, and cached embeddings still need to be assigned. They are not bundled
in Git. Until then, the demonstrated workflow assumes access to those files.

## Analysis catalogues

The astroML files are
[SDSSspecgalsDR8_1.fit.gz](https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_1.fit.gz)
and [SDSSspecgalsDR8_2.fit.gz](https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_2.fit.gz).
Put both under `sdss/` and compare their hashes with `analysis_inputs.json`.
The URLs point at a moving branch, so the hashes define the expected versions.

Obtain `galSpecExtra-dr8.fits` through the official
[SDSS DR8 galSpec data page](https://www.sdss3.org/dr8/spectro/galspec.php)
and place it at the data-root top level. Use the DR8 file with the recorded
hash, not a newer catalogue with a similar name. The target is `lgm_tot_p50`.

## CMD cache

The reproduction input is explicitly the pinned mirror snapshots. Their
upstream photometric release and original crossmatch construction remain
unresolved. Download them using the optional catalogue dependency:

```bash
uv sync --frozen --extra catalogues
uv run --frozen --extra catalogues wwdc-download-catalogues --outdir data
uv run --frozen wwdc-cmd-cache \
  --crossmatch-dir data/vac-cross --photometry-dir data/vac-photometry \
  --out data/vagc_join.parquet
```

The builder verifies all five shard hashes before joining on `VAC_ID`. It
preserves source-row order in the explicit shard order recorded in
`src/wwdc_repro/catalogue_sources.json`, then retains the first match per
`specObjID`. This is the historical paper rule: 13 IDs have multiple candidates,
and the resulting cache has 525,505 unique spectrum IDs. Seven ambiguous IDs
occur in the plotted sample; removing them was previously found to have a
negligible global effect. Changing match or quality rules would create a new
analysis, so that should be reported separately.

The rebuilt cache was compared across every row and column with the historical
cache and matched exactly in values. Parquet byte hashes can differ across
writer versions; its manifest records the new output hash. The historical cache
hash is `910bffdc9d2c16af1487d6e124c661a0ed720b2ea4c25655d8e97181b51d9548`.
