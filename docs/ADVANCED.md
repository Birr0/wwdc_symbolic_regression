# Advanced reproduction

The normal saved-result workflow is in [QUICKSTART.md](QUICKSTART.md).
Use this page only to regenerate embeddings or train the earlier embedding MLP
comparison. Section 6 inference, probe training and fibre MLP recipes are in
[CONDITIONING.md](CONDITIONING.md); fresh SR searches are in
[the experiment instructions](../experiments/README.md).

## Train flows or generate embeddings with the released source

Use the training walkthrough in [wwdc_spectra](https://github.com/Birr0/wwdc_spectra/blob/9349587aaa25c0210ba200b13f123e16084f66ea/docs/PAPER_TRAINING.md).
It covers `paper.py train` and `paper.py embed` for redshift and NORM,
using the full HF data bundle. The source commit is pinned in
`artifacts/conditioning/release_upstream.json`; remote source availability and the
HF upload are not yet verified.

## Historical checkpoint-inference route

The wrapper below preserves the earlier 32-test-spectrum verification. For the
new source/data release, use `paper.py embed` above.


Use the archived spectrum splits and checkpoint layout in [docs/INPUTS.md](INPUTS.md).
The wrapper rejects a changed upstream source, a different checkpoint or
configuration, and unexpected parent split counts.

```bash
git clone https://github.com/Birr0/wwdc_spectra.git /path/to/wwdc_spectra
git -C /path/to/wwdc_spectra checkout 0895b4ed9dbe6815cbe641fd1f936c6d53197aa5
uv sync --frozen --extra inference
uv run --frozen --extra inference wwdc-embed \
  --upstream /path/to/wwdc_spectra \
  --data-root /path/to/data \
  --splits test --limit 32 --batch-size 32 \
  --compare-stored /path/to/embeddings/7655991_0 \
  --outdir outputs/embedding-check
```

The command writes embeddings and compares them with saved rows by spectrum ID.
To generate all splits, omit `--splits test`, `--limit 32`, and `--compare-stored`
and use a new output directory. Full-dataset generation has not been run during
this migration; runtime and maximum error remain unmeasured. Partial outputs
are marked in `embedding_manifest.json` and cannot pass the paper sample checks.

The upstream Spender constructor may download its `sdss_I` encoder artifact
if it is not cached, before the paper checkpoint is loaded strictly. This cache
was available during verification. Downstream analysis reads local files.

## Reproduce the embedding MLP comparison

The matched `orig+z` and `cond+z` probe uses the same cleaned sample and
`lgm_tot_p50` target as the frontier. It fits scikit-learn MLPs with hidden
layers `(64, 32)` for seeds 42, 43 and 44, using 100 maximum iterations and no
early stopping. The final `fit_model_fn` definition in `malmquist_bias.ipynb`
fits the MLP on the full cleaned training split; all eligible test rows are
scored. Other MLP options use the locked scikit-learn defaults. The notebook
also contains an earlier, superseded function with a balanced 5,000-row draw;
that is not the function used by its reported MLP results.

```bash
uv run --frozen wwdc-mlp-probe \
  --embeddings /path/to/embeddings/7655991_0 \
  --galspec /path/to/galSpecExtra-dr8.fits \
  --specgals-home /path/to/sdss \
  --outdir outputs/mlp-probe
```

The command writes per-seed test scores (`scores.csv`), a paired comparison,
arm means and ranges, and `metadata.json` with settings and sample ID
fingerprints. The historical result is about 0.9103 for `orig+z` and 0.9072
for `cond+z` (a 0.0031 R² gap). The validation-selected symbolic frontier has
a larger 0.0288 R² gap on the same target.

