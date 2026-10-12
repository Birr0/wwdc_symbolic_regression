# Reproduction experiments

## Seeded symbolic-search reproduction

`rerun_sr.py` calls the historical `utils.fit_sym_fn` unchanged, preserving its
seeded 10,000-row draw, training-fitted latent scaling, raw redshift, operators,
eight threads and search budget. `utils.py` and its `processing_fun.py` import are
vendored byte-for-byte in `experiments/historical/` from the flower workspace's
`examples/spectra/symbolic_regression/`; `rerun_sr.py` refuses to run if either file's
SHA-256 differs from the version that produced the archived fronts. The search still
uses a separate Python 3.12 environment with pyoperon 0.4.0. Its direct
dependencies are pinned in the script's inline metadata, and all resolved
dependencies are recorded in `rerun_sr.py.lock`. `uv` creates this environment
automatically; the parent workspace and its virtual environment are not required.
The analysis package's `uv.lock` remains separate.

From this repository's root, first check installation without loading data:

```sh
uv run --script --locked experiments/rerun_sr.py --help
```

Then run a search against the exact archived inputs:

```sh
uv run --script --locked experiments/rerun_sr.py \
  --arm cond+z --seed 42 \
  --data-root /absolute/path/to/data \
  --outdir outputs/sr-rerun-condz-seed42
```

`--data-root` is required. Its layout is:

```text
data/
├── galSpecExtra-dr8.fits
├── sdss/
│   ├── SDSSspecgalsDR8_1.fit.gz
│   └── SDSSspecgalsDR8_2.fit.gz
└── sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0/
    ├── train/*.parquet
    ├── val/*.parquet
    └── test/*.parquet
```

See [input acquisition](../docs/INPUTS.md) for hashes and availability limits.
All catalogue files must already be present; the wrapper checks them before
loading the sample. Available arms are `orig`, `cond`, `uncond`, `z`, `orig+z`,
and `cond+z`, with archived seeds 42, 43, and 44. Quote arm names if required
by your shell. `orig+z` supplies original latents through the historical
conditional-plus-redshift route; it does not fit the conditional coordinates.

Use a new output directory for another run. Outputs include exact training IDs,
runtime versions, input/source/lockfile hashes, effective search parameters,
and freshly scored old/new fronts. `checks.json` compares ordered lengths and
equations at both archived export precisions. A mismatch produces a nonzero
exit status and retains the diagnostic outputs. The existing archived fronts
are never overwritten.

`comparison.json` retains the historical diagnostic comparison, which can use
native-tree validation scores and historical training flags. It is **not** the
current production selection policy. Use `wwdc-reproduce` for validation-only
selection and held-out evaluation of the archived exported expressions; see
[the selection audit](../docs/SELECTION_AUDIT.md).

The script lock does not guarantee identical floating-point results on every
platform. Keep the recorded runtime and exact-match checks when reporting a
rerun. To intentionally update the search environment, edit inline pins and
run `uv lock --script experiments/rerun_sr.py`, then repeat verification.

On 2026-10-12, an isolated repository copy using this script lock reproduced all
23 archived `cond+z` seed-42 equations, in order and at both export precisions,
in approximately 280 seconds. Training IDs also matched the earlier run. See
[the verification record](../artifacts/standalone_search_verification.json).
This used existing inputs on the same Linux aarch64 host.

The 2026-10-07 run completed in 279 seconds with pyoperon 0.4.0. All 23 equations
and lengths matched in order, including all 12-decimal coefficient exports.
Maximum differences from archived train/test R² were 5.44e-8 / 4.20e-8.
The selected L24 equation has validation/test R² 0.861582 / 0.859325; the best
candidate at budget 29 remains L28, with 0.870457 / 0.868169. Validation selected
the same equations as before. This verifies this arm and seed on this runtime;
it does not establish cross-platform determinism or verify every arm/seed.

The original-latent arm was checked on 2026-10-08 with the same command, replacing
`--arm cond+z` with `--arm orig` and the output directory with
`outputs/sr-rerun-orig-seed42`. It completed in 260 seconds. All 24 equations
matched, including their order, lengths and 12-decimal exports; the training IDs
matched the conditional-plus-redshift run in order. Validation selected the same
L24 equation at budget 24 (validation/test R² 0.807349 / 0.804212) and L29 equation
at budget 29 (0.811533 / 0.809466). Exact checks and runtime settings are recorded
in that output directory's `checks.json` and `manifest.json`.

## Recover training references independently of test flags

The old exporter marked both train and test scores nonfinite if either split
failed. Rerun `orig` seeds 42/43 and `orig+z` seeds 42/43/44 using the command
above, with output names `outputs/sr-rerun-orig-seed42`, etc.; use `origz` in the
directory names for the `orig+z` arm. This arm passes original latents through
the historical conditional-plus-redshift input route, exactly as the old control
did. Then run:

```sh
uv run --frozen python experiments/recover_training_scores.py
```

This requires every rerun front to match the archived lengths and expression
strings at both export precisions, and checks finite training scores within 1e-6.
Only independently evaluated native training scores replace ambiguous historical
training flags. It writes the packaged recovery artifact, including front hashes,
training-ID hashes and runtime provenance. Archived fronts remain unchanged.
The reproduction command performs training/validation eligibility and freezes
choices before evaluating the selected exported expressions on test.

Selection regression tests:

```sh
uv run --frozen python -m unittest discover -s tests -v
```

## Exploratory spectrum MLP

`spectra_mlp.py` predicts the catalogue `lgm_tot_p50` (log10 stellar mass in
solar units) directly from the processed spectrum. This is a pilot experiment,
not a paper result or a tuned comparison with the embedding representations.

- Preserve the archived train/validation/test spectrum splits and paper sample
  filters, using `wwdc_repro.data.load_sample` and `clean_target`.
- Select the exact SR 10,000-row draw using `np.random.default_rng(seed=42).choice`
  without replacement and the same parent-row order; use all
  eligible validation and test rows. Do not regroup repeated galaxies.
- Use the pinned upstream `SDSS` preprocessing: observed-frame 3,921-pixel flux
  vector normalized by median flux in the rest-frame 5300–5850 Å window.
  Supply exactly the encoder's spectrum input, without an added redshift,
  normalization factor or inverse-variance channel.
- Standardize each input pixel and the target using only the 10,000 training
  rows. Network: 3921 → hidden1 → hidden2 → 1, ReLU hidden activations.
  `--hidden 1024 512` runs the larger model; defaults are 128 and 64.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 256; at most 300
  epochs, stopping after 25 epochs without validation-MSE improvement.
- Restore the minimum-validation-MSE checkpoint before test evaluation.
- Save split IDs, predictions, training history, preprocessing statistics,
  checkpoint and R²/RMSE/MAE metrics. R² is computed on logarithmic mass.

Example from the flower root, using the environment containing the upstream
inference dependencies and scikit-learn:

```sh
.venv/bin/python -u papers/wwdc_symbolic_regression/reproduction/experiments/spectra_mlp.py \
  --upstream /tmp/wwdc-spectra-reproduction-check \
  --data-root /home/birr0/local_data \
  --outdir papers/wwdc_symbolic_regression/reproduction/outputs/spectra-mlp-sr42-1024-512 --hidden 1024 512
```

The upstream checkout used for the pilot is `Birr0/wwdc_spectra` revision
`0895b4ed9dbe6815cbe641fd1f936c6d53197aa5`. The output directory must not exist.
A single seed provides a quick baseline, not an uncertainty estimate.

The initial `outputs/spectra-mlp-10k` pilot used pandas sampling and therefore
did not use the exact SR sample. Preserve it as a historical result; use the
SR-matched outputs for comparisons. The larger run saves processed spectra.
`--prepared-from outputs/spectra-mlp-sr42-1024-512` can reuse them after checking
ordered spectrum IDs and targets against the newly reconstructed SR sample.

## Matched seed-42 results

Ordered spectrum IDs and targets match exactly between small and large runs in
all three splits. Each uses 10,000 SR training rows, 44,246 validation rows and
44,141 test rows. Independent float64 recomputation confirms the reported R².

| Hidden widths | Best validation epoch | Validation R² | Test R² | Test RMSE (dex) | Test MAE (dex) |
|---|---:|---:|---:|---:|---:|
| 128, 64 | 18 | 0.746802 | 0.666872 | 0.337441 | 0.189956 |
| 1024, 512 | 7 | 0.748621 | 0.769131 | 0.280915 | 0.189961 |

These are exploratory single-seed results with severe outliers. The larger
model's largest training residual is 144.799 dex, giving training R²=-5.188;
the ten largest residuals account for 97.9% of training squared error. Its
largest test residual is 11.399 dex. The smaller model's largest test residual
is 21.762 dex. Investigate these cases before interpreting model capacity or
representation quality. No observations were removed after inspecting errors.
Comparison and residual diagnostics: outputs/spectra-mlp-sr42-comparison.json.

## Asinh-only follow-up

Run with `--flux-transform asinh`, otherwise identical settings and cached inputs.
Transform normalized flux as asinh(f / s), then standardize pixels; s is the median
absolute pixel flux across only the 10,000 training spectra: 0.9188145399093628.
The scalar is fixed by this rule, not tuned using validation or test outcomes.
All sample IDs/targets are unchanged and checkpoint selection remains validation
MSE. Checkpoints store the transform and fitted scale alongside the scaler.

| Hidden widths | Validation R² | Test R² | Test RMSE (dex) | Test MAE (dex) |
|---|---:|---:|---:|---:|
| 128, 64 | 0.822354 | 0.818738 | 0.248912 | 0.179323 |
| 1024, 512 | 0.815878 | 0.813154 | 0.252716 | 0.183002 |

The smaller model is preferred by validation in this comparison. Outliers persist:
maximum test errors are 5.02 and 4.64 dex, and maximum training errors are 15.18
and 54.28 dex respectively. The larger model still has negative training R²
(-0.124), so input compression alone has not resolved training instability.
No mask, optimizer, learning-rate or gradient-clipping changes were made.
These are single-initialization exploratory comparisons, not fresh blind tests.
Outputs: outputs/spectra-mlp-sr42-asinh-{128-64,1024-512}; combined comparison:
outputs/spectra-mlp-sr42-asinh-comparison.json. Checkpoint reloads reproduce saved
predictions on inspected rows, including each split's worst outlier.

## Learning-rate-only follow-up (2026-10-07)

Added `--learning-rate 0.0001` to both asinh runs. All other settings, including
seed, SR sample, transform scale, optimizer, batch size, weight decay, maximum
epochs and stopping patience are unchanged. Ordered row IDs and targets match
previous runs in every split; independent float64 R² checks pass.

| Hidden widths | Best epoch | Train R² | Validation R² | Test R² | Test RMSE (dex) | Test MAE (dex) |
|---|---:|---:|---:|---:|---:|---:|
| 128, 64 | 14 | 0.912481 | 0.827877 | 0.821186 | 0.247225 | 0.177116 |
| 1024, 512 | 8 | 0.926233 | 0.844177 | 0.827355 | 0.242923 | 0.168204 |

Largest training errors fall from 15.18 to 2.87 dex (small) and 54.28 to 3.88
 dex (large). Largest test errors do not improve: 5.02 to 5.15 dex and 4.64 to
7.17 dex respectively. Lower learning rate improves aggregate validation and
test performance and reduces training extremes, but does not resolve all outliers.
The large model is preferred by validation for these fixed single-seed runs.
Repeated test reporting is exploratory, not a new blind evaluation.
Outputs: outputs/spectra-mlp-sr42-asinh-lr1e-4-{128-64,1024-512}.
Comparison: outputs/spectra-mlp-sr42-learning-rate-comparison.json.
