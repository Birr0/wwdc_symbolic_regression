# Paper reproduction experiments

See the [script guide](../docs/SCRIPTS.md) for every retained file and its paper role.
For Section 6 and the fibre appendix, start with the
[conditioning walkthrough](../docs/CONDITIONING.md).

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
