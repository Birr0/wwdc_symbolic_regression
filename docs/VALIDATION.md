# Migration verification, 2026-10-05

## Review release verification (2026-10-12)

The evaluator now validates its supported grammar, variables and array shapes,
and records failure reasons. All pre-existing columns of candidate scores,
frozen selections, test evaluations and aggregate curves match the preceding
full run exactly. All 15 reference checks pass; 365/387 candidates are eligible.
See [the numerical report](../artifacts/evaluator_release_verification.json).

A clean copy of the proposed release installed from `uv.lock`, passed all 14
tests, loaded the analysis/figure/MLP command help, provisioned the separately
locked SR command, and built both wheel and source distribution. Required JSON
resources are in the wheel; the source distribution includes search code/lock,
the historical MIT notice, documentation and tests. See
[the clean-copy report](../artifacts/release_clean_check.json).

The GitHub Actions workflow runs these lightweight checks without paper datasets.
It has not yet been run on GitHub. This local check used the same host and existing
package caches; it does not demonstrate public input availability.

## Data validation follow-up (2026-10-12)

Explicit input checks, ordered spectrum-ID references, and intermediate sample
counts were added without changing the archived numerical inputs. Exact
before/after comparisons passed for all three cleaned DataFrames, all six arms'
feature/target matrices, scaler parameters, and the three seeded training draws.
The full saved-equation workflow passed 15 reference checks with 365/387 eligible
candidates. Frozen selections, evaluated selections and aggregate curves match
the preceding validation-only run exactly after normalising invocation-relative
source-path prefixes. Nine unit tests pass.

See [the data review](DATA_REVIEW.md) and
[verification artifact](../artifacts/data_validation_verification.json).
Local full outputs: `outputs/data-validation-2026-10-12/`.

## Standalone SR search environment (2026-10-12)

An isolated copy containing only this repository's source, experiments,
artifacts, and tests ran `experiments/rerun_sr.py` through its separate script
lockfile. No parent-workspace virtual environment was used. The `cond+z`,
seed-42 search reproduced all 23 archived equations in order, their lengths,
and their 12-decimal exports. Training IDs matched the earlier verified run;
sample reference and fit-input array checks passed. Runtime was approximately
280 seconds on the same Linux aarch64 host.

The [verification artifact](../artifacts/standalone_search_verification.json)
records input/source/lockfile hashes, runtime versions, and checks. Full local
outputs are under `outputs/standalone-search-2026-10-12-condz-seed42/` (ignored
by Git). CLI checks also confirmed missing arguments/data fail without creating
outputs, and the three existing selection tests pass.

This verifies the standalone search environment for one arm/seed with existing
local inputs. It does not establish public input acquisition or cross-platform
determinism. See [the exact command](../experiments/README.md#seeded-symbolic-search-reproduction).

## Historical migration and selection correction

The historical checks below describe the original migration. The 2026-10-08
selection correction supersedes its 447-identical-selections result: see
[SELECTION_AUDIT.md](SELECTION_AUDIT.md). Training/validation eligibility now
uses recovered native training references, choices are frozen before test
evaluation, and failures are retained without replacements. All 12 updated
reference checks and three selection regression tests pass. Five affected
arm/seed fronts matched exactly at both exported coefficient precisions.

The corrected run has 365 eligible candidates (previously 360), 13 changed
budget-level choices and seven failed test evaluations across four distinct
exported equations. All cond+z L24 interpretation equations and all budget-29
choices are unchanged. Native-tree and exported-expression domain validity can
differ after rounding; reported selection and test evaluation consistently use
the exported expressions. The reference eligibility count was independently
recalculated from cached training/validation scores and native recovery records,
then checked against the fresh full run.

The standalone implementation was compared with the saved pre-submission
audit and, for score/anatomy, the original inspected plotting functions.
All runs used the same local data. These are numerical reproduction checks,
not independent validation of the underlying scientific interpretations.

| Check | Result |
| --- | --- |
| Selection | All 447 selected equation strings and saved test scores match the audited validation-selected results |
| Samples | Exact train/validation/test spectrum-ID hashes and counts; 12 reference checks pass |
| Zero-latent figure | All 400 exported grid rows match the audited numerical CSV exactly |
| Raw latent histograms | Entire moments JSON matches the audited recreation exactly |
| Score projection | Maximum per-row difference across three seeds 3.56e-15 |
| Anatomy | Maximum baseline difference 1.78e-15; prediction-SD difference below 1e-16 |
| CMD | All 440,140 rows match by spectrum ID; magnitudes and colours identical; score difference below 9e-16 |
| CMD cache rebuild | All 525,505 rows, all columns and all duplicate-match choices match the old cache |
| Checkpoint inference | Clean pinned upstream source, locked inference extra, first 32 test spectra; original max error 9.88e-7, conditional 1.884e-5, unconditional 1.419e-5; redshift/mask ratio exact |

The tested checkpoint tolerances are 2e-6 for original and 2e-5 for conditional
and unconditional embeddings. They are numerical tolerances, not a guarantee
for every spectrum or hardware platform. [The compact inference report](../artifacts/embedding_check_32.json)
records the environment, input fingerprints and observed errors.

Figure images were inspected for axis labels, colour scales and layout.
Styling differs from the historical manuscript figures; its numerical recipes
and the known latent-distribution recreation are retained. No new figure was
automatically copied into the paper.

The clean-clone run passed at code commit
`aeb853fc890f7f1dee7a79036ac4a4f0fbfe315a`: installation from the lockfile,
12 analysis reference checks, the cache rebuild, all seven PNG/PDF figure pairs,
and the 32-spectrum checkpoint comparison. The machine-readable result is
[artifacts/verification.json](../artifacts/verification.json).

## Embedding MLP probe (2026-10-12)

The new `wwdc-mlp-probe` command was run on the same archived embeddings and
catalogues with the locked standalone environment. Across seeds 42, 43, and 44,
the mean test R² was 0.910293518 for `orig+z` and 0.907162251 for `cond+z`;
the gap was 0.003131266. Both means agree with the archived
`examples/spectra/symbolic_regression/probe_arms_results/probe_arms.csv` to
within 1.2e-9. The two arms use the same 352,896 cleaned training rows and
44,141 test rows.

The active, final `fit_model_fn` in `malmquist_bias.ipynb` fits its MLP on the
full cleaned training split. A later stand-alone `probe_arms.py` defaults to a
5,000-row redshift-balanced draw; reproducing that default yields different
scores (0.838178 and 0.794626 in this environment). The reproduction command
follows the notebook fit that produced the archived probe values.

This used a new virtual environment on the same host, existing external data,
and the cached Spender constructor artifact. It is not a second-machine or
public-download verification. Full inference across all spectra remains open.
