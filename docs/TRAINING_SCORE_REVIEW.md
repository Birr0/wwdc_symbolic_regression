# Training-score mismatch investigation

Reviewed 2026-11-03. No production selection, equations or tolerance changed.
Evidence is in `outputs/training-score-review/` in the author checkout.

## What was tested

Rebuilt the archived cleaned sample, the training-fitted scalers, and each seed's
10,000 training rows. Evaluated all 387 candidates using `Equation_p12` with the
current evaluator; compared training scores to the same independent references
used by the production workflow. Recomputed validation scores and selections
with the existing 1e-6 tolerance. Test scores were not used in this experiment.

The initial run used the temporary downloaded-data verification copy. Diagnostic
outputs are persistent under the repository's `outputs/` directory. The scripts
now read the adjacent `inputs.json`, pointing to the persistent staged release
of the same archived inputs, for future reruns.

## Findings

17 candidates have finite training-score differences above 1e-6 in the current
six-decimal exports. Eleven fall below that threshold with the 12-decimal export:

- z, seed 42: lengths 13, 14, 17, 19, 20, 22, 23, 24, 27.
- z, seed 43: length 5.
- cond+z, seed 44: length 5.

This isolates coefficient export precision as the cause of their eligibility
failure under the current evaluator. It does not imply their original printed
expressions reproduce the native model.

For z/seed43/length5, the six-decimal export includes
`(-169032960) * (-0.000000 * X1)`. The 12-decimal export retains the tiny
coefficient `-0.000000049894`. Keeping it restores training R² from approximately
-2.37633 to the native reference 0.54411 (difference about 1.2e-8).

### One export cannot be recovered by choosing the other column

cond+z/seed43/length5 contains a coefficient near 6.914e20 multiplying a
coefficient printed as zero in BOTH exports. Both exported expressions are
constant predictions. Its reference R² is 0.54411; the 12-decimal expression
scores approximately -2.37635. The missing coefficient cannot be recovered
from those strings. Recovering it requires the native tree or a verified exact
seeded rerun with a round-trip-safe coefficient export. Do not infer a coefficient
from the reference score.

### Five equations are sensitive to arithmetic precision

z/seed44 lengths 17, 19, 20, 24 and 28 still differ by about 0.00046–0.00098
with 12-decimal coefficients. They subtract nearly equal exponential terms
with coefficients of roughly 4e5 to 1.3e6, amplifying small numerical differences.

Literal tree-order NumPy float64 evaluation agrees closely with the SymPy-based
float64 scores, so removing SymPy simplification alone does not solve these cases.
Literal float32 evaluation changes the scores substantially but still differs
from the archived references by approximately 4e-6–3e-4. This demonstrates
arithmetic sensitivity; it does NOT reproduce the original Operon evaluator or
prove the complete cause of the archived discrepancy. Native-tree evaluation
is needed to resolve the remaining differences.

## Selection impact

With 12-decimal exports and unchanged eligibility policy:

- Eligible candidates increase from 365 to 376.
- All 447 existing arm/seed/budget selections retain the same candidate length
  (candidate identity within these archived fronts). Coefficient strings change.
- Two previously absent points become selectable: z/seed43/budget5 and
  cond+z/seed44/budget5.
- Six finite training-score mismatches remain.

This is a training/validation sensitivity check, not a complete regenerated-paper
comparison. Test predictions, failure patterns and figures must be compared
before adopting higher precision in the production pipeline.

## Recommendation

Use an export that preserves coefficients for scientific evaluation and round
only for manuscript display. First recover the lost coefficient and check the
five sensitive equations against native execution. Retain the existing results
as the reference while comparing candidate eligibility, validation selections,
test failures and paper outputs. Do not simply relax the training-score tolerance:
it would mix small precision effects with a catastrophically damaged equation.

## Evidence and rerunning

From the repository root, with the analysis environment installed:

```bash
.venv/bin/python outputs/training-score-review/investigate.py
.venv/bin/python outputs/training-score-review/check_arithmetic.py
```

Local diagnostic files:

- `precision_comparison.csv`: all 387 original and higher-precision scores.
- `selection_comparison.csv`: old versus higher-precision validation selections.
- `arithmetic_comparison.csv`: five sensitive expressions under literal
  float32/float64 arithmetic and the current evaluator.
- `inputs.json`: persistent local input paths for rerunning these diagnostics.

These diagnostics are review artifacts, not additional required paper products.
