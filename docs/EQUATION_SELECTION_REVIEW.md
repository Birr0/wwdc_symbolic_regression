# Equation evaluation and validation selection review

Reviewed 2026-10-12. This review leaves the numerical implementation unchanged.
The [machine-readable record](../artifacts/equation_selection_review.json)
contains source hashes, archived-result checks, and small diagnostic examples.

**Follow-up:** explicit grammar, variable, input/output shape checks and failure
reasons have now been implemented. The findings below describe the pre-change
review. SymPy simplification and coefficient precision remain unchanged.
The tests in `tests/test_equations.py` cover these contracts. Candidate outputs
now include `train_failure_reason` and `validation_failure_reason`; selected
test outputs include `test_failure_reason`. Empty reasons indicate success;
failures distinguish invalid expressions/inputs, nonfinite predictions/scores,
and numerical evaluation errors.

## Read the code in this order

1. [`equations.evaluate`](../src/wwdc_repro/equations.py) constructs `X1` through
   `Xn`, parses the saved expression using SymPy, and converts it to a NumPy
   function. `^` becomes `**`; supported explicit names include `exp`, `log`,
   `abs`, and `Abs`. Scalar constant predictions are broadcast across rows.
   Evaluation uses the `Equation` column, not the higher-precision `Equation_p12`.
2. [`frontier.score_expression`](../src/wwdc_repro/frontier.py) returns R² when
   all predictions are finite. Nonfinite predictions and the listed exceptions
   produce NaN. One invalid row invalidates that whole candidate evaluation;
   the function does not drop rows to improve the score.
3. `score_fronts` reconstructs the seed's 10,000 training rows and evaluates
   every candidate on those rows and the cleaned validation split. Eligibility
   requires finite training/reference/validation scores and absolute training
   R² agreement within `1e-6`. Historical ambiguous training flags are resolved
   using independently recovered native training scores, bound to the front
   hash and exact equation string.
4. `select_by_budget` selects the highest validation R² among eligible equations
   with length at most the budget, separately for each arm and seed. A validation
   tie prefers the shorter expression. A budget can therefore reuse an equation
   selected at a smaller budget: budget is not necessarily chosen equation length.
5. `cli.run` writes `selected_before_test.csv` before invoking `evaluate_selected`.
   The latter evaluates frozen expressions on test, retaining failed scores and
   never selecting a replacement. Repeated expressions within an arm are cached.
6. `curves` reports seed means and min–max ranges. A failed selected seed leaves
   a gap, rather than a mean over only the remaining successful seeds.

Candidate selection does not use test scores. The shared matrix builder prepares
all splits and now validates their input integrity, but test predictions are not
used in eligibility or ranking. Input-integrity checks should not be confused
with candidate-performance selection.

## Verification against the current run

The review inspected `outputs/data-validation-2026-10-12/`, the full run completed
after the data validation changes:

| Check | Result |
|---|---:|
| Saved candidate expressions | 387 |
| Eligible candidates | 365 |
| Arm/seed/budget selections independently checked | 447 |
| Failed selected test budget points | 7 |
| Distinct failed arm/seed/equation combinations | 4 |

- Eligibility matches the explicit finite-score/training-agreement formula for
  every candidate.
- Every selected validation score equals the eligible maximum at that budget;
  chosen lengths respect the shorter-expression tie rule.
- All frozen selection columns remain exactly unchanged after test evaluation.
- Aggregates with a failed seed retain NaN mean and range.
- All archived variables are within their arm's feature count; the function
  atoms found in these expressions are `Abs`, `exp`, and `log`.

The separate manuscript plotting convention is documented in
[FIGURES.md](FIGURES.md#frontier-failure-markers): hollow triangles display a
test-informed lower-budget fallback. That display never changes the selections.

## Evaluator limitations found

### Symbolic simplification affects domains

SymPy's default parsing can simplify before numerical evaluation. For inputs
`X1 = [-1, 0, 1]`, the current evaluator produces:

| Expression | Predictions |
|---|---|
| `log(X1)` | NaN, −infinity, 0 |
| `log(abs(X1))` | 0, −infinity, 0 |
| `exp(1000*X1)` | 0, 1, infinity |
| `X1/X1` | 1, 1, 1 |
| `log(X1)-log(X1)` | 0, 0, 0 |

The last two examples show that evaluation is of the SymPy-parsed exported
expression, not a literal replay of each original tree operation. Thus "rejects
nonfinite predictions" means predictions produced by this evaluator. It does
not prove that every intermediate native-tree operation is defined.

This is distinct from coefficient rounding, which can also change domains.
The examples establish the evaluator's behaviour; they do not establish an
effect on any paper result. Do not silently change simplification or switch to
12-decimal coefficients during a repository tidy-up. A change requires comparing
eligibility, choices, and numerical results under both evaluation conventions.

### Invalid expressions are not all handled uniformly

An undefined function such as `unknown(X1)` produces `NameError`, which is not
in `score_expression`'s caught exception list. An out-of-range variable can reach
the numeric conversion stage and fail there instead of receiving a clear symbol
validation error. The reviewed archive contains neither issue.

The evaluator also relies on downstream scoring for shape validity; it does not
explicitly enforce a two-dimensional feature array, a real-valued scalar/vector
expression, or an output vector of exactly one value per spectrum.

### Failure diagnostics could be more informative

Parsing errors, domain failures, and nonfinite metrics currently collapse to NaN
and `evaluation_failure`. Recording a failure reason would make independent
reproduction easier to investigate. Shape/input errors should be distinguishable
from a mathematically invalid candidate prediction.

## Recommended next changes

1. Validate variable/function names and input/output shapes before scoring, with
   clear failures for unsupported expressions. Preserve the current parsing and
   numerical convention for all archived expressions.
2. Add meaningful evaluator tests for constants, powers, logarithm domains,
   overflow, unknown symbols/functions, and the documented simplification cases.
3. Add explicit candidate failure reasons to the audit outputs, retaining the
   same eligibility and test-failure rules.
4. Verify exact unchanged candidate scores, selections and test outputs on the
   archive after these changes. Treat any change to symbolic simplification or
   coefficient precision as a separate scientific comparison.

No selection-policy correction is indicated by this review. The changes above
are about making the numerical contract and failure behaviour explicit.
