# Training/validation-only selection audit

- Candidates: 387; eligible: 365.
- Changed eligibility: 5 candidates.
- Changed choices: 13 of 447 arm/seed/budget points.
- Failed test evaluations: 7 selected budget points.
- All three conditional-plus-redshift L24 interpretation equations are unchanged.
- Every selection matches an independent validation ranking; frozen selections are unchanged after test evaluation.
- Maximum test-R² difference for unchanged exported equations versus historical native scores: 1.04891522e-06.

## Changed choices

| Arm | Seed | Budget | Old length | New length | Old test R² | New test R² | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| orig | 43 | 23 | 21 | 23 | 0.799980 | 0.805146 | ok |
| orig | 43 | 24 | 24 | 23 | 0.803513 | 0.805146 | ok |
| orig | 43 | 25 | 24 | 23 | 0.803513 | 0.805146 | ok |
| orig | 43 | 26 | 24 | 23 | 0.803513 | 0.805146 | ok |
| orig | 43 | 27 | 24 | 23 | 0.803513 | 0.805146 | ok |
| orig | 43 | 28 | 28 | 23 | 0.803528 | 0.805146 | ok |
| orig+z | 42 | 12 | 11 | 12 | 0.857188 | — | evaluation_failure |
| orig+z | 42 | 13 | 11 | 12 | 0.857188 | — | evaluation_failure |
| orig+z | 43 | 22 | 21 | 22 | 0.889214 | — | evaluation_failure |
| orig+z | 44 | 10 | 8 | 10 | 0.840963 | — | evaluation_failure |
| orig+z | 44 | 11 | 8 | 10 | 0.840963 | — | evaluation_failure |
| orig+z | 44 | 12 | 8 | 12 | 0.840963 | — | evaluation_failure |
| orig+z | 44 | 13 | 8 | 12 | 0.840963 | — | evaluation_failure |

Test failures are retained without replacements. Aggregate curves omit the mean and range at budgets with a failed seed, rather than averaging only successful seeds.

The selected prediction object is the saved exported expression. Native Operon trees are used to verify training-score agreement; their validation/test evaluations can differ near restricted domains after coefficient rounding. In particular, orig seed 43 L23 is finite as an exported expression but nonfinite as a native tree on validation/test. This is retained explicitly as an export-precision limitation, not presented as native-tree reproducibility.
