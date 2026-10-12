"""Compare corrected validation selections against the preserved historical audit."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()
    old = pd.read_csv(args.old / "selected.csv")
    new = pd.read_csv(args.new / "selected.csv")
    frozen = pd.read_csv(args.new / "selected_before_test.csv")
    pd.testing.assert_frame_equal(new[frozen.columns], frozen)
    keys = ["arm", "seed", "budget"]
    comparison = old.merge(new, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    assert len(comparison) == len(old) == len(new)
    changed = comparison[comparison.equation_old != comparison.equation_new]
    changed.to_csv(args.new / "changed_selections.csv", index=False)
    failures = new[new.test_status != "ok"]
    failures.to_csv(args.new / "test_evaluation_failures.csv", index=False)
    candidates = pd.read_csv(args.new / "candidate_scores.csv")
    # Independent ranking check: maximum validation score at each budget, then shortest.
    for row in frozen.itertuples():
        pool = candidates[(candidates.arm == row.arm) & (candidates.seed == row.seed)
                          & (candidates.length <= row.budget) & candidates.selectable]
        best = pool.sort_values(["validation_r2", "length"], ascending=[False, True]).iloc[0]
        assert row.equation == best.equation
    old_candidates = pd.read_csv(args.old / "candidate_scores.csv")
    eligibility = old_candidates.merge(candidates, on=["arm", "seed", "length"],
                                      suffixes=("_old", "_new"), validate="one_to_one")
    eligibility = eligibility[eligibility.selectable_old != eligibility.selectable_new]
    eligibility.to_csv(args.new / "changed_eligibility.csv", index=False)
    main = comparison[(comparison.arm == "cond+z") & (comparison.budget == 24)]
    assert (main.equation_old == main.equation_new).all()
    unchanged = comparison[comparison.equation_old == comparison.equation_new]
    delta = float((unchanged.test_r2_new - unchanged.test_r2_old).abs().max())
    lines = ["# Training/validation-only selection audit", "",
        f"- Candidates: {len(candidates)}; eligible: {int(candidates.selectable.sum())}.",
        f"- Changed eligibility: {len(eligibility)} candidates.",
        f"- Changed choices: {len(changed)} of {len(new)} arm/seed/budget points.",
        f"- Failed test evaluations: {len(failures)} selected budget points.",
        "- All three conditional-plus-redshift L24 interpretation equations are unchanged.",
        "- Every selection matches an independent validation ranking; frozen selections are unchanged after test evaluation.",
        f"- Maximum test-R² difference for unchanged exported equations versus historical native scores: {delta:.9g}.",
        "", "## Changed choices", "",
        "| Arm | Seed | Budget | Old length | New length | Old test R² | New test R² | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in changed.itertuples():
        value = f"{row.test_r2_new:.6f}" if np.isfinite(row.test_r2_new) else "—"
        lines.append(f"| {row.arm} | {row.seed} | {row.budget} | {row.chosen_length_old} | {row.chosen_length_new} | {row.test_r2_old:.6f} | {value} | {row.test_status} |")
    lines += ["", "Test failures are retained without replacements. Aggregate curves omit the mean and range at budgets with a failed seed, rather than averaging only successful seeds.",
        "", "The selected prediction object is the saved exported expression. Native Operon trees are used to verify training-score agreement; their validation/test evaluations can differ near restricted domains after coefficient rounding. In particular, orig seed 43 L23 is finite as an exported expression but nonfinite as a native tree on validation/test. This is retained explicitly as an export-precision limitation, not presented as native-tree reproducibility."]
    (args.new / "SELECTION_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
