"""Compare original and conditional representations fitted to fibre-mass using training-default or optional validation selection, reported on test."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sympy as sp
from wwdc_repro.fibre_mass.fibre_mass_conditioning import (
    OUT,
    CONDITIONED_ARMS,
)  # directory containing the saved fibre mass equations
from wwdc_repro.shared.equations import to_sympy
from wwdc_repro.shared.selection import report_front

DEST = OUT / "original_control_comparison"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', choices=('train', 'validation'), default='train')
    parser.add_argument('--outdir', type=Path, default=DEST,
                        help='Destination for the selected equations and performance summary')
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    selections = []
    train_indices = {seed: np.load(OUT / f"train_indices_seed{seed}.npy")
                     for seed in (42, 43, 44)}
    for arm in ('original', *CONDITIONED_ARMS):
        audits = []
        for seed in (42, 43, 44):
            path = OUT / f"{arm}_seed{seed}"
            manifest = json.loads((path / 'manifest.json').read_text())
            reference = json.loads((OUT / f"znorm_seed{seed}/manifest.json").read_text())
            if manifest['parameters'] != reference['parameters']:
                raise ValueError(f'Search parameters differ for {arm}, seed {seed}')
            audits.append(pd.read_csv(path / 'candidate_audit.csv'))
        with np.load(OUT / f'{arm}_matrices.npz') as data:
            selections.append(report_front(pd.concat(audits, ignore_index=True), data,
                                           selection=args.selection, train_indices=train_indices))
    combined = pd.concat(selections, ignore_index=True)
    combined.to_csv(args.outdir / 'selected.csv', index=False)
    mats = np.load(OUT / "original_matrices.npz") # load the saved matrices.
    combined["test_rmse"] = np.sqrt((1 - combined.test_r2) * np.var(mats["y_test"])) 
    combined["validation_rmse"] = np.sqrt((1 - combined.val_r2) * np.var(mats["y_val"])) # compute rmse for each val and test
    summary = (
        combined.groupby(["arm", "budget"])
        .agg(
            test_r2=(
                "test_r2",
                lambda v: v.mean() if len(v) == 3 and v.notna().all() else np.nan,
            ),
            test_rmse=("test_rmse", "mean"),
            validation_rmse=("validation_rmse", "mean"),
            min_r2=("test_r2", "min"),
            max_r2=("test_r2", "max"),
        )
        .reset_index()
    ) # create summary of performance
    complete = combined.groupby(["arm", "budget"]).test_r2.apply(lambda v: len(v) == 3 and np.isfinite(v).all()).to_numpy()
    summary.loc[~complete, ["test_r2", "test_rmse", "min_r2", "max_r2"]] = np.nan
    summary["selection"] = args.selection
    summary.to_csv(args.outdir / "performance.csv", index=False)
    # Report the latent structure of the equations
    structure = []
    for arm in ("original", "norm"):
        chosen = combined[(combined.arm == arm) & (combined.budget == 15)]
        assert len(chosen) == 3 and set(chosen.seed) == {42, 43, 44}
        for row in chosen.itertuples():
            expression = to_sympy(row.equation, 12)
            expression = expression.xreplace(
                {v: sp.Symbol(v.name, real=True) for v in expression.free_symbols}
            )
            expression = expression.xreplace(
                {v: sp.Rational(str(v)) for v in expression.atoms(sp.Float)}
            )
            latents = {sp.Symbol(f"X{i}", real=True) for i in range(1, 11)}
            active = expression.free_symbols & latents
            derivatives = [sp.diff(expression, v) for v in active] # check the contribution of the latent sum via its derivatives.
            structure.append(
                dict(
                    arm=arm,
                    seed=row.seed,
                    budget=15,
                    length=row.Length,
                    additive_linear_latents=bool(active)
                    and all(not derivative.free_symbols for derivative in derivatives),
                )
            )
    print(summary[summary.budget.isin([15, 29])].to_string(index=False), flush=True)
    print(pd.DataFrame(structure).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
