"""Paired total-mass SR ablation: remove only the explicit NORM predictor.
"""
from pathlib import Path
from wwdc_repro.shared.search_runs import add_search_options, positive_int
import argparse
import json
import shutil

import numpy as np
import pandas as pd

import wwdc_repro.norm.matched_conditioning_controls as controls
from wwdc_repro.shared.selection import report_front

OUT = Path('norm_ablation')
ARMS = ('original', 'norm')
SEEDS = (42, 43, 44)


def load_results(arms: tuple[str, ...] = ARMS) -> dict:
    """Load saved explicit-NORM results for the requested representations.

    Parameters
    ----------
    arms : tuple of str, optional
        Representations to load: ``original``, ``norm``, or both.

    Returns
    -------
    dict
        Each arm maps to its selected-equation DataFrame, candidate-audit
        DataFrame and matrix-file Path. Matrix arrays are not loaded here.

    Raises
    ------
    ValueError
        If an unsupported representation is requested.
    FileNotFoundError
        If a required file is missing or an original search is incomplete.
    """
    if set(arms) - set(ARMS):
        # check requested arm is supported
        raise ValueError(f'Unsupported representations: {set(arms) - set(ARMS)}')
    groups = {}
    for arm in arms:
        if arm == 'norm':
            selected = pd.read_csv(controls.BASE / 'validation_selected_final.csv')
            audited = pd.read_csv(controls.BASE / 'candidate_audit_final.csv')
            name = 'norm_cond+z'
            groups[arm] = (
                selected[selected.arm == name].assign(arm=arm),
                audited[audited.arm == name].assign(arm=arm),
                controls.BASE / f'{name}_matrices.npz',
            )
        else:
            runs = [controls.OUT / f'{arm}_seed{seed}' for seed in SEEDS]
            if not all((run / 'manifest.json').exists() for run in runs):
                raise FileNotFoundError(f'{arm}: not all three searches are complete')
            groups[arm] = (
                pd.concat([pd.read_csv(run / 'selected.csv') for run in runs],
                          ignore_index=True),
                pd.concat([pd.read_csv(run / 'candidate_audit.csv') for run in runs],
                          ignore_index=True),
                controls.OUT / f'{arm}_matrices.npz',
            )
    return groups


def prepare():
    OUT.mkdir(exist_ok=False)
    sources = {
        'original': controls.OUT / 'original_matrices.npz',
        'norm': controls.BASE / 'norm_cond+z_matrices.npz',
    }
    metadata = {} # metadata for details on preparation.
    reference = np.load(sources['original']) # load the original reference matrices
    for arm in ARMS:
        path = sources[arm]
        mats = np.load(path) # load matrices
        reduced = {} # for arrays with NORM column removed
        for split in ('train', 'val', 'test'):
            x, y = mats[f'X_{split}'], mats[f'y_{split}']
            assert x.shape[1] == 12 # 10 latents, NORM and z
            np.testing.assert_array_equal(y, reference[f'y_{split}'])
            np.testing.assert_array_equal(x[:, 10:], reference[f'X_{split}'][:, 10:])
            ids = np.load(controls.OUT / f'{split}_ids.npy')
            if arm == 'norm':
                # check ids
                np.testing.assert_array_equal(ids, np.load(controls.BASE / f'norm_{split}_ids.npy'))
            # Remove explicit NORM (column 10); keep the ten latents and raw redshift.
            # Redshift becomes X11 in equations fitted to the reduced inputs.
            reduced[f'X_{split}'] = x[:, list(range(10)) + [11]]
            reduced[f'y_{split}'] = y
        np.savez(OUT / f'{arm}_matrices.npz', **reduced)
        metadata[arm] = {'source_matrices': str(path), 'sha256': controls.digest(path),
                         'rows': {s: len(reduced[f'y_{s}']) for s in ('train', 'val', 'test')}}
    for seed in SEEDS:
        for name in (f'train_indices_seed{seed}.npy', 
                     f'train_ids_seed{seed}.csv'):
            shutil.copy2(controls.OUT / name, OUT / name)
        idx = np.load(OUT / f'train_indices_seed{seed}.npy')
        # check the ids match
        np.testing.assert_array_equal(idx, np.random.default_rng(seed).choice(len(reference['y_train']), 10000, replace=False))
    # record setup for experiment
    metadata['design'] = 'Total mass; original and NORM-conditioned latents; raw z retained in both; only explicit standardized log10(NORM) removed. Same rows, targets, scaling, 10k subsets, seeds and search settings. Existing +NORM results reused.'
    metadata['selection'] = 'Chosen at reporting time: training default, validation optional; frozen before test.'
    (OUT / 'preparation.json').write_text(json.dumps(metadata, indent=2))

def run(selection="train", *, threads=8, max_workers=2, restart_incomplete=False):
    """Run shared searches on prepared ablation inputs, then compare results."""
    controls.run_searches(OUT, ARMS, SEEDS, prepared_indices=True, selection=selection, threads=threads, max_workers=max_workers, restart_incomplete=restart_incomplete)
    compare_results(selection=selection)


def compare_results(selection="train"):
    rows = []
    sources = load_results(arms=ARMS)
    for arm in ARMS:
        for with_norm in (False, True):
            if with_norm:
                selected, audit, path = sources[arm]
                mats = np.load(path)
            else:
                selected = pd.concat([pd.read_csv(OUT / f'{arm}_seed{s}/selected.csv') for s in SEEDS])
                audit = pd.concat([pd.read_csv(OUT / f'{arm}_seed{s}/candidate_audit.csv') for s in SEEDS])
                mats = np.load(OUT / f'{arm}_matrices.npz')
            selected = report_front(audit, mats, selection=selection)
            selected = selected.assign(arm=arm, with_norm=with_norm)
            selected['test_rmse'] = np.sqrt((1 - selected.test_r2) * np.var(mats['y_test']))
            selected['validation_rmse'] = np.sqrt((1 - selected.val_r2) * np.var(mats['y_val']))
            rows.append(selected) # add selected row
    allrows = pd.concat(rows, ignore_index=True) # combine all four arms into a common data frame
    allrows.to_csv(OUT / 'selected_all_budgets.csv', index=False)
    summary = allrows.groupby(['arm', 'with_norm', 'budget']).agg(
        test_r2=('test_r2', 'mean'), test_r2_sd=('test_r2', 'std'),
        test_rmse=('test_rmse', 'mean'), validation_r2=('val_r2', 'mean'),
        seeds=('seed', 'nunique')).reset_index() # summarise results across seeds
    assert (summary.seeds == 3).all() # check all seeds are present as pandas can ignore missing values
    complete = allrows.groupby(['arm', 'with_norm', 'budget']).test_r2.apply(lambda v: len(v) == 3 and np.isfinite(v).all()).to_numpy()
    summary.loc[~complete, ['test_r2', 'test_r2_sd', 'test_rmse']] = np.nan
    summary['selection'] = selection
    summary.to_csv(OUT / 'summary.csv', index=False)
    print(summary[summary.budget.isin([15, 29])].to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'fit', 'run', 'compare_results'])
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--seed', type=int, choices=SEEDS)
    ap.add_argument('--selection', choices=('train', 'validation'), default='train')
    add_search_options(ap)
    ap.add_argument("--workers", type=positive_int, default=2, help="Concurrent local searches (run stage)")
    args = ap.parse_args()
    if args.stage == 'fit':
        if args.arm is None or args.seed is None:
            ap.error('fit requires --arm and --seed')
        controls.OUT = OUT
        controls.fit(args.arm, args.seed, prepared_indices=True, selection=args.selection, threads=args.threads, restart_incomplete=args.restart_incomplete)
    elif args.stage == 'prepare':
        prepare()
    elif args.stage == "run":
        run(selection=args.selection, threads=args.threads, max_workers=args.workers, restart_incomplete=args.restart_incomplete)
    else:
        compare_results(selection=args.selection)
