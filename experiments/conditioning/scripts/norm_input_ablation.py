"""Paired total-mass SR ablation: remove only the explicit NORM predictor.

Run from this directory: norm_input_ablation.py prepare|run|report.
Existing validation-selected +NORM fits are reused without changing selection.
"""
from pathlib import Path
import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import matched_conditioning_controls as controls
from analyse_matched_conditioning import inputs
from validate_conditioning import score, select

OUT = Path('norm_input_ablation_2026-10-16')
ARMS = ('original', 'norm')
SEEDS = (42, 43, 44)


def prepare():
    OUT.mkdir(exist_ok=False)
    sources = inputs(False)
    metadata = {}
    reference = np.load(sources['original'][2])
    for arm in ARMS:
        selected, audit, path = sources[arm]
        mats = np.load(path)
        reduced = {}
        frames = {}
        for split in ('train', 'val', 'test'):
            x, y = mats[f'X_{split}'], mats[f'y_{split}']
            assert x.shape[1] == 12
            np.testing.assert_array_equal(y, reference[f'y_{split}'])
            np.testing.assert_array_equal(x[:, 10:], reference[f'X_{split}'][:, 10:])
            ids = np.load(controls.OUT / f'{split}_ids.npy')
            if arm == 'norm':
                np.testing.assert_array_equal(ids, np.load(controls.BASE / f'norm_{split}_ids.npy'))
            # Drop x11 = standardized log10(NORM); retain raw redshift as x11.
            reduced[f'X_{split}'] = x[:, list(range(10)) + [11]]
            reduced[f'y_{split}'] = y
            frames[split] = pd.DataFrame({'specObjID': ids, 'lgm_tot_p50': y,
                                         'z_x': x[:, 11]})
        np.savez(OUT / f'{arm}_matrices.npz', **reduced)
        pd.to_pickle(frames, OUT / f'{arm}_frames.pkl')
        selected.to_csv(OUT / f'{arm}_with_norm_selected.csv', index=False)
        audit.to_csv(OUT / f'{arm}_with_norm_audit.csv', index=False)
        metadata[arm] = {'source_matrices': str(path), 'sha256': controls.digest(path),
                         'rows': {s: len(reduced[f'y_{s}']) for s in frames}}
    for seed in SEEDS:
        for name in (f'train_indices_seed{seed}.npy', f'train_ids_seed{seed}.csv'):
            shutil.copy2(controls.OUT / name, OUT / name)
        idx = np.load(OUT / f'train_indices_seed{seed}.npy')
        np.testing.assert_array_equal(idx, np.random.default_rng(seed).choice(len(reference['y_train']), 10000, replace=False))
    # Verify the recorded NORM seed-42 rerun used the same search settings.
    original_params = json.loads((controls.OUT / 'original_seed42/manifest.json').read_text())['parameters']
    norm_params = json.loads((controls.BASE / 'rerun_norm_cond+z_seed42/manifest.json').read_text())['parameters']
    assert original_params == norm_params
    metadata['design'] = 'Total mass; original and NORM-conditioned latents; raw z retained in both; only explicit standardized log10(NORM) removed. Same rows, targets, scaling, 10k subsets, seeds and search settings. Existing +NORM results reused.'
    metadata['selection'] = 'Best validation R2 at each length budget; finite full train/validation and native training agreement <=1e-6; freeze before test. Native MDL recorded for new searches, not used for selection.'
    (OUT / 'preparation.json').write_text(json.dumps(metadata, indent=2))


def run():
    def task(pair):
        arm, seed = pair
        if (OUT / f'{arm}_seed{seed}/manifest.json').exists():
            return
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (OUT / f'{arm}_seed{seed}.log').open('w') as log:
            subprocess.run([sys.executable, __file__, 'fit', '--arm', arm, '--seed', str(seed)],
                           stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        print('Completed', arm, seed, flush=True)
    # Two searches, eight Operon threads each, on this 20-core machine.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(task, [(a, s) for s in SEEDS for a in ARMS]))
    report()


def report():
    rows = []
    checks = []
    sources = inputs(False)
    for arm in ARMS:
        for with_norm in (False, True):
            if with_norm:
                selected = pd.read_csv(OUT / f'{arm}_with_norm_selected.csv')
                audit = pd.read_csv(OUT / f'{arm}_with_norm_audit.csv')
                mats = np.load(sources[arm][2])
            else:
                selected = pd.concat([pd.read_csv(OUT / f'{arm}_seed{s}/selected.csv') for s in SEEDS])
                audit = pd.concat([pd.read_csv(OUT / f'{arm}_seed{s}/candidate_audit.csv') for s in SEEDS])
                mats = np.load(OUT / f'{arm}_matrices.npz')
                for seed in SEEDS:
                    frozen = pd.read_csv(OUT / f'{arm}_seed{seed}/selection_frozen_before_test.csv')
                    assert not any('test' in c for c in frozen.columns)
                    pd.testing.assert_frame_equal(
                        frozen[['budget', 'equation']].reset_index(drop=True),
                        selected.loc[selected.seed == seed, ['budget', 'equation']].reset_index(drop=True))
            for r in selected.itertuples():
                chosen = select(audit[audit.seed == r.seed], r.budget)
                assert chosen.equation == r.equation
                actual = score(r.equation, mats['X_test'], mats['y_test'])
                np.testing.assert_allclose(actual, r.test_r2, atol=1e-10, rtol=0, equal_nan=True)
            selected = selected.assign(arm=arm, with_norm=with_norm)
            selected['test_rmse'] = np.sqrt((1 - selected.test_r2) * np.var(mats['y_test']))
            selected['validation_rmse'] = np.sqrt((1 - selected.val_r2) * np.var(mats['y_val']))
            rows.append(selected)
            checks.append(dict(arm=arm, with_norm=with_norm, candidates=len(audit),
                               eligible=int(audit.selectable.sum()),
                               selected_test_failures=int(selected.test_r2.isna().sum())))
    allrows = pd.concat(rows, ignore_index=True)
    allrows.to_csv(OUT / 'selected_all_budgets.csv', index=False)
    summary = allrows.groupby(['arm', 'with_norm', 'budget']).agg(
        test_r2=('test_r2', 'mean'), test_r2_sd=('test_r2', 'std'),
        test_rmse=('test_rmse', 'mean'), validation_r2=('val_r2', 'mean'),
        seeds=('seed', 'nunique')).reset_index()
    assert (summary.seeds == 3).all()
    summary.to_csv(OUT / 'summary.csv', index=False)
    paired = allrows[allrows.with_norm].merge(allrows[~allrows.with_norm],
        on=['arm', 'seed', 'budget'], suffixes=('_with', '_without'), validate='one_to_one')
    paired['delta_test_r2'] = paired.test_r2_with - paired.test_r2_without
    paired['test_rmse_reduction'] = paired.test_rmse_without - paired.test_rmse_with
    paired.to_csv(OUT / 'paired_differences.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, arm in zip(axes, ARMS):
        for flag, style in ((False, '--'), (True, '-')):
            s = summary[(summary.arm == arm) & (summary.with_norm == flag)]
            ax.plot(s.budget, s.test_r2, style, label='Latents + z' + (' + NORM' if flag else ''))
            ax.fill_between(s.budget, s.test_r2-s.test_r2_sd, s.test_r2+s.test_r2_sd, alpha=.15)
        ax.set(title='Original' if arm == 'original' else 'NORM-conditioned',
               xlabel='Expression-length budget', ylabel='Mean test $R^2$')
        ax.legend()
    fig.savefig(OUT / 'norm_input_ablation.pdf')
    fig.savefig(OUT / 'norm_input_ablation.png', dpi=300)
    plt.close(fig)
    lines = ['# Explicit NORM input ablation: total stellar mass', '',
        'Paired comparisons retain redshift in every fit. The only input removed is standardized log10(NORM). '
        'Original and NORM-conditioned representations use the same galaxies, target, existing train/validation/test '
        'splits and 10,000 training rows for each of seeds 42, 43 and 44. Latent standardization is unchanged. '
        'Search trees are limited to 25 nodes (up to 29 including export scale/offset); all other recorded search '
        'settings are retained. Existing +NORM searches are reused; the six without-NORM searches are new.', '',
        'Equations are selected by validation R² within each length budget, with finite training/validation '
        'predictions and native/export training-score agreement. Test scores are evaluated after selection; '
        'no replacement is selected using test performance. Native MDL is recorded for new fits, not used to select.', '',
        '| Representation | Budget | Without NORM test R² | With NORM test R² | ΔR² | RMSE without / with [dex] |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for budget in (12, 15, 29):
            s = summary[(summary.arm == arm) & (summary.budget == budget)].set_index('with_norm')
            a, b = s.loc[False], s.loc[True]
            lines.append(f'| {arm} | {budget} | {a.test_r2:.6f} | {b.test_r2:.6f} | {b.test_r2-a.test_r2:+.6f} | {a.test_rmse:.6f} / {b.test_rmse:.6f} |')
    lines += ['', '## Paired changes by seed', '',
              '| Representation | Seed | Budget | Δ test R² (with − without NORM) |',
              '|---|---:|---:|---:|']
    for r in paired[paired.budget.isin([15, 29])].sort_values(['arm', 'seed', 'budget']).itertuples():
        lines.append(f'| {r.arm} | {r.seed} | {r.budget} | {r.delta_test_r2:+.6f} |')
    lines += ['', 'Values are means over three search seeds. The plot shows ± one seed standard deviation, '
        'not an uncertainty interval over independent galaxy samples or flow trainings. Budget denotes a '
        'maximum allowed expression length; the validation-selected expression can be shorter.', '',
        'This tests the predictive utility of supplying NORM explicitly, conditional on already supplying z. '
        'It does not itself show that conditioning improves accuracy or establish a physical interpretation of NORM. '
        'For the NORM-conditioned representation, adding NORM also restores conditioning information needed '
        'to invert the conditional transformation. All numbers describe this finite SR search budget.', '',
        'Reproduce from `examples/spectra/symbolic_regression` using the project virtual environment:', '',
        '```bash', '../../../.venv/bin/python norm_input_ablation.py prepare',
        '../../../.venv/bin/python norm_input_ablation.py run', '```', '',
        'Preparation refuses to overwrite existing inputs. `run` skips completed fits. '
        'Use `report` to regenerate the report from saved results. See `preparation.json`, per-run '
        '`manifest.json`, `candidate_audit.csv`, and `selection_frozen_before_test.csv` for provenance and selection.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    (OUT / 'verification.json').write_text(json.dumps(checks, indent=2))
    print(summary[summary.budget.isin([15, 29])].to_string(index=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'fit', 'run', 'report'])
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--seed', type=int, choices=SEEDS)
    args = ap.parse_args()
    if args.stage == 'fit':
        if args.arm is None or args.seed is None:
            ap.error('fit requires --arm and --seed')
        controls.OUT = OUT
        controls.fit(args.arm, args.seed, prepared_indices=True)
    else:
        globals()[args.stage]()
