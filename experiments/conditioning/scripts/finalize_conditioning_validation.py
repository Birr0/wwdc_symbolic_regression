"""Apply verified rerun exports, freeze validation selections, then report tests."""
from pathlib import Path
import hashlib
import json
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_conditioning import score, select

OUT = Path('conditioning_validation_2026-10-13')
LABELS = {
    'norm_cond': 'NORM flow: latents + NORM',
    'norm_cond+z': 'NORM flow: latents + NORM + z',
    'znorm_cond+z': 'z+NORM flow: latents + NORM + z',
    'z_cond': 'z flow: latents',
    'z_cond+z': 'z flow: latents + z',
}

def main():
    a = pd.read_csv(OUT / 'candidate_audit.csv')
    a['recovered_export'] = False
    recoveries = []
    for arm in ('norm_cond', 'norm_cond+z'):
        directory = OUT / f'rerun_{arm}_seed42'
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not manifest['exact_front_match']:
            raise RuntimeError(f'{directory}: rerun did not reproduce the saved front')
        oldpath = Path(a.loc[(a.arm == arm) & (a.seed == 42), 'source'].iloc[0])
        assert hashlib.sha256(oldpath.read_bytes()).hexdigest() == manifest['original_sha256']
        fresh = pd.read_csv(directory / 'native_front.csv')
        mats = np.load(OUT / f'{arm}_matrices.npz')
        idx = np.random.default_rng(42).choice(len(mats['X_train']), 10000, replace=False)
        for i in a.index[(a.arm == arm) & (a.seed == 42)]:
            r = fresh.iloc[int(a.loc[i, 'row'])]
            assert int(r.Length) == int(a.loc[i, 'Length'])
            eq = r.Equation_p60
            tr = score(eq, mats['X_train'][idx], mats['y_train'][idx])
            va = score(eq, mats['X_val'], mats['y_val'])
            full = score(eq, mats['X_train'], mats['y_train'])
            delta = abs(tr - r.Train_R2)
            ok = bool(np.isfinite(tr) and np.isfinite(r.Train_R2) and delta <= 1e-6 and np.isfinite(va) and np.isfinite(full))
            for k, v in dict(equation=eq, train_r2=tr, train_r2_stored=r.Train_R2,
                             train_delta=delta, train_full_finite=bool(np.isfinite(full)),
                             val_r2=va, selectable=ok, recovered_export=True).items():
                a.loc[i, k] = v
        recoveries.append(manifest)
    a.to_csv(OUT / 'candidate_audit_final.csv', index=False)
    frozen = []
    for (arm, seed), group in a.groupby(['arm', 'seed'], sort=False):
        for budget in range(5, 30):
            r = select(group, budget)
            if r is not None:
                frozen.append(dict(r.to_dict(), budget=budget))
    frozen = pd.DataFrame(frozen)
    # This file has no test columns and is written before test evaluation.
    frozen.to_csv(OUT / 'selection_frozen_before_test.csv', index=False)
    test_ids = {flow: np.load(OUT / f'{flow}_test_ids.npy') for flow in ('norm', 'znorm', 'z')}
    common = set.intersection(*(set(ids) for ids in test_ids.values()))
    result = []
    for arm, group in frozen.groupby('arm', sort=False):
        mats = np.load(OUT / f'{arm}_matrices.npz')
        flow = arm.split('_')[0]
        mask = np.isin(test_ids[flow], list(common))
        for _, r in group.iterrows():
            test = score(r.equation, mats['X_test'], mats['y_test'])
            matched = score(r.equation, mats['X_test'][mask], mats['y_test'][mask])
            result.append(dict(r.to_dict(), test_r2=test, common_test_r2=matched,
                               test_status='ok' if np.isfinite(test) else 'evaluation_failure',
                               test_rows=len(mask), common_test_rows=int(mask.sum())))
    result = pd.DataFrame(result)
    result.to_csv(OUT / 'validation_selected_final.csv', index=False)
    final = result[result.budget == 29]
    final.to_csv(OUT / 'selected_equations_budget29.csv', index=False)
    audit_manifest = {
        'selection_policy': 'finite full-training and validation predictions; native fit-subsample train R2 agreement <= 1e-6; maximise validation R2 within budget; no test gating',
        'reruns': recoveries,
        'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scikit-learn', 'sympy', 'pyoperon')},
        'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            *[Path(p) for p in sorted(a.source.unique())],
            Path('validate_conditioning.py'), Path('rerun_conditioning_exports.py'),
            Path(__file__), Path('val_rescore.py'),
            OUT / 'selection_frozen_before_test.csv', OUT / 'manifest.json']},
    }
    (OUT / 'selection_manifest.json').write_text(json.dumps(audit_manifest, indent=2))
    def complete_mean(s):
        return float(s.mean()) if len(s) == 3 and np.isfinite(s).all() else np.nan
    curves = result.groupby(['arm', 'budget']).agg(
        mean_test_r2=('test_r2', complete_mean), min_test_r2=('test_r2', 'min'),
        max_test_r2=('test_r2', 'max'), selected_seeds=('seed', 'nunique'),
        finite_tests=('test_r2', 'count'), common_mean_test_r2=('common_test_r2', complete_mean)).reset_index()
    curves.to_csv(OUT / 'frontier_summary.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for arm, label in LABELS.items():
        c = curves[curves.arm == arm]
        line, = ax.plot(c.budget, c.mean_test_r2, label=label)
        ax.fill_between(c.budget, c.min_test_r2, c.max_test_r2, color=line.get_color(), alpha=.12)
    ax.set(xlabel='Expression-length budget', ylabel='Test $R^2$ (validation-selected)')
    ax.legend(fontsize=9)
    fig.savefig(OUT / 'validation_frontiers.pdf')
    fig.savefig(OUT / 'validation_frontiers.png', dpi=300)
    plt.close(fig)
    lines = ['# Conditioning comparison: validation selection', '',
        'Selection uses the largest validation R² at each length budget (5–29), separately per seed; ties prefer shorter equations.',
        'Eligibility requires finite predictions on the full training and validation splits and agreement with the native training-subsample R² within 1e-6.',
        'Only the 10,000 training rows used for each search establish score reproduction. Scaling uses the full filtered training split.',
        'Test predictions are evaluated after selection. Test failures are retained as failures, without replacement; incomplete three-seed means are not reported.', '',
        '## Budget 29', '', '| Flow and SR inputs | Seed 42 | Seed 43 | Seed 44 | Mean test R² | Common-test mean |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, label in LABELS.items():
        sub = final[final.arm == arm].sort_values('seed')
        vals = [f'{r.test_r2:.6f} (L={r.Length})' for r in sub.itertuples()]
        lines.append('| ' + ' | '.join([label, *vals, f'{complete_mean(sub.test_r2):.6f}', f'{complete_mean(sub.common_test_r2):.6f}']) + ' |')
    lines += ['', '## Provenance and limits', '',
        '- Original search CSVs are unchanged. `candidate_audit_final.csv` contains eligibility and numerical checks for every candidate.',
        '- `selection_frozen_before_test.csv` records decisions without test scores; `validation_selected_final.csv` adds test evaluation.',
        '- Two NORM-flow seed-42 searches were rerun through the original `utils.fit_sym_fn`. Their complete fronts must match both original printed precisions before training-score recovery is accepted.',
        '- Reruns export 60 decimal places to retain small float32 constants. Training-score agreement is checked again on those expressions.',
        '- Other arms use the highest available saved precision. Any unreproducible saved expression remains explicitly ineligible; selection is over the verified exported candidate pool.',
        '- The NORM and joint-flow datasets have identical ordered object IDs in all splits. The historical redshift baseline has 41 extra training rows and six extra validation/test rows. See `manifest.json` for counts and ordered-ID hashes.',
        '- Common-test scores evaluate the same frozen equations on shared object IDs. They do not make the different historical training subsamples identical.',
        '- The redshift baseline lacks NORM as an SR input. Its comparison with the new arms therefore changes both representation and available inputs; improvements cannot be attributed solely to flow conditioning.',
        '- These are three SR search seeds, not three independently trained flow checkpoints. The figure shading spans their minimum and maximum test scores.',
        '- Validation chooses models within the archived front; it does not recover candidates discarded during the original search.', '',
        '## Selected equations', '', 'X1–X10 are training-standardised conditional latents. For NORM and joint-flow arms, X11 is the appended conditioning normalisation after the SR training scaler, and X12 is raw redshift when present. For the redshift baseline with appended redshift, X11 is raw redshift.', '',
        'Full-precision expressions are in `selected_equations_budget29.csv`; use that file for evaluation rather than rounded presentation equations.', '',
        '## Reproduction commands', '',
        'From `examples/spectra/symbolic_regression`, using the flower `.venv`:', '',
        '```bash',
        'python validate_conditioning.py',
        'python rerun_conditioning_exports.py',
        'python finalize_conditioning_validation.py',
        'python -m unittest test_validate_conditioning -v',
        '```', '',
        'The rerun script refuses to overwrite its existing rerun directories. Archive the output directory before repeating the complete sequence. The analysis uses local data paths declared in `validate_conditioning.py`; saved matrices and ID arrays support repeating selection without loading catalogues again.',
        'Package versions and hashes of the original fronts, scoring code, selection file and data manifest are recorded in `selection_manifest.json`.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:16]))

if __name__ == '__main__':
    main()
