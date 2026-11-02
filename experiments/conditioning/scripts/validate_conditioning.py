"""Audit saved fronts; select by validation before evaluating selected test predictions.

Run with the analysis environment from this directory. Outputs never overwrite searches.
"""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from val_rescore import build_merged, build_matrices, evaluate_equation

ROOT = Path(__import__('os').environ.get('DATA_ROOT', 'data'))
SPECS = {
    'norm': ('sdss_II/spender_I_normalisation_flow/embeddings/norm_20261012183217_0', 'normflow_condnorm_', True),
    'znorm': ('sdss_II/spender_I_z_normalisation_flow/embeddings/znorm_20261013104946_0', 'znormflow_condnorm_', True),
    'z': ('sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0', '', False),
}

def score(eq, x, y):
    try:
        pred = evaluate_equation(eq, x)
        return float(r2_score(y, pred)) if np.isfinite(pred).all() else np.nan
    except Exception:
        return np.nan

def select(rows, budget):
    # No test information is read here. Stable row order resolves exact ties.
    pool = rows[rows.selectable & (rows.Length <= budget)]
    return None if pool.empty else pool.sort_values(['val_r2', 'Length'], ascending=[False, True], kind='stable').iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='conditioning_validation_2026-10-13')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    candidates, selected, manifest = [], [], {}
    for flow, (relative, prefix, append) in SPECS.items():
        print('Loading', flow, flush=True)
        frames = build_merged(str(ROOT / relative), str(ROOT / 'galSpecExtra-dr8.fits'), str(ROOT / 'sdss'))
        if append:
            for frame in frames.values():
                y = np.stack(frame.y.values).reshape(len(frame), -1)
                frame['cond'] = list(np.hstack([np.stack(frame.cond.values), y[:, -1:]]))
        manifest[flow] = {'embeddings': str(ROOT / relative), 'splits': {}}
        for split, frame in frames.items():
            clean = frame.dropna(subset=['lgm_tot_p50'])
            clean = clean[clean.lgm_tot_p50 != -9999]
            ids = clean.specObjID.astype(str).to_numpy()
            np.save(out / f'{flow}_{split}_ids.npy', ids.astype(str))
            manifest[flow]['splits'][split] = {'rows': len(ids), 'ordered_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
        for arm in (['cond+z'] if flow == 'znorm' else ['cond', 'cond+z']):
            mats = build_matrices(frames, 'lgm_tot_p50', arm)
            label = f'{flow}_{arm}'
            np.savez(out / f'{label}_matrices.npz', **mats)
            for seed in (42, 43, 44):
                path = Path(f'job_results_{prefix}lgm_tot_p50_{arm}_seed{seed}/pareto_fronts.csv')
                raw = pd.read_csv(path)
                idx = np.random.default_rng(seed).choice(len(mats['X_train']), min(10000, len(mats['X_train'])), replace=False)
                rows = []
                for i, r in raw.iterrows():
                    eq = r.get('Equation_p12', r.Equation)
                    tr = score(eq, mats['X_train'][idx], mats['y_train'][idx])
                    va = score(eq, mats['X_val'], mats['y_val'])
                    full = score(eq, mats['X_train'], mats['y_train'])
                    delta = abs(tr - r.Train_R2)
                    ok = bool(np.isfinite(tr) and np.isfinite(r.Train_R2) and delta <= 1e-6)
                    rows.append(dict(arm=label, seed=seed, row=i, source=str(path), Length=int(r.Length), equation=eq,
                                     train_r2=tr, train_r2_stored=r.Train_R2, train_delta=delta,
                                     train_full_finite=bool(np.isfinite(full)), val_r2=va,
                                     selectable=bool(ok and np.isfinite(full) and np.isfinite(va))))
                scored = pd.DataFrame(rows)
                candidates.extend(rows)
                for budget in range(5, 30):
                    r = select(scored, budget)
                    if r is None: continue
                    result = r.to_dict(); result['budget'] = budget
                    result['test_r2'] = score(r.equation, mats['X_test'], mats['y_test'])
                    result['test_status'] = 'ok' if np.isfinite(result['test_r2']) else 'evaluation_failure'
                    selected.append(result)
                print(label, seed, 'eligible', int(scored.selectable.sum()), '/', len(scored), flush=True)
                pd.DataFrame(candidates).to_csv(out / 'candidate_audit.csv', index=False)
                pd.DataFrame(selected).to_csv(out / 'validation_selected.csv', index=False)
        del frames
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(selected).query('budget == 29')[['arm', 'seed', 'Length', 'val_r2', 'test_r2']].to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
