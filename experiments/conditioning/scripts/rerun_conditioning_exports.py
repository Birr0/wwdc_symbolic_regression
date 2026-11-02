"""Recover native training scores and high-precision exports in isolated outputs."""
from pathlib import Path
import hashlib
import json
import time
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from utils import fit_sym_fn
from validate_conditioning import ROOT, SPECS
from val_rescore import build_merged

out = Path('conditioning_validation_2026-10-13')
frames = build_merged(str(ROOT / SPECS['norm'][0]), str(ROOT / 'galSpecExtra-dr8.fits'), str(ROOT / 'sdss'))
for frame in frames.values():
    frame['cond'] = list(np.hstack([np.stack(frame.cond.values), np.stack(frame.y.values).reshape(len(frame), 1)]))
for arm in ('cond', 'cond+z'):
    dest = out / f'rerun_norm_{arm}_seed42'
    dest.mkdir(exist_ok=False)
    start = time.time()
    print('START', arm, flush=True)
    result = fit_sym_fn('lgm_tot_p50', arm, frames, 42)
    model = result['Model']
    rows = []
    for member in model.pareto_front_:
        tree = member['tree']
        pred = model.evaluate_model(tree, result['X_train'])
        native = float(r2_score(result['y_train'], pred)) if np.isfinite(pred).all() else np.nan
        rows.append(dict(Length=tree.Length, Equation=member['model'],
                         Equation_p12=model.get_model_string(tree, precision=12),
                         Equation_p60=model.get_model_string(tree, precision=60), Train_R2=native))
    fresh = pd.DataFrame(rows)
    fresh.to_csv(dest / 'native_front.csv', index=False)
    oldpath = Path(f'job_results_normflow_condnorm_lgm_tot_p50_{arm}_seed42/pareto_fronts.csv')
    old = pd.read_csv(oldpath)
    identical = len(old) == len(fresh) and all(old[c].tolist() == fresh[c].tolist() for c in ('Length', 'Equation', 'Equation_p12'))
    manifest = dict(exact_front_match=identical, elapsed_seconds=time.time()-start,
                    original_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
                    utils_sha256=hashlib.sha256(Path('utils.py').read_bytes()).hexdigest(),
                    versions={p: importlib.metadata.version(p) for p in ('pyoperon','numpy','scikit-learn')},
                    parameters=model.get_params())
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    print('DONE', arm, manifest, flush=True)
