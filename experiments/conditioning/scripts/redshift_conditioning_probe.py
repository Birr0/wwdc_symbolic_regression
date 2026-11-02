"""Matched latent-only redshift probes: coordinate correlations, OLS and 64/32 MLP."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import d4000_mlp as mlp
from matched_conditioning_controls import digest

OUT=Path('redshift_conditioning_probe_2026-10-16')
BASE=Path('conditioning_validation_2026-10-13')
MATCH=BASE/'matched_controls'
ARMS=('original','zflow','norm','znorm')

def main():
 OUT.mkdir(exist_ok=False)
 paths={a: MATCH/f'{a}_matrices.npz' if a in ('original','zflow') else BASE/f'{a}_cond+z_matrices.npz' for a in ARMS}
 reference=np.load(paths['original']);corr=[];linear=[];baselines=[]
 for seed in (42,43,44):
  shutil.copy2(MATCH/f'train_indices_seed{seed}.npy', OUT/f'train_indices_seed{seed}.npy')
 for arm,path in paths.items():
  source=np.load(path);mats={}
  idsbase=MATCH if arm in ('original','zflow') else BASE
  for split in ('train','val','test'):
   ids=np.load(idsbase/(f'{split}_ids.npy' if arm in ('original','zflow') else f'{arm}_{split}_ids.npy'))
   np.testing.assert_array_equal(ids,np.load(MATCH/f'{split}_ids.npy'))
   np.testing.assert_array_equal(source[f'X_{split}'][:,11],reference[f'X_{split}'][:,11])
   mats[f'X_{split}']=source[f'X_{split}'][:,:10]
   mats[f'y_{split}']=source[f'X_{split}'][:,11]
  np.savez(OUT/f'{arm}_matrices.npz',**mats)
  for j in range(10):
   x=mats['X_test'][:,j];y=mats['y_test']
   corr.append(dict(arm=arm,coordinate=j+1,pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(spearmanr(x,y).statistic)))
  for seed in (42,43,44):
   idx=np.load(OUT/f'train_indices_seed{seed}.npy')
   model=LinearRegression().fit(mats['X_train'][idx],mats['y_train'][idx])
   pred=model.predict(mats['X_test']);np.save(OUT/f'{arm}_linear_seed{seed}_predictions.npy',pred)
   linear.append(dict(arm=arm,seed=seed,test_r2=r2_score(mats['y_test'],pred),test_rmse=np.sqrt(mean_squared_error(mats['y_test'],pred)),coefficients=model.coef_.tolist(),intercept=float(model.intercept_)))
   if arm=='original':
    pred=np.full(len(mats['y_test']),mats['y_train'][idx].mean())
    baselines.append(dict(seed=seed,test_r2=r2_score(mats['y_test'],pred),test_rmse=np.sqrt(mean_squared_error(mats['y_test'],pred))))
 pd.DataFrame(corr).to_csv(OUT/'coordinate_correlations.csv',index=False)
 pd.DataFrame(linear).to_json(OUT/'linear_models.json',orient='records',indent=2)
 pd.DataFrame(linear).drop(columns=['coefficients','intercept']).to_csv(OUT/'linear_results.csv',index=False)
 pd.DataFrame(baselines).to_csv(OUT/'mean_baseline.csv',index=False)
 (OUT/'preparation.json').write_text(json.dumps(dict(inputs='Ten latent coordinates only. Neither NORM nor redshift supplied as predictor. Target is raw redshift.',rows={s:len(reference[f'X_{s}']) for s in ('train','val','test')},source_hashes={str(p):digest(p) for p in paths.values()}),indent=2))
 mlp.BASE=OUT;mlp.OUT=OUT/'mlp';mlp.ARMS=ARMS
 mlp.PARAMS=dict(mlp.PARAMS,target='Raw redshift, standardized on fitted training subset')
 mlp.main()
 report()

def report():
 corr=pd.read_csv(OUT/'coordinate_correlations.csv');lin=pd.read_csv(OUT/'linear_results.csv');m=pd.read_csv(OUT/'mlp/summary.csv')
 results=pd.read_csv(OUT/'mlp/results.csv')
 for r in results.itertuples():
  history=pd.read_csv(OUT/f'mlp/{r.arm}_seed{r.seed}_history.csv')
  assert int(history.loc[history.val_mse.idxmin(),'epoch'])==r.best_epoch
  pred=np.load(OUT/f'mlp/{r.arm}_seed{r.seed}_test_predictions.npy');y=np.load(OUT/f'{r.arm}_matrices.npz')['y_test']
  assert np.isfinite(pred).all()
  np.testing.assert_allclose(r2_score(y,pred),r.test_r2,atol=1e-12)
 lines=['# Residual redshift information in the representations','','Ten latent inputs only; neither z nor NORM appended. Existing matched splits and exactly the same 10,000 training rows for each of seeds 42–44. Coordinate correlations use the full test set. OLS has no tuned parameters. MLP: 64/32 ReLU, Adam 0.001, batch 1024, at most 500 epochs, patience 40; checkpoint selected on existing full validation set before test evaluation. Input/target scaling fitted to each 10k training subset.','','| Representation | Max absolute coordinate Pearson r | Max absolute coordinate Spearman rho | Linear mean test R² | MLP mean test R² | MLP seed range | MLP RMSE in z |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in ARMS:
  c=corr[corr.arm==arm];r=m[m.arm==arm].iloc[0];l=lin[lin.arm==arm]
  lines.append(f'| {arm} | {c.pearson.abs().max():.4f} | {c.spearman.abs().max():.4f} | {l.test_r2.mean():.4f} | {r.test_r2:.4f} | {r.test_r2_min:.4f}–{r.test_r2_max:.4f} | {r.test_rmse:.5f} |')
 lines+=['','These are predictive probes on the filtered total-mass sample, not proofs of independence or estimates of all recoverable information. A weak coordinate correlation does not rule out nonlinear prediction. NORM conditioning does not explicitly impose independence from redshift. Seed variation includes training subset and model initialization, not independent flow training. Raw R² may be negative when worse than the test-mean reference.','','Epoch-cap counts: '+str(dict(zip(m.arm,m.capped_runs)))+'. All saved MLP predictions and validation checkpoint epochs verified.','','Reproduce: `../../../.venv/bin/python redshift_conditioning_probe.py`. Preparation refuses overwrite. Training loop reused from `d4000_mlp.py`.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
