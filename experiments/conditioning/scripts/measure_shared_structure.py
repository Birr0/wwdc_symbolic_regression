"""Numerical diagnostics at zero latent coordinates; no selection or refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from val_rescore import evaluate_equation
from analyse_matched_conditioning import gradient
OUT=Path('shared_structure_audit_2026-10-14');BASE=Path('conditioning_validation_2026-10-13');cosmo=FlatLambdaCDM(H0=70,Om0=.3)
d=pd.read_csv(OUT/'unique_equations.csv');rows=[]
for (target,arm),group in d[d.target.isin(['total','fibre'])].groupby(['target','arm']):
 if target=='fibre':path=Path('fibre_mass_conditioning_2026-10-14')/f'{arm}_matrices.npz'
 elif arm in ['original','zflow']:path=BASE/'matched_controls'/f'{arm}_matrices.npz'
 else:path=BASE/f'{arm}_cond+z_matrices.npz'
 mats=np.load(path);x=mats['X_test'].astype(float);zero=x.copy();zero[:,:10]=0;z=x[:,11];h=1e-4
 zp=z*10**h;zm=z*10**(-h);dd=np.log10(cosmo.luminosity_distance(zp).value)-np.log10(cosmo.luminosity_distance(zm).value)
 sigma=.2574361671996528 if target=='fibre' else .25743581284077843
 for r in group.itertuples():
  base,g=gradient(r.equation,x);full=evaluate_equation(r.equation,x);delta=full-base;lin=np.sum(g*x[:,:10],axis=1)
  xp=zero.copy();xm=zero.copy();xp[:,10]+=h/sigma;xm[:,10]-=h/sigma
  flux=(evaluate_equation(r.equation,xp)-evaluate_equation(r.equation,xm))/(2*h)
  xp=zero.copy();xm=zero.copy();xp[:,11]=zp;xm[:,11]=zm
  dist=(evaluate_equation(r.equation,xp)-evaluate_equation(r.equation,xm))/dd
  valid=np.isfinite(base)&np.isfinite(g).all(axis=1)&np.isfinite(full)&np.isfinite(flux)&np.isfinite(dist)
  row=dict(target=target,arm=arm,seed=r.seed,Length=r.Length,equation=r.equation,valid_reference_rows=int(valid.sum()),total_rows=len(x))
  if valid.any():
   gv=g[valid];u,s,v=np.linalg.svd(gv,full_matrices=False)
   const=np.mean(gv,axis=0);additive=x[valid,:10]@const
   row.update(flux_median=float(np.median(flux[valid])),flux_p16=float(np.percentile(flux[valid],16)),flux_p84=float(np.percentile(flux[valid],84)),distance_median=float(np.median(dist[valid])),distance_p16=float(np.percentile(dist[valid],16)),distance_p84=float(np.percentile(dist[valid],84)),correction_sd=float(np.std(delta[valid])),local_linearisation_rmse=float(np.sqrt(np.mean((delta[valid]-lin[valid])**2))),constant_score_approx_rmse=float(np.sqrt(np.mean((delta[valid]-additive)**2))),gradient_rank1_energy=float(s[0]**2/np.sum(s*s)) if np.sum(s*s)>1e-20 else np.nan)
  rows.append(row)
 print(target,arm,'measured',len(group),flush=True)
pd.DataFrame(rows).to_csv(OUT/'numerical_diagnostics.csv',index=False)
b=pd.read_csv(OUT/'all_budgets.csv');b=b.merge(pd.DataFrame(rows),on=['target','arm','seed','Length','equation'],how='left',validate='many_to_one');b.to_csv(OUT/'all_budgets_with_diagnostics.csv',index=False)
print(b[(b.target.isin(['total','fibre'])) & b.budget.isin([12,15,18,24,29])].groupby(['target','arm','budget'])[['constant_score_approx_rmse','local_linearisation_rmse','flux_median','distance_median','correction_sd']].mean().to_string())
