"""Compare original and conditional fibre-mass fits after validation selection."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fibre_mass_conditioning import OUT
from analyse_fibre_mass import has_fixed_latent_direction
from analyse_matched_conditioning import gradient,LABELS
from val_rescore import to_sympy
DEST=OUT/'original_control_comparison'

def main():
 DEST.mkdir(exist_ok=True)
 selections=[];verification=[]
 for seed in (42,43,44):
  path=OUT/f'original_seed{seed}';manifest=json.loads((path/'manifest.json').read_text());selected=pd.read_csv(path/'selected.csv');audit=pd.read_csv(path/'candidate_audit.csv');frozen=pd.read_csv(path/'selection_frozen_before_test.csv')
  assert not any('test' in c for c in frozen.columns)
  for r in selected.itertuples():assert abs(audit[audit.selectable & (audit.Length<=r.budget)].val_r2.max()-r.val_r2)<1e-12
  delta=float(abs(selected.test_r2-selected.native_test_r2).max());assert delta<1e-6
  reference=json.loads((OUT/f'znorm_seed{seed}/manifest.json').read_text());assert manifest['parameters']==reference['parameters']
  verification.append(dict(seed=seed,eligible_candidates=int(audit.selectable.sum()),candidates=len(audit),test_failures=int((selected.test_status!='ok').sum()),max_native_export_r2_delta=delta))
  selections.append(selected)
 original=pd.concat(selections,ignore_index=True);conditional=pd.read_csv(OUT/'comparison/selected_all_budgets.csv');combined=pd.concat([conditional,original],ignore_index=True)
 mats=np.load(OUT/'original_matrices.npz');combined['test_rmse']=np.sqrt((1-combined.test_r2)*np.var(mats['y_test']));combined['validation_rmse']=np.sqrt((1-combined.val_r2)*np.var(mats['y_val']))
 combined.to_csv(DEST/'selected_all_budgets.csv',index=False)
 summary=combined.groupby(['arm','budget']).agg(test_r2=('test_r2',lambda v:v.mean() if len(v)==3 and v.notna().all() else np.nan),test_rmse=('test_rmse','mean'),validation_rmse=('validation_rmse','mean'),min_r2=('test_r2','min'),max_r2=('test_r2','max')).reset_index();summary.to_csv(DEST/'performance.csv',index=False)
 old=pd.read_csv('shared_structure_audit_2026-10-14/all_budgets.csv');old=old[(old.target=='fibre')].copy();rows=[];atlas=[];cache={};x=mats['X_test'].astype(float)
 for r in original.itertuples():
  key=(r.seed,r.equation)
  if key not in cache:
   e=to_sympy(r.equation,12);e=e.xreplace({v:sp.Symbol(v.name,real=True) for v in e.free_symbols});e=e.xreplace({v:sp.Rational(str(v)) for v in e.atoms(sp.Float)})
   lat={sp.Symbol(f'X{i}',real=True) for i in range(1,11)};nz={sp.Symbol('X11',real=True),sp.Symbol('X12',real=True)};active=e.free_symbols&lat;g=[sp.diff(e,v) for v in active]
   const=bool(active) and all(not v.free_symbols for v in g);fixed=has_fixed_latent_direction(e) if active else False
   n,z=sp.Symbol('X11',real=True),sp.Symbol('X12',real=True);dn=sp.diff(e,n);dz=sp.diff(e,z)
   sep=bool(active) and all(not(v.free_symbols&nz) for v in g) and dn.free_symbols<={n} and dz.free_symbols<={z} and n in e.free_symbols and z in e.free_symbols
   base,grad=gradient(r.equation,x);from val_rescore import evaluate_equation
   full=evaluate_equation(r.equation,x);valid=np.isfinite(base)&np.isfinite(grad).all(axis=1)&np.isfinite(full)
   approx=x[valid,:10]@grad[valid].mean(axis=0)
   cache[key]=dict(target='fibre',arm='original',seed=r.seed,Length=r.Length,equation=r.equation,constant_latent_coefficients=const,fixed_latent_direction=fixed,fully_additive=bool(sep),linear_log_norm=bool(n in e.free_symbols and not dn.free_symbols),latent_coordinates=','.join(str(v) for v in sorted(active,key=str)),valid_reference_rows=int(valid.sum()),constant_score_approx_rmse=float(np.sqrt(np.mean((full[valid]-base[valid]-approx)**2))))
   rename={sp.Symbol(f'X{i}',real=True):sp.Symbol(f'x_{i}') for i in range(1,11)};rename[n]=sp.Symbol('n');rename[z]=sp.Symbol('z')
   atlas.extend([f'## Seed {r.seed}, L={r.Length}', '',r'$$\widehat m_{\rm fib}='+sp.latex(sp.N(e.subs(rename),6))+'$$',''])
  rows.append(dict(cache[key],budget=r.budget))
 structures=pd.concat([old,pd.DataFrame(rows)],ignore_index=True);structures.to_csv(DEST/'structure_all_budgets.csv',index=False)
 counts=structures.groupby(['arm','budget']).agg(additive_linear=('constant_latent_coefficients','sum'),fixed_direction=('fixed_latent_direction','sum'),fully_additive=('fully_additive','sum')).reset_index();counts.to_csv(DEST/'structure_counts.csv',index=False)
 (DEST/'equations_original.md').write_text('# Original-latent fibre-mass equations\n\nRounded for display; full precision in selected_all_budgets.csv. n is standardised log NORM; z is raw redshift.\n\n'+'\n'.join(atlas))
 merged=summary.merge(counts,on=['arm','budget']);merged.to_csv(DEST/'comparison.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
 for arm in ['original','zflow','norm','znorm']:
  g=merged[merged.arm==arm];line,=axes[0].plot(g.budget,g.test_r2,label=LABELS[arm]);axes[0].fill_between(g.budget,g.min_r2,g.max_r2,color=line.get_color(),alpha=.13);axes[1].step(g.budget,g.additive_linear,where='mid',color=line.get_color(),label=LABELS[arm])
 axes[0].set(xlabel='Expression-length budget',ylabel='Fibre-mass test R²');axes[1].set(xlabel='Expression-length budget',ylabel='Seeds with exact additive linear latent term',yticks=[0,1,2,3]);axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
 fig.savefig(DEST/'control_comparison.pdf');fig.savefig(DEST/'control_comparison.png',dpi=250)
 lines=['# Matched original-latent fibre-mass control','','All arms use ten representation coordinates + NORM + raw redshift. Identical targets, splits and 10,000 training IDs per seed. Seeds 42–44 use the same recorded SR settings; selection uses validation only.','', '| Representation | Budget | Test R² | RMSE [dex] | Additive linear latent term | Fixed direction |','|---|---:|---:|---:|---:|---:|']
 for r in merged[merged.budget.isin([12,15,18,24,29])].itertuples():lines.append(f'| {LABELS[r.arm]} | {r.budget} | {r.test_r2:.6f} | {r.test_rmse:.6f} | {r.additive_linear}/3 | {r.fixed_direction}/3 |')
 lines+=['','## Interpretation','','The original-latent control is more accurate than all three conditional arms at budgets 12, 15, 18 and 29. At budget 15 all three original runs select length-14 equations with an exact additive linear latent correction. The conditional fibre equations at that budget are nonlinear fixed-score forms. Therefore the fibre comparison does not support a general claim that conditioning improves separability; it instead shows that compact additive structure also exists in the original representation. The earlier total-mass comparison remains target-specific evidence and should not be generalized to fibre mass.','','Seed 42 at budget 15 (actual L=14), rounded: m_fib = 1.13722 + 0.231687 n + 0.886856 ln(243445.09375 z) - 0.0226336 x4 - 0.0649828 x6 + 0.145934 x9. All sample redshifts are positive. Here n is standardised log10 NORM; x_i are standardised original latents.','','Counts refer to exact algebraic structure of the selected equations, not whether an approximate decomposition is useful. They are descriptive results at fixed budgets; no length was chosen from test performance or appearance of the equations.','', 'The figure shows means and min/max test R² across SR seeds, not uncertainty over independent flow training. An additive latent term does not identify a physical mass-to-light ratio.','', 'Reproduce: python fibre_mass_original_control.py prepare; python fibre_mass_original_control.py fit --seed SEED (42,43,44); python analyse_fibre_original_control.py. Preparation refuses to overwrite saved inputs.']
 (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n');(DEST/'verification.json').write_text(json.dumps(verification,indent=2));print(merged[merged.budget.isin([12,15,18,29])].to_string(index=False),flush=True)

if __name__=='__main__':main()
