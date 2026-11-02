"""Exhaustive structure audit of frozen validation selections; no reselection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from val_rescore import to_sympy
from analyse_fibre_mass import has_fixed_latent_direction
OUT=Path('shared_structure_audit_2026-10-14');OUT.mkdir(exist_ok=True)
SOURCES={'fibre':'fibre_mass_conditioning_2026-10-14/comparison/selected_all_budgets.csv','total':'conditioning_validation_2026-10-13/matched_controls/comparison/selected_all_budgets.csv'}
rows=[];atlas=[]
for target,path in SOURCES.items():
 data=pd.read_csv(path)
 for (arm,seed),group in data.groupby(['arm','seed']):
  cache={}
  for r in group.sort_values('budget').itertuples():
   if r.equation not in cache:
    no_lat=arm=='F+z'; no_norm=arm in ['joint lat','joint lat+z'];no_z=arm in ['joint lat','joint lat+F','orig+F','zflow lat+F']
    nidx=None if no_norm else (1 if no_lat else 11);zidx=None if no_z else (2 if no_lat else (11 if no_norm else 12))
    width=max([10 if not no_lat else 0,nidx or 0,zidx or 0])
    expr=to_sympy(r.equation,width)
    expr=expr.xreplace({v:sp.Symbol(v.name,real=True) for v in expr.free_symbols})
    expr=expr.xreplace({v:sp.Rational(str(v)) for v in expr.atoms(sp.Float)})
    lat={sp.Symbol(f'X{i}',real=True) for i in range(1,11)} if not no_lat else set()
    n=sp.Symbol(f'X{nidx}',real=True) if nidx else None;z=sp.Symbol(f'X{zidx}',real=True) if zidx else None
    present=expr.free_symbols&lat
    grads=[sp.diff(expr,v) for v in sorted(present,key=str)]
    constant=bool(present) and all(not g.free_symbols for g in grads)
    affine=bool(present) and all(not(g.free_symbols&lat) for g in grads)
    # Derivative independence gives additive separation on smooth domains.
    sep=bool(present) and all(not (g.free_symbols & {v for v in (n,z) if v is not None}) for g in grads)
    dn=sp.diff(expr,n) if n is not None else sp.Integer(0)
    dz=sp.diff(expr,z) if z is not None else sp.Integer(0)
    nadd=n is not None and n in expr.free_symbols and dn.free_symbols <= {n}
    zadd=z is not None and z in expr.free_symbols and dz.free_symbols <= {z}
    linear_n=bool(nadd and not dn.free_symbols)
    fixed=has_fixed_latent_direction(expr) if present else False
    kind='no latent dependence' if not present else ('additive linear score' if constant else ('nonlinear fixed score' if fixed else ('affine with variable weights' if affine else 'multiple/nonlinear latent effects')))
    cache[r.equation]=dict(target=target,arm=arm,seed=seed,Length=int(r.Length),test_r2=r.test_r2,val_r2=r.val_r2,latent_count=len(present),latent_coordinates=','.join(str(v) for v in sorted(present,key=str)),constant_latent_coefficients=constant,affine_latents=affine,latent_conditioning_separable=sep,fixed_latent_direction=fixed,norm_separable=bool(nadd),redshift_separable=bool(zadd),linear_log_norm=linear_n,norm_coefficient=float(dn) if linear_n else np.nan,fully_additive=bool(present and sep and nadd and zadd),structure=kind,equation=r.equation)
    rename={sp.Symbol(f'X{i}',real=True):sp.Symbol(f'x_{i}') for i in range(1,11)} if not no_lat else {}
    if n is not None:rename[n]=sp.Symbol('n')
    if z is not None:rename[z]=sp.Symbol('z')
    atlas.append((target,arm,seed,int(r.Length),r.equation,sp.latex(sp.N(expr.subs(rename),6))))
   rows.append(dict(cache[r.equation],budget=r.budget))
  print(target,arm,seed,'unique equations',len(cache),flush=True)
d=pd.DataFrame(rows);d.to_csv(OUT/'all_budgets.csv',index=False)
u=d.drop_duplicates(['target','arm','seed','equation']);u.to_csv(OUT/'unique_equations.csv',index=False)
summary=d.groupby(['target','arm','budget']).agg(seeds=('seed','nunique'),mean_test_r2=('test_r2','mean'),additive_linear=('constant_latent_coefficients','sum'),fixed_direction=('fixed_latent_direction','sum'),fully_additive=('fully_additive','sum'),linear_log_norm=('linear_log_norm','sum')).reset_index();summary.to_csv(OUT/'structure_counts.csv',index=False)
lines=['# Shared structure across stellar-mass equations','','All frozen validation-selected budgets 5–29 are included. Equations carried forward across budgets are counted once in unique_equations.csv. No new selection or fitting was performed. n is standardised log10 NORM, z is raw redshift, and x_i are standardised latents. The target is log10 stellar mass in solar units.','','## Exact structural tests','','Derivatives are computed using real symbols and exact rational representations of exported decimal coefficients. Additive linear means constant coefficients on all active latents. Fixed direction includes nonlinear functions of a single fixed linear latent score. Fully additive means separability into a brightness function, redshift function and latent function on smooth connected domains. These tests do not establish a physical mass-to-light interpretation or independence of the coordinates.','','## Counts across three seeds','','| Target | Arm | Budget | Additive linear latent score | Fixed latent direction | Fully additive n/z/latents | Linear log N | Mean test R² |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in summary[summary.budget.isin([9,12,15,18,24,29])].itertuples():lines.append(f'| {r.target} | {r.arm} | {r.budget} | {r.additive_linear}/3 | {r.fixed_direction}/3 | {r.fully_additive}/3 | {r.linear_log_norm}/3 | {r.mean_test_r2:.4f} |')
lines+=['','## Scope and limitations','','This is an audit of reproducible selected exports, not every candidate produced by a search. The new_total fronts retain the export exclusions documented in ../new_sr_audit_2026-10-14/REPORT.md. Cross-arm catalogue coverage is not identical for every reference arm. Seeds represent SR search variability, not independent flow training. Structural resemblance at selected lengths is not a principled reason to choose that length retrospectively.','','Expected physical reference: m_fib = C + log10 N + 2 log10 D_L + log10(1+z) + log10 Upsilon. Total mass additionally involves an aperture factor. An empirical additive latent term is not automatically log10 Upsilon. A fitted log(1+kz) is not exactly the cosmological distance term. An exponential in n is not linear in log10 N.','','## Equation atlas','','Every unique selected equation is listed below, rounded for reading. Use unique_equations.csv for full precision.']
for target,arm,seed,L,eq,tex in atlas:lines.extend([f'### {target}: {arm}, seed {seed}, L={L}','', r'$$\widehat m='+tex+'$$',''])
(OUT/'REPORT.md').write_text('\n'.join(lines))
# Compact overview for the three conditional arms at both mass targets.
fig,axs=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True,layout='constrained')
for i,target in enumerate(['total','fibre']):
 for j,arm in enumerate(['zflow','norm','znorm']):
  g=summary[(summary.target==target)&(summary.arm==arm)];ax=axs[i,j]
  ax.step(g.budget,g.additive_linear,where='mid',label='Additive linear latent score')
  ax.step(g.budget,g.fixed_direction,where='mid',ls='--',label='Fixed latent direction')
  ax.step(g.budget,g.fully_additive,where='mid',ls=':',label='Separate brightness/redshift/latents')
  ax.set_title(f'{target}: {arm}');ax.set_yticks([0,1,2,3]);ax.set_ylim(-.1,3.2)
  if i==1:ax.set_xlabel('Expression-length budget')
  if j==0:ax.set_ylabel('Number of SR seeds')
axs[0,0].legend(fontsize=7)
fig.savefig(OUT/'structure_persistence.pdf');fig.savefig(OUT/'structure_persistence.png',dpi=220)
print(summary[summary.budget.isin([12,15,18,24,29])].to_string(index=False))
