"""Compare fibre and total mass SR on matched objects; selection remains frozen."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import sympy as sp
import yaml
from astropy.cosmology import FlatLambdaCDM
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fibre_mass_conditioning import OUT,BASE,ARMS,TARGET
from validate_conditioning import ROOT,SPECS
from val_rescore import evaluate_equation,to_sympy,build_merged
from analyse_matched_conditioning import gradient,complete_mean,LABELS
from figstyle import use_science

DEST=OUT/'comparison'
COSMO=FlatLambdaCDM(H0=70,Om0=.3)

def has_fixed_latent_direction(expr):
    """Test proportional symbolic gradients, including nonlinear single-score forms.

    Constant ratios imply dependence on one fixed linear combination on each
    connected differentiable domain; this does not imply an additive correction.
    """
    # Operon inputs are real. Without real assumptions, differentiating Abs
    # introduces independent re/im derivatives and falsely breaks proportionality.
    expr=expr.xreplace({v:sp.Symbol(v.name,real=True) for v in expr.free_symbols})
    expr=expr.xreplace({v:sp.Rational(str(v)) for v in expr.atoms(sp.Float)})
    grads=[sp.diff(expr,sp.Symbol(f'X{i}',real=True)) for i in range(1,11)]
    active=[g for g in grads if g!=0]
    if not active:return False
    return all(not sp.cancel(g/active[0]).free_symbols for g in active)

def baseline():
    DEST.mkdir(exist_ok=True)
    frames=pd.read_pickle(OUT/'norm_frames.pkl')
    config=yaml.safe_load((ROOT/'sdss_II/spender_I_normalisation_flow/ckpts/norm_20261012183217_0.yaml').read_text())
    scaling=config['normalisation']['train']; logn={}
    lum={}
    for split,frame in frames.items():
        logn[split]=np.stack(frame.cond)[:,10]*scaling['std']+scaling['mean']
        z=frame.z_x.to_numpy()
        lum[split]=logn[split]+2*np.log10(COSMO.luminosity_distance(z).value)+np.log10(1+z)
    intercept=float(np.mean(frames['train'][TARGET]-lum['train']))
    yp=lum['test']+intercept;y=frames['test'][TARGET].to_numpy()
    result=dict(target=TARGET,formula='log10 N + 2 log10 D_L/Mpc + log10(1+z) + C',intercept_training_fit=intercept,
        test_r2=float(r2_score(y,yp)),test_rmse=float(np.sqrt(np.mean((yp-y)**2))),test_target_sd=float(np.std(y)),
        logn_train_mean=float(logn['train'].mean()),logn_train_std=float(logn['train'].std()),
        cosmology='FlatLambdaCDM(H0=70, Om0=0.3)',train_rows=len(frames['train']),test_rows=len(y))
    (DEST/'physical_baseline.json').write_text(json.dumps(result,indent=2))
    print('PHYSICAL BASELINE',result,flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-only',action='store_true');args=ap.parse_args()
    physical=baseline()
    if args.baseline_only:return
    for arm in ARMS:
        for seed in (42,43,44):
            if not (OUT/f'{arm}_seed{seed}/manifest.json').exists():raise FileNotFoundError(f'Incomplete fit: {arm}, {seed}')
    original=pd.read_csv(BASE/'matched_controls/comparison/selected_all_budgets.csv')
    allrows=[];diagnostics=[];equations=[]
    # Shared catalogue rows are used only for post-selection interpretation.
    cat=build_merged(str(ROOT/SPECS['norm'][0]),str(ROOT/'galSpecExtra-dr8.fits'),str(ROOT/'sdss'))['test']
    ids=np.load(OUT/'test_ids.npy').astype('int64');cat=cat.set_index('specObjID',drop=False).loc[ids].reset_index(drop=True)
    vagc=pd.read_parquet('plot_cmd_latent_results/vagc_join.parquet')[['specObjID','ABSMAG_g','ABSMAG_r','MTOL_r']].drop_duplicates()
    cat=cat.merge(vagc,on='specObjID',how='left',validate='many_to_one',sort=False)
    np.testing.assert_array_equal(cat.specObjID,ids)
    validmag=cat.ABSMAG_g.between(-30,-5)&cat.ABSMAG_r.between(-30,-5)
    cat['gr']=(cat.ABSMAG_g-cat.ABSMAG_r).where(validmag)
    cat['log_mtol']=np.log10(cat.MTOL_r.where(cat.MTOL_r>0))
    cat['d4000']=cat.d4000.where(cat.d4000.between(.5,3))
    associations=[]
    for arm in ARMS:
        mats=np.load(OUT/f'{arm}_matrices.npz');x=mats['X_test'].astype(float);y=mats['y_test'];zero=x.copy();zero[:,:10]=0
        np.testing.assert_array_equal(y,cat[TARGET])
        selection=pd.concat([pd.read_csv(OUT/f'{arm}_seed{s}/selected.csv') for s in (42,43,44)],ignore_index=True)
        selection['validation_rmse']=np.sqrt(np.maximum(0,1-selection.val_r2)*np.var(mats['y_val']))
        selection['test_rmse']=np.sqrt(np.maximum(0,1-selection.test_r2)*np.var(y))
        allrows.append(selection)
        z=x[:,11];h=1e-4;zp=z*10**h;zm=z*10**(-h)
        dlogd=np.log10(COSMO.luminosity_distance(zp).value)-np.log10(COSMO.luminosity_distance(zm).value)
        expected=2+(np.log10(1+zp)-np.log10(1+zm))/dlogd
        for r in selection[selection.budget.isin([12,15,18,24,29])].itertuples():
            eq=r.equation;expr=to_sympy(eq,12);latent={sp.Symbol(f'X{i}') for i in range(1,11)}
            deriv=[sp.diff(expr,v) for v in sorted(latent,key=str)]
            affine=all(not (d.free_symbols&latent) for d in deriv);constant=all(not d.free_symbols for d in deriv)
            fixed_direction=has_fixed_latent_direction(expr)
            xp=zero.copy();xm=zero.copy();xp[:,10]+=h/physical['logn_train_std'];xm[:,10]-=h/physical['logn_train_std']
            f=(evaluate_equation(eq,xp)-evaluate_equation(eq,xm))/(2*h)
            xp=zero.copy();xm=zero.copy();xp[:,11]=zp;xm[:,11]=zm
            ds=(evaluate_equation(eq,xp)-evaluate_equation(eq,xm))/dlogd
            base,g=gradient(eq,x);full=evaluate_equation(eq,x);delta=full-base
            valid=np.isfinite(base)&np.isfinite(g).all(axis=1)&np.isfinite(f)&np.isfinite(ds)
            if not valid.any():
                diagnostics.append(dict(arm=arm,seed=r.seed,budget=r.budget,Length=r.Length,
                    affine_latents=affine,constant_latent_coefficients=constant,fixed_latent_direction=fixed_direction,latent_count=len(expr.free_symbols&latent),
                    flux_median=np.nan,distance_median=np.nan,reference_r2=np.nan,correction_sd=np.nan,
                    finite_reference_rows=0,rank1_gradient_energy=np.nan))
                if r.budget==15:equations.append(dict(arm=arm,seed=r.seed,Length=r.Length,equation=eq))
                continue
            singular=np.linalg.svd(g[valid],compute_uv=False)
            diagnostics.append(dict(arm=arm,seed=r.seed,budget=r.budget,Length=r.Length,
                affine_latents=affine,constant_latent_coefficients=constant,fixed_latent_direction=fixed_direction,latent_count=len(expr.free_symbols&latent),
                flux_median=float(np.median(f[valid])),flux_p16=float(np.percentile(f[valid],16)),flux_p84=float(np.percentile(f[valid],84)),distance_median=float(np.median(ds[valid])),expected_distance=float(np.median(expected[valid])),
                reference_r2=float(r2_score(y[valid],base[valid])),correction_sd=float(np.std(delta[valid])),
                finite_reference_rows=int(valid.sum()),rank1_gradient_energy=float(singular[0]**2/np.sum(singular**2)) if np.sum(singular**2)>1e-20 else np.nan))
            if r.budget==15:
                equations.append(dict(arm=arm,seed=r.seed,Length=r.Length,equation=eq))
                cell=pd.qcut(x[:,10],10,labels=False,duplicates='drop')*10+pd.qcut(z,10,labels=False,duplicates='drop')
                for prop in ('gr','log_mtol','d4000'):
                    sub=pd.DataFrame({'value':cat[prop].to_numpy(),'delta':delta,'cell':cell}).replace([np.inf,-np.inf],np.nan).dropna()
                    sub=sub[sub.groupby('cell').delta.transform('size')>=20]
                    residual=sub[['value','delta']]-sub.groupby('cell')[['value','delta']].transform('mean')
                    associations.append(dict(arm=arm,seed=r.seed,property=prop,n=len(sub),pearson=float(sub.value.corr(sub.delta)),within_flux_z_pearson=float(residual.value.corr(residual.delta))))
    combined=pd.concat(allrows,ignore_index=True);combined.to_csv(DEST/'selected_all_budgets.csv',index=False)
    summary=combined.groupby(['arm','budget']).agg(test_r2=('test_r2',complete_mean),test_rmse=('test_rmse',complete_mean),validation_rmse=('validation_rmse',complete_mean),min_test_r2=('test_r2','min'),max_test_r2=('test_r2','max')).reset_index()
    summary.to_csv(DEST/'frontier_summary.csv',index=False)
    total=original[original.arm.isin(ARMS)].groupby(['arm','budget']).agg(total_test_r2=('test_r2',complete_mean),total_test_rmse=('test_rmse',complete_mean)).reset_index()
    comparison=summary.merge(total,on=['arm','budget'],validate='one_to_one')
    comparison.to_csv(DEST/'fibre_vs_total.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(DEST/'physical_structure.csv',index=False)
    pd.DataFrame(associations).to_csv(DEST/'population_associations_L15.csv',index=False)
    pd.DataFrame(equations).to_csv(DEST/'equations_L15.csv',index=False)
    tex=['% Fibre-mass predictions; n is standardised log10 NORM; z is raw redshift.']
    rename={sp.Symbol(f'X{i}'):sp.Symbol(f'x_{i}') for i in range(1,11)};rename.update({sp.Symbol('X11'):sp.Symbol('n'),sp.Symbol('X12'):sp.Symbol('z')})
    for r in equations:
        tex.extend([f"% {r['arm']}, seed {r['seed']}, length {r['Length']}",r'\begin{equation}',r'\widehat m_{\mathrm{fib}}='+sp.latex(sp.N(to_sympy(r['equation'],12).subs(rename),5)),r'\end{equation}',''])
    (DEST/'equations_L15.tex').write_text('\n'.join(tex))
    use_science();fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for arm in ARMS:
        c=comparison[comparison.arm==arm];line,=axes[0].plot(c.budget,c.test_r2,label=LABELS[arm]);axes[0].fill_between(c.budget,c.min_test_r2,c.max_test_r2,color=line.get_color(),alpha=.12)
        axes[1].plot(c.budget,c.test_rmse,color=line.get_color(),label=LABELS[arm]+' (fibre)');axes[1].plot(c.budget,c.total_test_rmse,color=line.get_color(),ls='--',alpha=.7,label=LABELS[arm]+' (total)')
    axes[0].set(xlabel='Expression-length budget',ylabel='Fibre-mass test $R^2$');axes[1].set(xlabel='Expression-length budget',ylabel='Test RMSE [dex]')
    for ax in axes:ax.axvline(15,ls=':',color='grey');ax.legend(fontsize=7)
    fig.savefig(DEST/'fibre_mass_frontiers.pdf');fig.savefig(DEST/'fibre_mass_frontiers.png',dpi=300);plt.close(fig)
    lines=['# Fibre-mass symbolic regression', '',
        'Target: MPA-JHU `LGM_FIB_P50`, already log10 stellar mass in solar units. Each arm uses ten conditional latents + exact NORM + raw z. No flow retraining or original-latent SR arm is included.', '',
        '## Validation-selected results', '',
        '| Representation | Budget | Fibre test R² | Fibre test RMSE [dex] | Total test R² | Total test RMSE [dex] |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in comparison[comparison.budget.isin([15,29])].itertuples():lines.append(f'| {LABELS[r.arm]} | {r.budget} | {r.test_r2:.6f} | {r.test_rmse:.6f} | {r.total_test_r2:.6f} | {r.total_test_rmse:.6f} |')
    lines += ['', 'Numbers average SR seeds 42–44, not independent flow-training runs. Test and validation objects are identical to the total-mass comparison. One invalid fibre target is removed from the full training pool; it was absent from all three 10,000-row training subsets, so the fitted objects are exactly preserved. Scalers are refitted on the remaining full training pool.', '',
        '## Physical reference and equation structure', '',
        f"The fixed-slope reference log10 N + 2 log10 D_L + log10(1+z) + C, with C fitted on training data only, has fibre-mass test R²={physical['test_r2']:.6f} and RMSE={physical['test_rmse']:.6f} dex. The fibre target's test SD is {physical['test_target_sd']:.6f} dex.", '',
        'The fibre target removes the explicit total-to-fibre mass aperture factor from the physical comparison. Continuum flux-density scaling, stellar populations, dust and catalogue modelling still matter. The expected flux slope is 1 and the median expected distance slope is approximately 2.087 for the stated cosmology.', '',
        '| Representation | Budget-15 flux slope | Budget-15 distance slope | Exact constant latent coefficients | Fixed latent direction |',
        '| --- | ---: | ---: | ---: | ---: |']
    diag=pd.DataFrame(diagnostics)
    for arm in ARMS:
        d=diag[(diag.arm==arm)&(diag.budget==15)]
        lines.append(f'| {LABELS[arm]} | {d.flux_median.min():.3f}–{d.flux_median.max():.3f} | {d.distance_median.min():.3f}–{d.distance_median.max():.3f} | {int(d.constant_latent_coefficients.sum())}/3 | {int(d.fixed_latent_direction.sum())}/3 |')
    lines += ['', 'Slopes are medians of local zero-latent reference sensitivities over test objects, with ranges across seeds. They are not universal physical constants or confidence intervals. `physical_structure.csv` gives finite-reference counts, slope spread, latent counts and local gradient energy. A rank-one local gradient does not alone establish global linearity.',
        'Fixed latent direction is checked separately by symbolic proportionality of all nonzero latent gradients. It permits nonlinear single-score forms such as a + b ln[F(n,z) + w.x]. Such equations use one fixed linear latent combination but are not exactly F(n,z) + A s. The symbolic criterion applies within connected differentiable domains.', '',
        'For a recovered form m = a + b ln|U(n,z) + v.x|, the zero-latent reference is a + b ln|U| and the exact correction is b ln|1 + (v.x)/U| wherever U is nonzero. Its first-order approximation is (b/U) v.x. Thus one fixed latent score can coexist with a nonlinear correction and a conditioning-dependent amplitude. This should not be described as the constant-amplitude additive form found for the total-mass L=15 equations.',
        'Matching the aperture does not guarantee unit log-flux sensitivity at fixed latent coordinates. A conditional reparameterisation can change which physical properties are held fixed when its latent vector is held fixed, and the reference can absorb population/selection trends. Better fibre-mass predictions are not by themselves evidence of exact recovery of the physical mass-to-light relation.', '',
        '## Reproduction', '',
        'Run `python fibre_mass_conditioning.py prepare`, then `python fibre_mass_conditioning.py fit --arm ARM --seed SEED` for zflow/norm/znorm and 42/43/44; finish with `python analyse_fibre_mass.py`.',
        'Fits use exactly the recorded total-mass SR configuration, including eight threads, maximum search-tree length 25, up to 1000 generations and 10^7 evaluations. Explicit training indices preserve the object subsets after removal of the invalid target. Exports use 60 decimal places.',
        'Eligibility requires finite full-training/validation predictions and native training-score agreement within 1e-6. The best validation score within each budget is frozen before test evaluation. Test failures are not replaced. Original output folders remain unchanged.',
        'Scripts, training-ID hashes, target checks and model parameters are recorded in preparation.json and the per-run manifests. The equations file is rounded for display; evaluate the full precision CSV instead.', '',
        'Physical reference: https://ned.ipac.caltech.edu/level5/Hogg/Hogg7.html ; fibre/total mass distinction: https://wwwmpa.mpa-garching.mpg.de/SDSS/DR7/mass_comp.html']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(comparison[comparison.budget.isin([15,29])].to_string(index=False))

if __name__=='__main__':main()
