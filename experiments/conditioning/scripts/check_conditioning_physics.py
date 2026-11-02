"""Post-selection physical diagnostics; no refitting or reselecting SR equations.

Run from this directory with the flower analysis environment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from astropy.cosmology import FlatLambdaCDM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_conditioning import ROOT, SPECS
from val_rescore import build_merged, evaluate_equation

BASE = Path('conditioning_validation_2026-10-13')
OUT = BASE / 'physical_interpretation'
COSMO = FlatLambdaCDM(H0=70, Om0=.3)

def corr(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(np.asarray(a)[good], np.asarray(b)[good])[0, 1])

def main():
    OUT.mkdir(exist_ok=True)
    frames = build_merged(str(ROOT / SPECS['norm'][0]), str(ROOT/'galSpecExtra-dr8.fits'), str(ROOT/'sdss'))
    frames = {k: v.dropna(subset=['lgm_tot_p50']).query('lgm_tot_p50 != -9999').reset_index(drop=True) for k,v in frames.items()}
    for split, frame in frames.items():
        np.testing.assert_array_equal(frame.specObjID.astype(str), np.load(BASE/f'norm_{split}_ids.npy'))
    config = yaml.safe_load((ROOT/'sdss_II/spender_I_normalisation_flow/ckpts/norm_20261012183217_0.yaml').read_text())
    stats = config['normalisation']['train']
    lf = {k: np.stack(v.y).reshape(-1)*stats['std']+stats['mean'] for k,v in frames.items()}
    sigma = float(lf['train'].std())
    mu = float(lf['train'].mean())
    data = frames['test']
    z = data.z_x.to_numpy()
    logd = np.log10(COSMO.luminosity_distance(z).value)
    q = 2*logd+np.log10(1+z)
    # Fixed physical slopes, only a training-set intercept fitted to the mass target.
    ztrain = frames['train'].z_x.to_numpy()
    lt = lf['train']+2*np.log10(COSMO.luminosity_distance(ztrain).value)+np.log10(1+ztrain)
    intercept = float(np.mean(frames['train'].lgm_tot_p50-lt))
    phys_r2 = r2_score(data.lgm_tot_p50, lf['test']+q+intercept)
    vagc = pd.read_parquet('plot_cmd_latent_results/vagc_join.parquet')
    vagc = vagc[['specObjID','ABSMAG_g','ABSMAG_r','MTOL_r']].drop_duplicates()
    assert not vagc.specObjID.duplicated().any(), 'ambiguous VAGC join'
    cat = data[['specObjID','d4000','lgm_tot_p50','LGM_FIB_P50','petroR50_r']].copy()
    cat['specObjID'] = cat.specObjID.astype('int64')
    cat = cat.merge(vagc, on='specObjID', how='left', validate='many_to_one', sort=False)
    assert len(cat) == len(data)
    goodmag = np.isfinite(cat.ABSMAG_g)&np.isfinite(cat.ABSMAG_r)&cat.ABSMAG_g.between(-30,-5)&cat.ABSMAG_r.between(-30,-5)
    cat['gr'] = (cat.ABSMAG_g-cat.ABSMAG_r).where(goodmag)
    cat['log_mtol'] = np.log10(cat.MTOL_r.where(cat.MTOL_r>0))
    cat['d4000'] = cat.d4000.where(cat.d4000.between(.5,3))
    cat['mass_aperture'] = (cat.lgm_tot_p50-cat.LGM_FIB_P50).where(cat.LGM_FIB_P50.between(5,14))
    cat['log_size'] = np.log10(cat.petroR50_r.where(cat.petroR50_r>0))
    # 10x10 quantile bins; associations use within-cell deviations with >=20 valid rows.
    cat['cell'] = pd.qcut(z,10,labels=False,duplicates='drop')*10+pd.qcut(lf['test'],10,labels=False,duplicates='drop')
    selected = pd.read_csv(BASE/'selected_equations_budget29.csv')
    slopes, associations, bins = [], [], []
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for arm in ('norm_cond+z','znorm_cond+z'):
        mats = np.load(BASE/f'{arm}_matrices.npz')
        x = mats['X_test'].astype(float)
        np.testing.assert_allclose(x[:,10], (lf['test']-mu)/sigma, atol=2e-6)
        np.testing.assert_allclose(x[:,11], z)
        zero = x.copy(); zero[:,:10]=0
        for r in selected[selected.arm==arm].itertuples():
            full = evaluate_equation(r.equation,x)
            ref = evaluate_equation(r.equation,zero)
            delta = full-ref
            # Derivatives at observed (N,z), holding latent coordinates fixed at zero.
            eps=1e-4
            xp=zero.copy();xm=zero.copy();xp[:,10]+=eps/sigma;xm[:,10]-=eps/sigma
            flux_slope=(evaluate_equation(r.equation,xp)-evaluate_equation(r.equation,xm))/(2*eps)
            xp=zero.copy();xm=zero.copy();xp[:,11]=z*10**eps;xm[:,11]=z*10**(-eps)
            zp=xp[:,11];zm=xm[:,11]
            dlogd=np.log10(COSMO.luminosity_distance(zp).value)-np.log10(COSMO.luminosity_distance(zm).value)
            distance_slope=(evaluate_equation(r.equation,xp)-evaluate_equation(r.equation,xm))/dlogd
            expected=2+(np.log10(1+zp)-np.log10(1+zm))/dlogd
            plane=LinearRegression().fit(np.column_stack([lf['test'],q]),ref)
            approx=plane.predict(np.column_stack([lf['test'],q]))
            slopes.append(dict(arm=arm,seed=r.seed,Length=r.Length,flux_median=np.median(flux_slope),flux_p16=np.percentile(flux_slope,16),flux_p84=np.percentile(flux_slope,84),distance_median=np.median(distance_slope),distance_expected_median=np.median(expected),plane_flux=plane.coef_[0],plane_luminosity_geometry=plane.coef_[1],plane_reference_rmse=np.sqrt(np.mean((ref-approx)**2)),reference_r2=r2_score(data.lgm_tot_p50,ref),correction_sd=np.std(delta)))
            for prop in ('gr','log_mtol','d4000','mass_aperture','log_size'):
                sub=cat[['cell',prop]].copy();sub['delta']=delta;sub=sub.replace([np.inf,-np.inf],np.nan).dropna()
                counts=sub.groupby('cell').delta.transform('size');sub=sub[counts>=20]
                residual=sub[[prop,'delta']]-sub.groupby('cell')[[prop,'delta']].transform('mean')
                associations.append(dict(arm=arm,seed=r.seed,property=prop,n=len(sub),pearson=corr(sub.delta,sub[prop]),spearman=float(spearmanr(sub.delta,sub[prop]).statistic),within_flux_z_pearson=corr(residual.delta,residual[prop])))
            for j,prop in enumerate(('gr','log_mtol')):
                sub=cat[[prop]].copy();sub['delta']=delta;sub=sub.replace([np.inf,-np.inf],np.nan).dropna()
                sub['bin']=pd.qcut(sub[prop],12,duplicates='drop')
                curve=sub.groupby('bin',observed=True).agg(x=(prop,'median'),delta=('delta','median'),n=('delta','size'))
                for row in curve.itertuples(): bins.append(dict(arm=arm,seed=r.seed,property=prop,x=row.x,delta=row.delta,n=row.n))
                axes[j].plot(curve.x,curve.delta,alpha=.75,label=f'{arm}, {r.seed}')
    axes[0].set(xlabel='Catalogue rest-frame g-r [mag]',ylabel='Median latent correction [dex]')
    axes[1].set(xlabel='Catalogue log10(M/L_r)',ylabel='Median latent correction [dex]')
    axes[0].legend(fontsize=7)
    fig.savefig(OUT/'latent_corrections.pdf');fig.savefig(OUT/'latent_corrections.png',dpi=250)
    pd.DataFrame(slopes).to_csv(OUT/'scaling.csv',index=False)
    pd.DataFrame(associations).to_csv(OUT/'associations.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'correction_bins.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps(dict(log_norm_train_std=sigma,log_norm_train_mean=mu,test_rows=len(data),cosmology='FlatLambdaCDM(H0=70, Om0=0.3)',physical_baseline_test_r2=phys_r2,physical_baseline_intercept=intercept,flow_scaling=stats),indent=2))
    print('FIXED PHYSICAL BASELINE R2',phys_r2)
    print(pd.DataFrame(slopes).to_string(index=False))
    print(pd.DataFrame(associations).to_string(index=False))

if __name__=='__main__':
    main()
