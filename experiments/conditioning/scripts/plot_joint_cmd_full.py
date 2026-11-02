"""Full-sample counterpart of Figure 10; frozen equations, three SR seeds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from cmd_helpers import H1_TO_H70, masters10_red_edge, masters10_ridge
from figstyle import use_science
from val_rescore import to_sympy, evaluate_equation
from matched_conditioning_controls import digest


def main():
    base = Path('conditioning_validation_2026-10-13')
    out = Path('joint_cmd_full'); out.mkdir(exist_ok=True)
    selection = base/'validation_selected_final.csv'
    phot_path = Path('plot_cmd_latent_results/vagc_join.parquet')
    selected = pd.read_csv(selection)
    mats = {a: np.load(base/f'{a}_cond+z_matrices.npz') for a in ('norm','znorm')}
    equations = []; weights = {}
    for arm in mats:
        for seed in (42,43,44):
            row = selected[(selected.arm==arm+'_cond+z') & (selected.seed==seed) & (selected.budget==15)].iloc[0]
            expr = to_sympy(row.equation,12)
            derivatives = [sp.diff(expr,sp.Symbol(f'X{i}')) for i in range(1,11)]
            assert all(not d.free_symbols for d in derivatives)
            weights[arm,seed] = np.array([float(d) for d in derivatives])
            equations.append(dict(arm=arm,seed=seed,Length=int(row.Length),equation=row.equation))
    frames = []; sources = [selection,phot_path]
    for split in ('train','val','test'):
        ids = np.load(base/f'norm_{split}_ids.npy').astype('int64')
        np.testing.assert_array_equal(ids,np.load(base/f'znorm_{split}_ids.npy').astype('int64'))
        np.testing.assert_array_equal(mats['norm'][f'y_{split}'],mats['znorm'][f'y_{split}'])
        np.testing.assert_array_equal(mats['norm'][f'X_{split}'][:,10:],mats['znorm'][f'X_{split}'][:,10:])
        frame = pd.DataFrame(dict(specObjID=ids,split=split,z=mats['norm'][f'X_{split}'][:,11]))
        for e in equations:
            x = mats[e['arm']][f'X_{split}']; score = x[:,:10] @ weights[e['arm'],e['seed']]
            zero = x.copy(); zero[:,:10] = 0
            np.testing.assert_allclose(score,evaluate_equation(e['equation'],x)-evaluate_equation(e['equation'],zero),atol=1e-10,rtol=0)
            frame[f"{e['arm']}_{e['seed']}"] = score
        frames.append(frame)
        sources.extend(base/f'{a}_{split}_ids.npy' for a in mats)
    data = pd.concat(frames,ignore_index=True)
    phot = pd.read_parquet(phot_path)[['specObjID','ABSMAG_g','ABSMAG_r']]
    data = data.merge(phot,on='specObjID',how='inner',validate='many_to_one',sort=False)
    for col in ('ABSMAG_g','ABSMAG_r'):
        data = data[np.isfinite(data[col]) & (data[col]>-99) & (data[col]<0)]
    data = data[(data.z>.02)&(data.z<=.3)].copy()
    data['gr']=data.ABSMAG_g-data.ABSMAG_r; data['Mr']=data.ABSMAG_r+H1_TO_H70
    xe=np.linspace(-24,-18,61); ye=np.linspace(.15,1.2,57)
    counts,_,_=np.histogram2d(data.Mr,data.gr,bins=(xe,ye))
    sample=data.sample(min(60000,len(data)),random_state=0)
    kde=gaussian_kde(np.vstack([sample.Mr,sample.gr]))
    kx=np.linspace(-24,-18,180); ky=np.linspace(.15,1.2,180); xx,yy=np.meshgrid(kx,ky)
    density=kde(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
    ranked=np.sort(density.ravel())[::-1]; mass=np.cumsum(ranked)*(kx[1]-kx[0])*(ky[1]-ky[0])
    fractions=np.array([.1,.3,.5,.7,.9]); assert mass[-1]>.9
    levels=[ranked[np.searchsorted(mass,f)] for f in fractions]
    bins=dict(counts=counts,xedges=xe,yedges=ye,kx=kx,ky=ky,density=density,contour_fractions=fractions,contour_levels=levels)
    use_science(); plt.rcParams.update({'font.size':12,'axes.labelsize':14})
    fig,axes=plt.subplots(3,3,figsize=(15,12),sharex=True,sharey=True,layout='constrained')
    grid=np.linspace(-24,-18,200)
    for i,seed in enumerate((42,43,44)):
        count_image=axes[i,0].pcolormesh(xe,ye,np.log10(np.where(counts>0,counts,np.nan)).T,cmap='Greys',rasterized=True)
        cs=axes[i,0].contour(kx,ky,density,levels=sorted(levels),colors='#9b1b4a',linewidths=.8)
        axes[i,0].clabel(cs,fmt={v:f'{f:.0%}' for v,f in zip(levels,fractions)},fontsize=8)
        axes[i,0].set_title(f'Full sample: {len(data):,} spectra')
        for j,arm in enumerate(('norm','znorm'),1):
            total,_,_=np.histogram2d(data.Mr,data.gr,bins=(xe,ye),weights=data[f'{arm}_{seed}'])
            mean=np.divide(total,counts,out=np.full_like(total,np.nan),where=counts>=25)
            bins[f'{arm}_{seed}']=mean
            score_image=axes[i,j].pcolormesh(xe,ye,mean.T,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.2,vcenter=0,vmax=.2),rasterized=True)
            axes[i,j].set_title(('NORM' if arm=='norm' else 'NORM + z')+f' conditioning: seed {seed}')
        for ax in axes[i]:
            ax.plot(grid,masters10_red_edge(grid),'--',color='.15',lw=1.1)
            ax.plot(grid,masters10_ridge(grid),':',color='.4',lw=1.1)
            ax.set(xlim=(-18.1,-23.9),ylim=(.2,.9))
            if i==2: ax.set_xlabel(r'$M_r$ [mag]')
        axes[i,0].set_ylabel(r'$g-r$ [mag]')
    fig.colorbar(count_image,ax=axes[:,0].tolist(),location='bottom',pad=.04,shrink=.85,label=r'$\log_{10}(\mathrm{count\ per\ bin})$')
    fig.colorbar(score_image,ax=axes[:,1:].ravel().tolist(),location='bottom',pad=.04,shrink=.75,extend='both',label=r'$\langle S\rangle$ [dex]')
    fig.savefig(out/'joint_norm_cmd_full.pdf',dpi=400); fig.savefig(out/'joint_norm_cmd_full.png',dpi=400); plt.close(fig)
    np.savez(out/'bins_and_contours.npz',**bins)
    data.to_parquet(out/'plot_data.parquet',index=False)
    sources.extend(base/f'{a}_cond+z_matrices.npz' for a in mats)
    meta=dict(rows=len(data),splits=data.split.value_counts().to_dict(),equations=equations,min_count=25,kde_rows=len(sample),kde_seed=0,kde_mass_on_grid=float(mass[-1]),contours='10/30/50/70/90 percent of estimated KDE probability mass; counts and mean S use all matched rows',count_panels='Identical sample repeated in each seed row for direct comparison',selection='Frozen budget-15 validation selections; no refitting or reselection. Full sample is descriptive, not held-out evaluation.',input_sha256={str(p):digest(p) for p in sources})
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n'); print('Full sample:',len(data),meta['splits'])

if __name__=='__main__':main()
