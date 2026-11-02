"""Full matched CMD counts and S with paper-style KDE probability contours."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from figstyle import use_science, INK, INK_MUTED, SURFACE, tidy
from cmd_helpers import masters10_red_edge, masters10_ridge
OUT=Path('norm_cmd_full_2026-10-18');OUT.mkdir(exist_ok=True)
d=pd.read_parquet('norm_cmd_s_2026-10-16/plot_data.parquet')
xe=np.linspace(-24,-18,61);ye=np.linspace(.15,1.2,57)
counts,_,_=np.histogram2d(d.Mr,d.gr,bins=(xe,ye))
total,_,_=np.histogram2d(d.Mr,d.gr,bins=(xe,ye),weights=d.S)
mean=np.divide(total,counts,out=np.full_like(total,np.nan),where=counts>=25)
sample=d.sample(min(60000,len(d)),random_state=0)
kde=gaussian_kde(np.vstack([sample.Mr,sample.gr]))
kx=np.linspace(-24,-18,180);ky=np.linspace(.15,1.2,180);xx,yy=np.meshgrid(kx,ky)
density=kde(np.vstack([xx.ravel(),yy.ravel()])).reshape(xx.shape)
ranked=np.sort(density.ravel())[::-1];enclosed=np.cumsum(ranked)*(kx[1]-kx[0])*(ky[1]-ky[0])
fracs=np.array([.1,.3,.5,.7,.9]);assert enclosed[-1]>.9
levels=[ranked[np.searchsorted(enclosed,f)] for f in fracs]
use_science()
fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,sharey=True,gridspec_kw={'wspace':.30})
ordered=sorted(levels)
cf=axes[0].contourf(kx,ky,density,levels=[*ordered,density.max()],cmap='Greys',extend='neither')
axes[0].contour(kx,ky,density,levels=ordered,colors=[INK_MUTED],linewidths=.7,zorder=4)
cb0=fig.colorbar(cf,ax=axes[0],fraction=.046,pad=.02,ticks=ordered)
cb0.ax.set_yticklabels([f'{frac:.0%}' for frac in fracs[::-1]])
cb0.set_label('enclosed fraction of galaxies',fontsize=14)
im=axes[1].pcolormesh(xe,ye,mean.T,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.2,vcenter=0,vmax=.2),rasterized=True)
cb1=fig.colorbar(im,ax=axes[1],fraction=.046,pad=.02,ticks=[-.2,-.1,0,.1,.2],format='%.1f')
cb1.set_label(r'$\langle S\rangle$',fontsize=16)
grid=np.linspace(-24,-18,200)
for ax in axes:
 ax.plot(grid,masters10_red_edge(grid),linestyle=(0,(5.5,2.4)),color=INK,lw=1.6,zorder=5,label='Blue edge of the red sequence (Masters et al. 2010)')
 ax.plot(grid,masters10_ridge(grid),linestyle=(0,(1.2,1.6)),color=INK_MUTED,lw=1.3,zorder=5,label='Red-sequence fit (Masters et al. 2010)')
 ax.set(xlim=(-18,-24),ylim=(.15,1.0))
 ax.set_xlabel(r'$M_r$',fontsize=16);ax.set_ylabel(r'$g-r$',fontsize=16)
 tidy(ax,grid_axis=None,labelsize=12);ax.tick_params(labelleft=True)
leg=axes[0].legend(loc='lower left',fontsize=10,frameon=True,borderpad=.5)
leg.get_frame().set_edgecolor(INK_MUTED);leg.get_frame().set_linewidth(.6);leg.get_frame().set_facecolor(SURFACE)
for cb in (cb0,cb1):
 cb.outline.set_edgecolor(INK_MUTED);cb.outline.set_linewidth(.6);cb.ax.tick_params(labelsize=12)
fig.savefig(OUT/'norm_cmd_full.pdf',dpi=400,bbox_inches='tight',facecolor=SURFACE);fig.savefig(OUT/'norm_cmd_full.png',dpi=400,bbox_inches='tight',facecolor=SURFACE);plt.close(fig)
np.savez(OUT/'bins_and_contours.npz',counts=counts,mean_S=mean,xedges=xe,yedges=ye,kx=kx,ky=ky,density=density,contour_fractions=fracs,contour_levels=levels)
meta=dict(rows=len(d),splits=d.split.value_counts().to_dict(),kde_rows=len(sample),kde_seed=0,kde_bandwidth=float(kde.factor),kde_grid=180,kde_mass_on_grid=float(enclosed[-1]),min_count=25,colour_correlation=float(d.S.corr(d.gr)),contours='10,30,50,70,90 percent of estimated KDE probability mass; not exact empirical sample fractions',counts='All matched spectra; retained for minimum-count masking of S, not displayed in left panel',left_panel='Filled grey KDE probability contours matching the original-latent CMD style')
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
