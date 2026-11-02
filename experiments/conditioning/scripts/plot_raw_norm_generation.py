"""Inspect saved NORM traversals without any post-decoder rescaling."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import use_science

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--outdir',type=Path,default=Path('norm_generation_random_seed42'))
OUT=ap.parse_args().outdir
a=np.load(OUT/'generated_spectra.npz')
w=a['wave_rest'];raw=a['decoded_raw']; factors=a['factors']
n_examples,n_factors=raw.shape[:2]
assert n_factors==len(factors)
window=(w>5300)&(w<5850); display=(w>=3700)&(w<=7500)
med=np.median(raw[:,:,window],axis=2)
rows=[]
for i in range(n_examples):
 for j,f in enumerate(factors):
  rows.append(dict(example=i+1,norm_factor=f,window_median=float(med[i,j]),percent_from_unity=float(100*(med[i,j]-1)),max_rescaling_percent=float(100*abs(1/med[i,j]-1))))
pd.DataFrame(rows).to_csv(OUT/'raw_decoder_medians.csv',index=False)
use_science();plt.rcParams.update({'font.size':12,'axes.labelsize':14})
fig,axes=plt.subplots(n_examples,2,figsize=(13,3.4*n_examples),squeeze=False,layout='constrained',gridspec_kw={'width_ratios':[3,1.3]})
colours=plt.cm.viridis(np.linspace(.05,.95,n_factors))
for i in range(n_examples):
 for j,f in enumerate(factors):
  axes[i,0].plot(w[display],raw[i,j,display],color=colours[j],lw=.9,label=rf'$N/N_{{\rm ref}}={f:g}$')
  axes[i,1].scatter(f,med[i,j],color=colours[j],s=40,zorder=3)
 axes[i,0].plot(w[display],a['original_reconstruction'][i,display],'k--',lw=.8,label='Original reconstruction')
 axes[i,0].set(xlim=(3700,7500),ylabel='Raw decoder output',title=f'Example {i+1}: no post-decoder rescaling')
 axes[i,1].plot(factors,med[i],color='.6',lw=.8,zorder=1)
 axes[i,1].axhline(1,color='.3',ls='--',lw=1)
 axes[i,1].set(ylim=(min(.996,float(med.min())-.001),max(1.004,float(med.max())+.001)),xticks=factors,ylabel='Median output',title=r'5300--5850 $\mathrm{\AA}$ window')
 axes[i,1].ticklabel_format(axis='y',style='plain',useOffset=False)
axes[0,0].legend(fontsize=9,ncol=2)
axes[-1,0].set_xlabel(r'Rest wavelength [$\mathrm{\AA}$]')
axes[-1,1].set_xlabel(r'Requested $N/N_{\rm ref}$')
fig.savefig(OUT/'raw_decoder_norm_sequence.pdf',dpi=350)
fig.savefig(OUT/'raw_decoder_norm_sequence.png',dpi=300)
plt.close(fig)
(OUT/'RAW_OUTPUT_CHECK.md').write_text('# Raw decoder output check\n\nNo new generation or normalization is performed. These are the saved decoder outputs before median division or amplitude restoration.\n\n'
 f'The {med.size} window medians range from {med.min():.7f} to {med.max():.7f}. The largest absolute departure from unity is {100*np.max(abs(med-1)):.4f}%. '
 f'The largest multiplicative change imposed by dividing by the median is {100*np.max(abs(1/med-1)):.4f}%.\n\n'
 f'For these {n_examples} examples and {n_factors} conditions, the decoder already follows the unit-median convention closely. The additional normalization in the earlier plot makes a small correction; it does not create the substantial imposed amplitude differences in its right panels. This is a finite-sample check, not a hard decoder constraint.\n\n'
 'See raw_decoder_medians.csv and raw_decoder_norm_sequence.pdf. Reproduce with ../../../.venv/bin/python plot_raw_norm_generation.py from the symbolic_regression directory.\n')
print((OUT/'RAW_OUTPUT_CHECK.md').read_text())
