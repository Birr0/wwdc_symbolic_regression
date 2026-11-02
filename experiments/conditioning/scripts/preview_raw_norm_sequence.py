"""Preview raw decoder spectra and raw output times requested NORM; no median division."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from figstyle import use_science


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True)
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    a=np.load(args.input/'generated_spectra.npz');obs=np.load(args.input/'observed_spectra.npz')
    wave=a['wave_rest'];raw=a['decoded_raw'];factors=a['factors'];nref=a['reference_norm']
    requested=nref[:,None]*factors[None,:];restored=raw*requested[:,:,None]
    mask=(wave>=3700)&(wave<=7500);window=(wave>5300)&(wave<5850)
    median=np.median(raw[:,:,window],axis=2)
    assert raw.shape[:2]==requested.shape
    assert np.isfinite(raw).all() and np.isfinite(restored).all()
    use_science();plt.rcParams.update({'font.size':12,'axes.labelsize':14})
    fig,axes=plt.subplots(len(nref),2,figsize=(14,3.4*len(nref)),squeeze=False,sharex=True,layout='constrained')
    scale=Normalize(vmin=float(factors.min()),vmax=float(factors.max()))
    colours=plt.cm.viridis(scale(factors))
    for i in range(len(nref)):
        axes[i,0].plot(obs['wave_rest'][i],obs['normalised_flux'][i],color='.5',lw=.65,alpha=.8,zorder=2)
        axes[i,1].plot(obs['wave_rest'][i],obs['flux'][i],color='.5',lw=.65,alpha=.8,zorder=4,label='Observed spectrum')
        for j in range(len(factors)):
            axes[i,0].plot(wave[mask],raw[i,j,mask],color=colours[j],lw=.9)
            axes[i,1].plot(wave[mask],restored[i,j,mask],color=colours[j],lw=.9)
        axes[i,0].set_title(f"Spectrum ID: {obs['specObjID'][i]}")
        axes[i,1].set_title(rf'Decoder output $\times\,N$; $N_{{\rm ref}}={nref[i]:.2f}$')
        axes[i,0].set_ylabel('Decoded normalised flux')
        axes[i,1].set_ylabel('Flux density\n'+r'[$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]')
        axes[i,1].legend(fontsize=10,loc='lower right',frameon=True)
        for ax in axes[i]:ax.set_xlim(3700,7500)
    for ax in axes[-1]:ax.set_xlabel(r'Rest wavelength [$\mathrm{\AA}$]')
    fig.colorbar(ScalarMappable(norm=scale,cmap='viridis'),ax=axes,location='right',pad=.02,fraction=.025,shrink=.6,ticks=factors,label=r'Relative normalisation $N/N_{\rm ref}$')
    fig.savefig(args.outdir/'raw_decoder_sequence_preview.pdf',dpi=400)
    fig.savefig(args.outdir/'raw_decoder_sequence_preview.png',dpi=200)
    plt.close(fig)
    meta=dict(input=str(args.input.resolve()),method='Left: saved decoded_raw unchanged. Right: decoded_raw times reference_norm times requested factor. No post-decoder median division.',minimum_window_median=float(median.min()),maximum_window_median=float(median.max()),max_previous_rescaling_percent=float(100*np.max(abs(1/median-1))),hashes={name:hashlib.sha256((args.input/name).read_bytes()).hexdigest() for name in ['generated_spectra.npz','observed_spectra.npz']})
    (args.outdir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
