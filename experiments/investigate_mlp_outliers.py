"""Inspect saved MLP failures without changing samples or refitting models."""
import json
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
LARGE = ROOT / 'outputs/spectra-mlp-sr42-1024-512'
SMALL = ROOT / 'outputs/spectra-mlp-sr42-128-64'
OUT = ROOT / 'outputs/spectra-mlp-outlier-audit'
UPSTREAM = Path('/tmp/wwdc-spectra-reproduction-check')
os.environ['DATA_ROOT'] = '/home/birr0/local_data'
sys.path.insert(0, str(UPSTREAM/'src'))
from wwdc_spectra.data.sdss import SDSS

torch.set_num_threads(4)
OUT.mkdir(exist_ok=True)
models = {}
for name, directory in [('large', LARGE), ('small', SMALL)]:
    ckpt = torch.load(directory/'model.pt', map_location='cpu', weights_only=False)
    widths = json.loads((directory/'metrics.json').read_text())['architecture']
    model = torch.nn.Sequential(torch.nn.Linear(widths[0],widths[1]),torch.nn.ReLU(),torch.nn.Linear(widths[1],widths[2]),torch.nn.ReLU(),torch.nn.Linear(widths[2],1)).eval()
    model.load_state_dict(ckpt['state_dict'])
    models[name] = (model, ckpt)

def predict(name, arr):
    model, ckpt = models[name]
    x = ((arr.astype(np.float64)-ckpt['input_mean'])/ckpt['input_scale']).astype(np.float32)
    with torch.inference_mode():
        return model(torch.from_numpy(x)).squeeze(-1).numpy()*ckpt['target_scale']+ckpt['target_mean']

rows, population = [], {}
fig, axes = plt.subplots(3,2,figsize=(12,10),layout='constrained')
for si, split in enumerate(['train','val','test']):
    table = pd.read_csv(LARGE/f'{split}_predictions.csv')
    small = pd.read_csv(SMALL/f'{split}_predictions.csv')
    spectra = np.load(LARGE/f'{split}_spectra.npy',mmap_mode='r')
    error = np.abs(table.prediction-table.truth).to_numpy()
    peaks = np.max(np.abs(spectra),axis=1)
    population[split] = {'peak_abs_flux_quantiles':dict(zip(['50%','90%','99%','99.9%','max'],np.quantile(peaks,[.5,.9,.99,.999,1]).tolist())), 'target_range':[float(table.truth.min()),float(table.truth.max())]}
    table.assign(absolute_error=error,peak_abs_flux=peaks).to_csv(OUT/f'{split}_all_residuals.csv',index=False)
    chosen = sorted(set(np.argsort(error)[-5:]) | set(np.argsort(np.abs(small.prediction-small.truth))[-5:]))
    ds = SDSS(split=split,y_catalog={'variables':{'Z':{}},'drop_variables':[]})
    lookup = {int(re.search(r'\d+',str(v)).group()):i for i,v in enumerate(ds.dataset['object_id'])}
    largest = int(np.argmax(error))
    for idx in chosen:
        identity = int(table.specObjID.iloc[idx])
        item = ds.dataset[lookup[identity]]
        sp = item['spectrum']
        prepared, w, norm, rest, z = ds.prepare_spectrum(sp['flux'],sp['ivar'],sp['lambda'],sp['mask'],item['Z'])
        arr = prepared.numpy(); weights = w.numpy(); bad = weights <= 0
        assert np.array_equal(arr,spectra[idx])
        row = {'split':split,'specObjID':identity,'truth':float(table.truth.iloc[idx]),'large_prediction_saved':float(table.prediction.iloc[idx]),'small_prediction_saved':float(small.prediction.iloc[idx]),'normalization':float(norm),'redshift':float(z),'masked_fraction':float(bad.mean()),'raw_peak_abs_flux':float(np.nanmax(np.abs(sp['flux']))),'processed_peak_abs_flux':float(np.abs(arr).max()),'processed_peak_percentile':float((peaks<=np.abs(arr).max()).mean()*100),'peak_pixel_masked':bool(bad[np.argmax(np.abs(arr))]),'peak_wavelength_A':float(ds._wave_obs[np.argmax(np.abs(arr))]),'masked_peak_abs_flux':float(np.abs(arr[bad]).max()) if bad.any() else 0,'unmasked_peak_abs_flux':float(np.abs(arr[~bad]).max()) if (~bad).any() else 0,'reprepared_matches_cache':True}
        for name in models:
            row[name+'_prediction_cpu'] = float(predict(name,arr[None])[0])
            # Diagnostic interventions only: no metric or training selection uses these.
            row[name+'_prediction_masked_pixels_zeroed'] = float(predict(name,np.where(bad,0,arr)[None])[0])
        rows.append(row)
        if idx==largest:
            wave = ds._wave_obs.numpy()
            axes[si,0].plot(wave,arr,lw=.6)
            axes[si,0].scatter(wave[bad],arr[bad],s=2,c='red',label='zero-weight pixels')
            axes[si,0].set(title=f'{split}: {identity}\ntrue {row["truth"]:.2f}, predicted {row["large_prediction_saved"]:.2f}',ylabel='normalized flux')
            axes[si,0].legend(fontsize=7)
            axes[si,1].scatter(peaks,error,s=2,alpha=.3)
            axes[si,1].set(xscale='log',yscale='log',xlabel='maximum absolute normalized flux',ylabel='absolute mass error [dex]',title=f'{split}: all rows')
    print(f'finished {split}',flush=True)
result=pd.DataFrame(rows)
result.to_csv(OUT/'top_outliers.csv',index=False)
(OUT/'population_summary.json').write_text(json.dumps(population,indent=2)+'\n')
for ax in axes[:,0]: ax.set_xlabel('observed wavelength [Å]')
fig.savefig(OUT/'outliers.pdf');fig.savefig(OUT/'outliers.png',dpi=150)
print(result.sort_values(['split','large_prediction_saved']).to_string(index=False),flush=True)
