"""Hold one conditional flow seed fixed per test spectrum and vary NORM."""
from pathlib import Path
import os,sys,json,argparse
os.environ.setdefault('DATA_ROOT', str(Path('data').resolve()))
if os.environ.get('WWDC_UPSTREAM'):
 sys.path.insert(0, str(Path(os.environ['WWDC_UPSTREAM'])/'src'))
import numpy as np
import pandas as pd
import torch
import yaml
from flow_matching.solver import ODESolver
from wwdc_spectra.models.model import LightningFlowMatching, WrappedModel
from wwdc_spectra.models.load_spender import PretrainedSpender
from wwdc_spectra.data.sdss import SDSS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from figstyle import use_science
from matched_conditioning_controls import digest

ROOT=Path(os.environ['DATA_ROOT'])/'sdss_II/spender_I_normalisation_flow'
TAG='norm_20261012183217_0'
OUT=Path('norm_generation_random_seed42')

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--factors',type=float,nargs='+',default=[.5,.75,1.,1.5,2.])
 ap.add_argument('--seed',type=int,default=42)
 ap.add_argument('--n-spectra',type=int,default=3)
 ap.add_argument('--outdir',type=Path,default=OUT)
 args=ap.parse_args()
 factors=np.asarray(args.factors,dtype=float)
 if not np.isfinite(factors).all() or (factors<=0).any():ap.error('factors must be finite and positive')
 if args.n_spectra<1:ap.error('n-spectra must be positive')
 out=args.outdir.resolve()
 out.mkdir(exist_ok=True)
 torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
 cfg=yaml.safe_load((ROOT/f'ckpts/{TAG}.yaml').read_text())
 mu=cfg['normalisation']['train']['mean'];sd=cfg['normalisation']['train']['std']
 raw=pd.read_parquet(ROOT/f'embeddings/{TAG}/test/embeddings.parquet')
 raw['specObjID']=raw.id.astype(str).str.extract(r'(\d+)')[0].astype('int64')
 raw['context']=np.stack(raw.y).reshape(-1)
 phot=pd.read_parquet('norm_cmd_s_2026-10-16/plot_data.parquet').query('split == "test"')
 pool=raw.merge(phot[['specObjID','gr']].drop_duplicates('specObjID'),on='specObjID',validate='many_to_one')
 pool=pool[np.isfinite(pool.context)].sort_values('specObjID',kind='stable').reset_index(drop=True)
 if args.n_spectra>len(pool):ap.error('n-spectra exceeds available test rows')
 positions=np.random.default_rng(args.seed).choice(len(pool),args.n_spectra,replace=False)
 chosen=[pool.iloc[i] for i in positions]
 n_examples,n_factors=len(chosen),len(factors)
 base=PretrainedSpender('sdss_I',10)
 model=LightningFlowMatching(base_model=base,lr=.0001,batch_size=512,code_dim=10,hidden_dim=1024,n_layers=3,catalog=cfg['data']['y_catalog'],n_steps=20)
 checkpoint=ROOT/f'ckpts/{TAG}.ckpt'
 model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict'],strict=True)
 model.to(device).eval();solver=ODESolver(velocity_model=WrappedModel(model.vf))
 def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=device)
 def flow(x,y,reverse=False,steps=100):
  if x.shape[0]!=y.shape[0]:raise ValueError(f'Latent/context batch mismatch: {x.shape} vs {y.shape}')
  return solver.sample(x_init=x,y=y,cfg_scale=1.,time_grid=tensor([1.,0.] if reverse else [0.,1.]),step_size=1/steps,method='midpoint')
 orig=tensor(np.stack([r.orig for r in chosen]));ctx=tensor([[r.context] for r in chosen])
 nref=10**(ctx.cpu().numpy().ravel()*sd+mu)
 contexts=ctx.cpu().numpy()+np.log10(factors)[None,:]/sd
 train=pd.read_parquet(ROOT/f'embeddings/{TAG}/train/embeddings.parquet',columns=['y'])
 training_context=np.stack(train.y).ravel();bounds=np.quantile(training_context,[.01,.99])
 outside=(contexts<bounds[0])|(contexts>bounds[1])
 with torch.inference_mode():
  seed=flow(orig,ctx,reverse=True)
  stored=tensor(np.stack([r.cond for r in chosen]));re20=flow(orig,ctx,reverse=True,steps=20)
  stored_error=float((stored-re20).abs().max())
  assert stored_error<1e-3, f'Saved conditional embedding mismatch: {stored_error}'
  gen=flow(seed.repeat_interleave(n_factors,dim=0),tensor(contexts.reshape(-1,1)))
  gen200=flow(seed.repeat_interleave(n_factors,dim=0),tensor(contexts.reshape(-1,1)),steps=200)
  spectra=model.base_model.decoder(gen).cpu().numpy().reshape(n_examples,n_factors,-1)
  spectra200=model.base_model.decoder(gen200).cpu().numpy().reshape(n_examples,n_factors,-1)
  original_spectra=model.base_model.decoder(orig).cpu().numpy()
  recovered=flow(seed,ctx)
  recovered_spectra=model.base_model.decoder(recovered).cpu().numpy()
 wave=model.base_model.model.wave_rest.detach().cpu().numpy()
 window=(wave>5300)&(wave<5850)
 med=np.median(spectra[:,:,window],axis=2)
 assert np.isfinite(spectra).all() and (med>0).all()
 shapes=spectra/med[:,:,None]
 restored=spectra*(nref[:,None]*factors[None,:])[:,:,None]
 original_shapes=original_spectra/np.median(original_spectra[:,window],axis=1)[:,None]
 # Retrieve the actual input spectra, preserving the upstream masks and scale.
 dataset=SDSS(split='test',y_catalog={'variables':{'Z':{}},'drop_variables':[]})
 source_ids=pd.Series(dataset.dataset['object_id']).astype(str).str.extract(r'(\d+)')[0].astype('int64')
 observed=[];observed_wave=[];observed_norms=[];source_rows=[];encoder_errors=[]
 for i,r in enumerate(chosen):
  matches=np.flatnonzero(source_ids.to_numpy()==int(r.specObjID))
  if len(matches)!=1:raise ValueError(f'Expected one input spectrum for {r.specObjID}, found {len(matches)}')
  source_row=int(matches[0]);item=dataset.dataset[source_row];s=item['spectrum']
  np.testing.assert_allclose(float(item['Z']),float(r.z),atol=1e-7,rtol=0)
  spectrum,weights,norm,restwave,_=dataset.prepare_spectrum(s['flux'],s['ivar'],s['lambda'],s['mask'],item['Z'])
  np.testing.assert_allclose(float(norm),nref[i],rtol=2e-6)
  with torch.inference_mode():encoded=model.base_model.encode(spectrum[None].to(device))
  error=float((encoded-orig[i:i+1]).abs().max());encoder_errors.append(error)
  assert error<1e-4,f'Input spectrum does not reproduce saved original latent: {error}'
  values=spectrum.numpy().copy();values[(weights.numpy()<=0)|~np.isfinite(values)]=np.nan
  observed.append(values);observed_wave.append(restwave.numpy());observed_norms.append(float(norm));source_rows.append(source_row)
 observed=np.stack(observed);observed_wave=np.stack(observed_wave);observed_norms=np.array(observed_norms)
 np.savez(out/'observed_spectra.npz',wave_rest=observed_wave,normalised_flux=observed,
  flux=observed*observed_norms[:,None],normalisation=observed_norms,source_test_rows=source_rows,
  specObjID=np.array([int(r.specObjID) for r in chosen],dtype=np.int64))
 mask=(wave>=3700)&(wave<=7500)
 roundtrip=float(np.max(np.abs(recovered.cpu().numpy()-orig.cpu().numpy())))
 recovered_shapes=recovered_spectra/np.median(recovered_spectra[:,window],axis=1)[:,None]
 meta=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),condition_mean=mu,condition_std=sd,
  observed_source_test_rows=source_rows,observed_encoder_max_abs_errors=encoder_errors,
  selection_seed=args.seed,selection='Uniform random rows without replacement from finite-context matched CMD test sample; no colour or NORM-magnitude selection',eligible_rows=len(pool),sampled_row_positions=positions.tolist(),outside_training_1_99_count=int(outside.sum()),factors=factors.tolist(),training_context_1_99_percentiles=bounds.tolist(),generated_contexts=contexts.tolist(),
  rows=[dict(specObjID=int(r.specObjID),gr=float(r.gr),z=float(r.z),reference_norm=float(nref[i])) for i,r in enumerate(chosen)],
  saved_20step_embedding_max_abs=stored_error,roundtrip_latent_max_abs=roundtrip,
  decoder_100_200_step_max_abs=float(np.max(abs(spectra-spectra200))),
  shape_reference_max_abs=float(np.max(abs(recovered_shapes-original_shapes))),
  array_schema=2,shape_array_usage='Diagnostic median-normalised shape retained for comparison only; not plotted',
  method='Invert original Spender latents at original NORM with 100-step midpoint, hold each base seed fixed; forward 100-step midpoint at the requested NORM values. Training-percentile checks are recorded without changing the random sample. No redshift conditioning.',
  plotting='Rest-wavelength decoder grid. Left: raw decoded spectra with no post-decoder median division. Right: multiply raw decoder output by requested NORM. Restoring amplitude is explicit, not predicted by the decoder. No luminosity conversion. Grey curves show the observed spectra with masked pixels omitted; reconstruction curves are not plotted.')
 np.savez(out/'generated_spectra.npz',wave_rest=wave,decoded_raw=spectra,shape=shapes,amplitude_restored=restored,
  original_reconstruction=original_spectra,conditional_seed=seed.cpu().numpy(),generated_latents=gen.cpu().numpy().reshape(n_examples,n_factors,10),factors=factors,reference_norm=nref)
 use_science();plt.rcParams.update({'font.size':12,'axes.labelsize':14})
 fig,axes=plt.subplots(n_examples,2,figsize=(14,3.4*n_examples),squeeze=False,sharex=True,layout='constrained')
 colour_norm=Normalize(vmin=float(factors.min()),vmax=float(factors.max()) if factors.max()>factors.min() else float(factors.min()+1))
 colours=plt.cm.viridis(colour_norm(factors))
 for i,r in enumerate(chosen):
  axes[i,0].plot(observed_wave[i],observed[i],color='.5',lw=.65,alpha=.8,zorder=2,label='Observed spectrum')
  axes[i,1].plot(observed_wave[i],observed[i]*observed_norms[i],color='.5',lw=.65,alpha=.8,zorder=4,label='Observed spectrum')
  for j,f in enumerate(factors):
   label=rf'$N/N_{{\rm ref}}={f:g}$'
   axes[i,0].plot(wave[mask],spectra[i,j,mask],color=colours[j],lw=.9,label=label)
   axes[i,1].plot(wave[mask],restored[i,j,mask],color=colours[j],lw=.9)
  axes[i,1].legend(fontsize=10,loc='lower right',frameon=True)
  axes[i,0].set_title(f'Spectrum ID: {int(r.specObjID)}')
  axes[i,1].set_title(rf'Decoder output $\times\,N$; $N_{{\rm ref}}={nref[i]:.2f}$')
  axes[i,0].set_ylabel('Decoded normalised flux')
  axes[i,1].set_ylabel('Flux density\n'+r'[$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]')
  for ax in axes[i]: ax.set_xlim(3700,7500)
 for ax in axes[-1]:ax.set_xlabel(r'Rest wavelength [$\mathrm{\AA}$]')
 fig.colorbar(ScalarMappable(norm=colour_norm,cmap='viridis'),ax=axes,location='right',pad=.02,fraction=.025,shrink=.6,ticks=np.unique(factors),label=r'Relative normalisation $N/N_{\rm ref}$')
 fig.savefig(out/'norm_conditional_sequence.pdf',dpi=350)
 fig.savefig(out/'norm_conditional_sequence.png',dpi=300)
 fig.savefig(out/'norm_conditional_sequence_labelled.pdf',dpi=350)
 fig.savefig(out/'norm_conditional_sequence_labelled.png',dpi=300)
 fig.savefig(out/'norm_conditional_sequence_with_observed.pdf',dpi=350)
 fig.savefig(out/'norm_conditional_sequence_with_observed.png',dpi=300)
 fig.savefig(out/'norm_sequence_colourbar.pdf',dpi=350)
 fig.savefig(out/'norm_sequence_colourbar.png',dpi=300)
 plt.close(fig)
 (out/'metadata.json').write_text(json.dumps(meta,indent=2))
 (out/'README.md').write_text('# Conditional NORM traversal\n\nRandom matched test examples selected without replacement using the recorded seed (default 42), without selecting on colour or NORM amplitude. Each fixed conditional seed is transported to the requested NORM factors using the trained NORM-only flow, then decoded with its frozen Spender model.\n\nLeft panels show raw decoder outputs without post-decoder median division; right panels multiply those outputs by the requested NORM. The shape array is retained only as a diagnostic and is not used for either panel. Grey curves show the observed spectra with masked pixels omitted. Reconstruction curves are not plotted. The shared colour bar gives the requested NORM factor. These are model-defined conditional traversals, not guaranteed physical changes to one galaxy. NORM-only conditioning does not hold redshift or stellar population fixed. No redshift wavelength shift is applied. The metadata records how many requested conditions fall outside the training 1st–99th percentiles; random examples are not discarded on this basis.\n\nFull arrays and provenance/numerical checks are saved alongside the PDF and PNG. Reproduce from this directory’s parent: `../../../.venv/bin/python generate_norm_sequence.py`.\n')
 print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
