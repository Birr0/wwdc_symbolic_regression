"""Fibre-mass SR for z-, NORM-, and jointly conditioned latent representations.

All three searches use conditional latents + exact NORM + raw z. Stages:
prepare; fit --arm zflow|norm|znorm --seed 42|43|44.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
from val_rescore import build_merged,build_matrices
from validate_conditioning import ROOT,SPECS
import matched_conditioning_controls as controls

BASE=Path('conditioning_validation_2026-10-13')
OUT=Path('fibre_mass_conditioning_2026-10-14')
TARGET='LGM_FIB_P50'
ARMS=('zflow','norm','znorm')

def prepare():
    OUT.mkdir(exist_ok=False)
    frames={arm:{} for arm in ARMS};summary={}
    source=build_merged(str(ROOT/SPECS['norm'][0]),str(ROOT/'galSpecExtra-dr8.fits'),str(ROOT/'sdss'))
    joint=build_merged(str(ROOT/SPECS['znorm'][0]),str(ROOT/'galSpecExtra-dr8.fits'),str(ROOT/'sdss'))
    zframes=pd.read_pickle(BASE/'matched_controls/zflow_frames.pkl')
    for split in ('train','val','test'):
        ids=np.load(BASE/'matched_controls'/f'{split}_ids.npy').astype('int64')
        def align(frame):
            assert not frame.specObjID.duplicated().any()
            indexed=frame.set_index('specObjID',drop=False)
            return indexed.loc[ids].reset_index(drop=True)
        n=align(source[split]);j=align(joint[split]);z=zframes[split].reset_index(drop=True)
        np.testing.assert_array_equal(z.specObjID,ids)
        np.testing.assert_array_equal(n[TARGET],j[TARGET])
        valid=np.isfinite(n[TARGET].to_numpy())&(n[TARGET].to_numpy()!=-9999)
        norm=np.stack(n.y).reshape(len(n),1)
        np.testing.assert_array_equal(norm,np.stack(j.y)[:,-1:])
        latents={'norm':np.hstack([np.stack(n.cond),norm]),'znorm':np.hstack([np.stack(j.cond),norm]),'zflow':np.stack(z.cond)}
        for arm in ARMS:
            frame=pd.DataFrame({'cond':list(latents[arm]),'z_x':n.z_x.to_numpy(),TARGET:n[TARGET].to_numpy(),'specObjID':ids})
            frames[arm][split]=frame[valid].reset_index(drop=True)
        np.save(OUT/f'{split}_ids.npy',ids[valid].astype(str))
        np.save(OUT/f'{split}_parent_rows.npy',np.flatnonzero(valid))
        summary[split]=dict(parent_rows=len(ids),rows=int(valid.sum()),excluded_invalid_fibre_mass=int((~valid).sum()),fibre_mass_min=float(n.loc[valid,TARGET].min()),fibre_mass_max=float(n.loc[valid,TARGET].max()),ordered_ids_sha256=hashlib.sha256('\n'.join(ids[valid].astype(str)).encode()).hexdigest())
    reference=None
    for arm in ARMS:
        pd.to_pickle(frames[arm],OUT/f'{arm}_frames.pkl')
        mats=build_matrices(frames[arm],TARGET,'cond+z')
        if reference is None:reference=mats
        else:
            for split in ('train','val','test'):
                np.testing.assert_array_equal(mats[f'y_{split}'],reference[f'y_{split}'])
                np.testing.assert_array_equal(mats[f'X_{split}'][:,10:],reference[f'X_{split}'][:,10:])
        np.savez(OUT/f'{arm}_matrices.npz',**mats)
        parent=np.load(BASE/('matched_controls/zflow_matrices.npz' if arm=='zflow' else f'{arm}_cond+z_matrices.npz'))
        if all(summary[s]['excluded_invalid_fibre_mass']==0 for s in summary):
            for split in summary:np.testing.assert_array_equal(mats[f'X_{split}'],parent[f'X_{split}'])
    subset_checks={}
    kept_parent=np.load(OUT/'train_parent_rows.npy')
    inverse=np.full(summary['train']['parent_rows'],-1,dtype=int)
    inverse[kept_parent]=np.arange(len(kept_parent))
    for seed in (42,43,44):
        parent_idx=np.load(BASE/'matched_controls'/f'train_indices_seed{seed}.npy')
        idx=inverse[parent_idx]
        missing=idx<0
        if missing.any():
            available=np.setdiff1d(np.arange(len(kept_parent)),idx[~missing])
            idx[missing]=np.random.default_rng(seed).choice(available,int(missing.sum()),replace=False)
        subset_checks[seed]=dict(replaced_invalid_targets=int(missing.sum()),shared_training_objects=int((~missing).sum()))
        np.save(OUT/f'train_indices_seed{seed}.npy',idx)
        pd.DataFrame({'row':idx,'specObjID':frames['norm']['train'].specObjID.iloc[idx].to_numpy()}).to_csv(OUT/f'train_ids_seed{seed}.csv',index=False)
        if not missing.any():
            np.testing.assert_array_equal(kept_parent[idx],parent_idx)
    manifest=dict(target=TARGET,target_definition='MPA-JHU median log10 stellar mass inside fibre; no additional log transform',samples=summary,training_subsets=subset_checks,
        models='zflow: redshift-conditioned; norm: NORM-conditioned; znorm: jointly conditioned',
        inputs='Ten conditional latents, exact standardised-log NORM conditioning value, raw z. Full matched training scaler; same seeded 10000 rows across arms.',
        hashes={str(p):controls.digest(p) for p in [Path(__file__),Path('matched_conditioning_controls.py'),Path('utils.py'),*OUT.glob('*_frames.pkl'),*OUT.glob('train_ids_seed*.csv')]})
    (OUT/'preparation.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest['samples'],indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','fit']);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--seed',type=int,choices=[42,43,44]);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:
        if args.arm is None or args.seed is None:ap.error('fit requires --arm and --seed')
        controls.OUT=OUT
        controls.fit(args.arm,args.seed,target=TARGET,prepared_indices=True)
