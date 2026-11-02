"""Matched original latents + NORM + z control for fibre-mass SR."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from fibre_mass_conditioning import OUT,BASE,TARGET
from val_rescore import build_matrices
import matched_conditioning_controls as controls

def prepare():
    dest=OUT/'original_frames.pkl'
    if dest.exists():raise FileExistsError(dest)
    parent=BASE/'matched_controls/original_frames.pkl'
    source=pd.read_pickle(parent);reference=pd.read_pickle(OUT/'norm_frames.pkl');frames={};checks={}
    for split in ('train','val','test'):
        rows=np.load(OUT/f'{split}_parent_rows.npy');f=source[split].iloc[rows].reset_index(drop=True).copy();ref=reference[split]
        np.testing.assert_array_equal(f.specObjID.astype(str),ref.specObjID.astype(str))
        np.testing.assert_array_equal(np.stack(f.cond)[:,10],np.stack(ref.cond)[:,10])
        np.testing.assert_array_equal(f.z_x,ref.z_x)
        f[TARGET]=ref[TARGET].to_numpy();frames[split]=f
        checks[split]=dict(rows=len(f),ids_equal=True,norm_equal=True,z_equal=True)
    mats=build_matrices(frames,TARGET,'cond+z');refmat=np.load(OUT/'norm_matrices.npz')
    for split in frames:
        np.testing.assert_array_equal(mats[f'y_{split}'],refmat[f'y_{split}'])
        np.testing.assert_array_equal(mats[f'X_{split}'][:,10:],refmat[f'X_{split}'][:,10:])
    for seed in (42,43,44):
        idx=np.load(OUT/f'train_indices_seed{seed}.npy');ids=pd.read_csv(OUT/f'train_ids_seed{seed}.csv')
        np.testing.assert_array_equal(frames['train'].specObjID.iloc[idx].astype(str),ids.specObjID.astype(str))
    pd.to_pickle(frames,dest);np.savez(OUT/'original_matrices.npz',**mats)
    (OUT/'original_preparation.json').write_text(json.dumps(dict(target=TARGET,inputs='Ten original Spender latents + standardised log NORM + raw z',checks=checks,training_subsets='Identical saved 10000 indices and IDs for seeds 42,43,44',hashes={str(p):controls.digest(p) for p in [Path(__file__),parent,dest,OUT/'original_matrices.npz',*OUT.glob('train_ids_seed*.csv')]}),indent=2))
    print(checks,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit']);ap.add_argument('--seed',type=int,choices=[42,43,44]);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:
        if args.seed is None:ap.error('fit requires --seed')
        controls.OUT=OUT;controls.fit('original',args.seed,target=TARGET,prepared_indices=True)
