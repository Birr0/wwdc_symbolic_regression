"""Matched SR controls: identical rows, normalisation, redshift and search settings.

prepare; fit --arm original|zflow --seed 42|43|44. Original fronts remain untouched.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from val_rescore import build_merged, build_matrices, evaluate_equation
from validate_conditioning import ROOT, SPECS, score, select

BASE=Path('conditioning_validation_2026-10-13')
OUT=BASE/'matched_controls'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def clean(frames):
    return {k:v.dropna(subset=['lgm_tot_p50']).query('lgm_tot_p50 != -9999').reset_index(drop=True) for k,v in frames.items()}

def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'preparation.json').exists() or list(OUT.glob('*_frames.pkl')):
        raise FileExistsError('Refusing to overwrite prepared controls')
    norm=clean(build_merged(str(ROOT/SPECS['norm'][0]),str(ROOT/'galSpecExtra-dr8.fits'),str(ROOT/'sdss')))
    old=clean(build_merged(str(ROOT/SPECS['z'][0]),str(ROOT/'galSpecExtra-dr8.fits'),str(ROOT/'sdss')))
    frames={'original':{},'zflow':{}}
    summary={}
    for split,frame in norm.items():
        ids=frame.specObjID.astype(str).to_numpy()
        np.testing.assert_array_equal(ids,np.load(BASE/f'norm_{split}_ids.npy'))
        np.testing.assert_array_equal(ids,np.load(BASE/f'znorm_{split}_ids.npy'))
        # Occurrence index preserves repeated spectra instead of a many-to-many join.
        left=pd.DataFrame({'id':ids,'row':np.arange(len(ids))})
        right=pd.DataFrame({'id':old[split].specObjID.astype(str),'oldrow':np.arange(len(old[split]))})
        for f in (left,right):f['occurrence']=f.groupby('id',sort=False).cumcount()
        aligned=left.merge(right,on=['id','occurrence'],how='left',validate='one_to_one',sort=False)
        assert aligned.oldrow.notna().all(), 'normalisation rows missing from old embeddings'
        np.testing.assert_array_equal(aligned.row,np.arange(len(frame)))
        of=old[split].iloc[aligned.oldrow.astype(int)].reset_index(drop=True)
        np.testing.assert_array_equal(of.specObjID.astype(str),ids)
        np.testing.assert_array_equal(of.lgm_tot_p50,frame.lgm_tot_p50)
        np.testing.assert_allclose(of.z_x,frame.z_x,atol=0,rtol=0)
        orig=np.stack(frame.orig);oldorig=np.stack(of.orig)
        # Independently generated embeddings differ at floating-point inference
        # precision; IDs, targets and redshifts above must still match exactly.
        np.testing.assert_allclose(orig,oldorig,atol=1e-4,rtol=0)
        normval=np.stack(frame.y).reshape(len(frame),1)
        for arm,latent in [('original',orig),('zflow',np.stack(of.cond))]:
            frames[arm][split]=pd.DataFrame({'cond':list(np.hstack([latent,normval])),
                'z_x':frame.z_x.to_numpy(),'lgm_tot_p50':frame.lgm_tot_p50.to_numpy(),'specObjID':frame.specObjID.to_numpy()})
        np.save(OUT/f'{split}_ids.npy',ids.astype(str))
        summary[split]=dict(rows=len(frame),repeated_ids=int(left.id.duplicated().sum()),
            old_orig_max_abs_difference=float(np.max(np.abs(orig-oldorig))),ordered_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
    for arm in frames:
        pd.to_pickle(frames[arm],OUT/f'{arm}_frames.pkl')
        mats=build_matrices(frames[arm],'lgm_tot_p50','cond+z')
        reference=np.load(BASE/'norm_cond+z_matrices.npz')
        for split in frames[arm]:
            np.testing.assert_array_equal(mats[f'y_{split}'],reference[f'y_{split}'])
            np.testing.assert_array_equal(mats[f'X_{split}'][:,10:],reference[f'X_{split}'][:,10:])
        np.savez(OUT/f'{arm}_matrices.npz',**mats)
    for seed in (42,43,44):
        idx=np.random.default_rng(seed).choice(len(norm['train']),10000,replace=False)
        np.save(OUT/f'train_indices_seed{seed}.npy',idx)
        pd.DataFrame({'row':idx,'specObjID':norm['train'].specObjID.iloc[idx].to_numpy()}).to_csv(OUT/f'train_ids_seed{seed}.csv',index=False)
    manifest=dict(samples=summary,inputs='Exact NORM flow conditioning values, appended to original or aligned redshift-flow latents; common rows and existing split assignments retained.',
        policy='Best validation R2 per seed at each length budget 5..29; finite full-train and validation predictions; native training agreement <=1e-6; test evaluated only after freezing selections.',
        sha256={str(p):digest(p) for p in [Path(__file__),Path('utils.py'),Path('val_rescore.py'),*OUT.glob('*frames.pkl'),*OUT.glob('train_ids*.csv')]})
    (OUT/'preparation.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

def fit(arm,seed,target='lgm_tot_p50',prepared_indices=False,parameter_overrides=None):
    if parameter_overrides and not prepared_indices:
        raise ValueError('parameter_overrides requires prepared_indices')
    from utils import fit_sym_fn
    dest=OUT/f'{arm}_seed{seed}';dest.mkdir(exist_ok=False)
    frames=pd.read_pickle(OUT/f'{arm}_frames.pkl')
    mats=np.load(OUT/f'{arm}_matrices.npz');idx=np.load(OUT/f'train_indices_seed{seed}.npy')
    started=time.time();print('START',arm,seed,flush=True)
    parameter_source=None
    if prepared_indices:
        # A target-specific invalid row must not silently redraw the 10k sample.
        # Instantiate the identical recorded search configuration on explicit rows.
        from pyoperon.sklearn import SymbolicRegressor
        parameter_source=BASE/'matched_controls'/f'zflow_seed{seed}/manifest.json'
        parameters=json.loads(parameter_source.read_text())['parameters']
        parameters.update(parameter_overrides or {})
        assert parameters['random_state']==seed
        estimator=SymbolicRegressor(**parameters)
        estimator.fit(mats['X_train'][idx],mats['y_train'][idx])
        result=dict(Model=estimator,X_train=mats['X_train'][idx],y_train=mats['y_train'][idx],X_test=mats['X_test'],y_test=mats['y_test'])
    else:
        result=fit_sym_fn(target,'cond+z',frames,seed)
    np.testing.assert_array_equal(result['X_train'],mats['X_train'][idx])
    np.testing.assert_array_equal(result['y_train'],mats['y_train'][idx])
    model=result['Model'];records=[];trees={}
    for i,member in enumerate(model.pareto_front_):
        tree=member['tree'];trees[i]=tree
        eq=model.get_model_string(tree,precision=60)
        native=model.evaluate_model(tree,result['X_train'])
        native_r2=float(r2_score(result['y_train'],native)) if np.isfinite(native).all() else np.nan
        tr=score(eq,result['X_train'],result['y_train'])
        full=score(eq,mats['X_train'],mats['y_train']);val=score(eq,mats['X_val'],mats['y_val'])
        ok=bool(np.isfinite(native_r2) and np.isfinite(tr) and abs(tr-native_r2)<=1e-6 and np.isfinite(full) and np.isfinite(val))
        records.append(dict(arm=arm,seed=seed,row=i,Length=int(tree.Length),equation=eq,
            native_mdl=float(member['minimum_description_length']),native_bic=float(member['bayesian_information_criterion']),native_aic=float(member['akaike_information_criterion']),
            equation_p12=model.get_model_string(tree,precision=12),train_r2_stored=native_r2,train_r2=tr,
            train_delta=abs(tr-native_r2),train_full_finite=bool(np.isfinite(full)),val_r2=val,selectable=ok))
    audit=pd.DataFrame(records);audit.to_csv(dest/'candidate_audit.csv',index=False)
    chosen=[]
    for budget in range(5,model.get_params()['max_length']+5):
        r=select(audit,budget)
        if r is not None:chosen.append(dict(r.to_dict(),budget=budget))
    chosen=pd.DataFrame(chosen);chosen.to_csv(dest/'selection_frozen_before_test.csv',index=False)
    evaluations=[]
    for r in chosen.itertuples():
        te=score(r.equation,mats['X_test'],mats['y_test'])
        native=model.evaluate_model(trees[r.row],mats['X_test'])
        nt=float(r2_score(mats['y_test'],native)) if np.isfinite(native).all() else np.nan
        evaluations.append(dict(r._asdict(),test_r2=te,native_test_r2=nt,test_status='ok' if np.isfinite(te) else 'evaluation_failure'))
    pd.DataFrame(evaluations).to_csv(dest/'selected.csv',index=False)
    manifest=dict(arm=arm,seed=seed,target=target,prepared_indices=prepared_indices,parameter_overrides=parameter_overrides or {},parameter_source=str(parameter_source) if parameter_source else None,elapsed_seconds=time.time()-started,parameters=model.get_params(),
        native_information_criteria={'stage':'after outer scale and intercept are appended to the native tree','likelihood':'gauss','sigma':model.get_params()['uncertainty'],'selection':'recorded only; validation remains the selection criterion'},
        versions={p:importlib.metadata.version(p) for p in ('pyoperon','numpy','scikit-learn','sympy','pandas')},
        input_hashes={str(p):digest(p) for p in [OUT/f'{arm}_frames.pkl',OUT/f'train_ids_seed{seed}.csv',Path('utils.py'),Path(__file__)]})
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    print('DONE',arm,seed,'minutes',manifest['elapsed_seconds']/60,'eligible',int(audit.selectable.sum()),'of',len(audit),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['prepare','fit']);ap.add_argument('--arm',choices=['original','zflow']);ap.add_argument('--seed',type=int,choices=[42,43,44]);args=ap.parse_args()
    if args.stage=='prepare':prepare()
    else:
        if args.arm is None or args.seed is None:ap.error('fit requires --arm and --seed')
        fit(args.arm,args.seed)
