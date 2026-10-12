"""Pilot: 10k existing training spectra; validation stopping; untouched test scoring."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hidden', nargs=2, type=int, default=[128, 64])
    p.add_argument('--prepared-from', type=Path)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--flux-transform', choices=['identity', 'asinh'], default='identity')
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)
    os.environ['DATA_ROOT'] = str(args.data_root)
    sys.path.insert(0, str(args.upstream / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from wwdc_spectra.data.sdss import SDSS
    from wwdc_repro.data import load_sample, clean_target
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    frames = clean_target(load_sample(
        args.data_root / 'sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0',
        args.data_root / 'galSpecExtra-dr8.fits', args.data_root / 'sdss'))
    from wwdc_repro.frontier import front_path, score_expression
    from wwdc_repro.data import matrices
    # Exactly the seed-specific SR draw, including parent-row order.
    indices = np.random.default_rng(seed=args.seed).choice(
        len(frames['train']), size=10000, replace=False)
    data = matrices(frames, 'z')
    front = pd.read_csv(front_path(Path(__file__).resolve().parents[1] / 'artifacts/fronts', 'z', args.seed))
    reproduced = [abs(score_expression(row.Equation, data['X_train'][indices], data['y_train'][indices]) - row.Train_R2) for row in front.itertuples()]
    finite_errors = [value for value in reproduced if np.isfinite(value)]
    if not finite_errors or min(finite_errors) > 1e-6:
        raise ValueError('Could not reproduce saved SR training score')
    frames['train'] = frames['train'].iloc[indices].copy()
    print(f'SR seed {args.seed} subsample verified against saved training scores', flush=True)
    arrays, targets = {}, {}
    for split, frame in frames.items():
        if args.prepared_from:
            expected = pd.read_csv(args.prepared_from / f'{split}_rows.csv')
            assert np.array_equal(expected.specObjID.to_numpy(), frame.specObjID.to_numpy())
            assert np.allclose(expected.lgm_tot_p50, frame.lgm_tot_p50, rtol=0, atol=1e-6)
            arrays[split] = np.load(args.prepared_from / f'{split}_spectra.npy')
            targets[split] = frame.lgm_tot_p50.to_numpy(dtype=np.float32)
            frame[['specObjID', 'lgm_tot_p50']].to_csv(args.outdir / f'{split}_rows.csv', index=False)
            continue
        ds = SDSS(split=split, y_catalog={'variables': {'Z': {}}, 'drop_variables': []}, return_id=True)
        ids = [int(re.search(r'\d+', str(v)).group()) for v in ds.dataset['object_id']]
        lookup = {identity: i for i, identity in enumerate(ids)}
        assert len(lookup) == len(ids)
        positions = [lookup[int(v)] for v in frame.specObjID]
        matrix = np.empty((len(frame), len(ds._wave_obs)), dtype=np.float32)
        for j, index in enumerate(positions):
            item = ds[index]
            assert int(re.search(r'\d+', str(item[2])).group()) == int(frame.specObjID.iloc[j])
            matrix[j] = item[0].numpy()
            if (j+1) % 5000 == 0:
                print(f'prepared {split} {j+1}/{len(frame)}', flush=True)
        assert np.isfinite(matrix).all()
        np.save(args.outdir / f'{split}_spectra.npy', matrix)
        arrays[split] = matrix
        targets[split] = frame.lgm_tot_p50.to_numpy(dtype=np.float32)
        frame[['specObjID', 'lgm_tot_p50']].to_csv(args.outdir / f'{split}_rows.csv', index=False)
        print(f'{split}: {matrix.shape}', flush=True)
    transform_scale = None
    if args.flux_transform == 'asinh':
        # One robust scalar fitted to training pixels only; no validation/test tuning.
        transform_scale = float(np.median(np.abs(arrays['train'])))
        if not np.isfinite(transform_scale) or transform_scale <= 0:
            raise ValueError('Training median absolute flux must be positive and finite')
        arrays = {split: np.arcsinh(values / transform_scale)
                  for split, values in arrays.items()}
        print(f'asinh scale from training median absolute flux: {transform_scale}', flush=True)
    scaler = StandardScaler().fit(arrays['train'])
    ymean, yscale = float(targets['train'].mean()), float(targets['train'].std())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = {s: torch.from_numpy(scaler.transform(a).astype(np.float32)).to(device) for s,a in arrays.items()}
    y = {s: torch.from_numpy((v-ymean)/yscale).to(device) for s,v in targets.items()}
    model = torch.nn.Sequential(torch.nn.Linear(x['train'].shape[1],args.hidden[0]),torch.nn.ReLU(),torch.nn.Linear(*args.hidden),torch.nn.ReLU(),torch.nn.Linear(args.hidden[1],1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    def predict(split):
        model.eval()
        with torch.inference_mode():
            return torch.cat([model(batch).squeeze(1) for batch in x[split].split(2048)])
    history, best, best_epoch, best_state = [], float('inf'), 0, None
    start = time.time()
    for epoch in range(1,301):
        model.train()
        order = torch.randperm(len(x['train']), device=device)
        total = 0.0
        for idx in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x['train'][idx]).squeeze(1), y['train'][idx])
            loss.backward()
            optimizer.step()
            total += loss.item()*len(idx)
        val = torch.nn.functional.mse_loss(predict('val'), y['val']).item()
        history.append({'epoch':epoch,'train_mse_dex2':total/len(order)*yscale**2,'val_mse_dex2':val*yscale**2})
        if val < best:
            best, best_epoch, best_state = val, epoch, copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f'epoch {epoch}: val RMSE={val**0.5*yscale:.5f}; best epoch={best_epoch}',flush=True)
        if epoch-best_epoch >= 25:
            break
    model.load_state_dict(best_state)
    result = {'seed':args.seed,'architecture':[x['train'].shape[1],*args.hidden,1], 'parameters':sum(p.numel() for p in model.parameters()), 'training_sample':'Exact SR default_rng(seed).choice, without replacement', 'sr_training_score_min_absolute_error':min(finite_errors), 'ordered_training_ids_sha256':hashlib.sha256(frames['train'].specObjID.to_numpy(dtype='<i8').tobytes()).hexdigest(), 'activation':'ReLU','optimizer':'AdamW','learning_rate':args.learning_rate,'weight_decay':1e-4,'batch_size':256,'patience':25,'max_epochs':300,'best_epoch':best_epoch,'epochs_run':epoch,'device':str(device),'training_seconds':time.time()-start,'preprocessing':'Pinned upstream SDSS normalized observed-frame spectrum, then per-pixel standardization fitted only on the 10000 training rows. Target standardized on those same rows. No explicit redshift input.','split_policy':'Archived spectrum-level splits preserved; no object regrouping. Full eligible validation/test samples.','upstream_source_sha256':hashlib.sha256((args.upstream/'src/wwdc_spectra/data/sdss.py').read_bytes()).hexdigest(),'command':sys.argv,'torch_version':torch.__version__,'metrics':{}}
    result['flux_transform'] = args.flux_transform
    result['flux_transform_scale'] = transform_scale
    result['flux_transform_scale_rule'] = 'median absolute pixel flux across only the 10000 training spectra' if transform_scale is not None else None
    if transform_scale is not None:
        result['preprocessing'] = result['preprocessing'].replace('then per-pixel', 'then asinh(flux / training-median-absolute-flux), then per-pixel')
    for split in frames:
        pred = predict(split).cpu().numpy()*yscale+ymean
        truth = targets[split]
        result['metrics'][split] = {'n':len(truth),'r2':float(r2_score(truth,pred)),'rmse_dex':float(mean_squared_error(truth,pred)**0.5),'mae_dex':float(mean_absolute_error(truth,pred))}
        pd.DataFrame({'specObjID':frames[split].specObjID.to_numpy(),'truth':truth,'prediction':pred}).to_csv(args.outdir/f'{split}_predictions.csv',index=False)
    torch.save({'state_dict':{k:v.cpu() for k,v in best_state.items()},'input_mean':scaler.mean_,'input_scale':scaler.scale_,'target_mean':ymean,'target_scale':yscale,'flux_transform':args.flux_transform,'flux_transform_scale':transform_scale},args.outdir/'model.pt')
    pd.DataFrame(history).to_csv(args.outdir/'history.csv',index=False)
    (args.outdir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    main()
