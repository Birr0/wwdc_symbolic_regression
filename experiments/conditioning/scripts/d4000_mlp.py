"""Broad D4000 latent-only small MLP control: real validation split; untouched test until checkpoint frozen."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import time,json,hashlib,copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import r2_score,mean_squared_error
BASE=Path('d4000_pilot_2026-10-16');OUT=BASE/'mlp'
ARMS=('original','zflow','norm');SEEDS=(42,43,44)
PARAMS=dict(hidden_layers=[64,32],activation='ReLU',optimizer='Adam',learning_rate=.001,batch_size=1024,max_epochs=500,patience=40,selection='minimum MSE on existing full validation split',inputs='ten latent coordinates only; all inputs standardized using the same 10000 training subset per seed',target='Broad D4000; standardized on fitted training subset')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 results=[];ref=np.load(BASE/'norm_matrices.npz')
 for arm in ARMS:
  data=np.load(BASE/f'{arm}_matrices.npz')
  for split in ['train','val','test']:
   np.testing.assert_array_equal(data[f'y_{split}'],ref[f'y_{split}'])
   np.testing.assert_array_equal(data[f'X_{split}'][:,10:],ref[f'X_{split}'][:,10:])
  for seed in SEEDS:
   start=time.time();torch.manual_seed(seed);np.random.seed(seed)
   idx=np.load(BASE/f'train_indices_seed{seed}.npy');x=data['X_train'][idx].astype(np.float64);y=data['y_train'][idx].astype(np.float64)
   mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;ym=float(y.mean());ys=float(y.std())
   def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=device)
   xt=tensor((x-mu)/sd);yt=tensor((y-ym)/ys).reshape(-1,1)
   xv=tensor((data['X_val'].astype(float)-mu)/sd);yv=tensor((data['y_val']-ym)/ys).reshape(-1,1)
   model=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1)).to(device)
   opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');best_epoch=0;state=None;history=[]
   for epoch in range(1,501):
    model.train();order=torch.randperm(len(xt),device=device)
    for batch in order.split(1024):
     opt.zero_grad(set_to_none=True);loss=nn.functional.mse_loss(model(xt[batch]),yt[batch]);loss.backward();opt.step()
    model.eval()
    with torch.no_grad():
     vl=float(nn.functional.mse_loss(model(xv),yv));tl=float(nn.functional.mse_loss(model(xt),yt))
    history.append(dict(epoch=epoch,train_mse=tl*ys**2,val_mse=vl*ys**2))
    if np.isfinite(vl) and vl<best:
     best=vl;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if epoch-best_epoch>=40:break
   if state is None:raise RuntimeError('No finite validation checkpoint')
   name=f'{arm}_seed{seed}'
   # Persist selected checkpoint before ever constructing test predictions.
   torch.save(dict(state_dict=state,input_mean=mu.tolist(),input_std=sd.tolist(),target_mean=ym,target_std=ys,best_epoch=best_epoch,parameters=PARAMS),OUT/f'{name}.pt')
   pd.DataFrame(history).to_csv(OUT/f'{name}_history.csv',index=False)
   model.load_state_dict(state);model.eval()
   with torch.no_grad():
    pred=model(tensor((data['X_test'].astype(float)-mu)/sd)).flatten().cpu().numpy()*ys+ym
    pv=model(xv).flatten().cpu().numpy()*ys+ym
   if not np.isfinite(pred).all():raise RuntimeError('Nonfinite selected test prediction; no replacement')
   np.save(OUT/f'{name}_test_predictions.npy',pred)
   row=dict(arm=arm,seed=seed,best_epoch=best_epoch,epochs_run=epoch,hit_epoch_cap=epoch==500,train_rows=len(idx),val_r2=r2_score(data['y_val'],pv),test_r2=r2_score(data['y_test'],pred),test_rmse=np.sqrt(mean_squared_error(data['y_test'],pred)),seconds=time.time()-start)
   results.append(row);pd.DataFrame(results).to_csv(OUT/'results.csv',index=False);print(row,flush=True)
 summary=pd.DataFrame(results).groupby('arm').agg(test_r2=('test_r2','mean'),test_r2_min=('test_r2','min'),test_r2_max=('test_r2','max'),test_rmse=('test_rmse','mean'),val_r2=('val_r2','mean'),capped_runs=('hit_epoch_cap','sum')).reset_index();summary.to_csv(OUT/'summary.csv',index=False)
 manifest=dict(parameters=PARAMS,device=str(device),torch_version=torch.__version__,deterministic_algorithms=True,seeds=SEEDS,hashes={str(p):digest(p) for p in [Path(__file__),*[BASE/f'{a}_matrices.npz' for a in ARMS],*[BASE/f'train_indices_seed{s}.npy' for s in SEEDS]]})
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
