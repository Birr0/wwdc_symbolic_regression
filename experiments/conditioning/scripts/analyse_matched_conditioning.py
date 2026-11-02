"""Prediction/complexity sensitivity and latent-direction diagnostics.

--available permits analysing existing arms while matched control searches run.
All selections were frozen on validation before these diagnostics.
"""
from pathlib import Path
import argparse
import json
from itertools import combinations
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from astropy.cosmology import FlatLambdaCDM
from val_rescore import evaluate_equation, to_sympy
from validate_conditioning import select
from figstyle import use_science

BASE=Path('conditioning_validation_2026-10-13')
OUT=BASE/'matched_controls'
BUDGETS=list(range(12,19))+[20,24,29]
LABELS={'norm':'NORM-conditioned','znorm':'z+NORM-conditioned','original':'Original latents','zflow':'z-conditioned'}

def complete_mean(values):
    return float(values.mean()) if len(values)==3 and np.isfinite(values).all() else np.nan

def gradient(eq,x,h=1e-4):
    zero=x.copy();zero[:,:10]=0
    base=evaluate_equation(eq,zero)
    cols=[]
    for k in range(10):
        xp=zero.copy();xm=zero.copy();xp[:,k]+=h;xm[:,k]-=h
        cols.append((evaluate_equation(eq,xp)-evaluate_equation(eq,xm))/(2*h))
    return base,np.column_stack(cols)

def inputs(available):
    selected=pd.read_csv(BASE/'validation_selected_final.csv')
    audited=pd.read_csv(BASE/'candidate_audit_final.csv')
    groups={}
    for arm in ('norm','znorm'):
        name=arm+'_cond+z'
        groups[arm]=(selected[selected.arm==name].assign(arm=arm),audited[audited.arm==name].assign(arm=arm),BASE/f'{name}_matrices.npz')
    for arm in ('original','zflow'):
        if not all((OUT/f'{arm}_seed{s}/manifest.json').exists() for s in (42,43,44)):
            if available:continue
            raise FileNotFoundError(f'{arm}: not all three searches are complete')
        groups[arm]=(pd.concat([pd.read_csv(OUT/f'{arm}_seed{s}/selected.csv') for s in (42,43,44)],ignore_index=True),
            pd.concat([pd.read_csv(OUT/f'{arm}_seed{s}/candidate_audit.csv') for s in (42,43,44)],ignore_index=True),OUT/f'{arm}_matrices.npz')
    return groups

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--available',action='store_true');args=ap.parse_args()
    destination=OUT/('preliminary_existing_arms' if args.available else 'comparison')
    destination.mkdir(exist_ok=True)
    groups=inputs(args.available)
    allsel=[];structures=[];weights={};scores={};deltas={};approximations=[];tolerances=[];physical=[]
    sigma=json.loads((BASE/'physical_interpretation/metadata.json').read_text())['log_norm_train_std']
    cosmo=FlatLambdaCDM(H0=70,Om0=.3)
    for arm,(selected,audited,path) in groups.items():
        mats=np.load(path);x=mats['X_test'].astype(float);y=mats['y_test']
        varval=np.var(mats['y_val']);vartest=np.var(y)
        selected=selected.copy()
        selected['validation_rmse']=np.sqrt(np.maximum(0,1-selected.val_r2)*varval)
        selected['test_rmse']=np.sqrt(np.maximum(0,1-selected.test_r2)*vartest)
        allsel.append(selected)
        # These are zero-latent reference sensitivities, not interventions on a
        # fixed physical galaxy. Retain the distinction for unconditioned controls.
        zero=x.copy();zero[:,:10]=0;h=1e-4;z=x[:,11]
        zp=z*10**h;zm=z*10**(-h)
        dlogd=np.log10(cosmo.luminosity_distance(zp).value)-np.log10(cosmo.luminosity_distance(zm).value)
        for r in selected[selected.budget==15].itertuples():
            xp=zero.copy();xm=zero.copy();xp[:,10]+=h/sigma;xm[:,10]-=h/sigma
            df=(evaluate_equation(r.equation,xp)-evaluate_equation(r.equation,xm))/(2*h)
            xp=zero.copy();xm=zero.copy();xp[:,11]=zp;xm[:,11]=zm
            dd=(evaluate_equation(r.equation,xp)-evaluate_equation(r.equation,xm))/dlogd
            physical.append(dict(arm=arm,seed=r.seed,Length=r.Length,flux_median=float(np.median(df)),flux_p16=float(np.percentile(df,16)),flux_p84=float(np.percentile(df,84)),distance_median=float(np.median(dd)),expected_distance_median=float(np.median(2+(np.log10(1+zp)-np.log10(1+zm))/dlogd))))
        for seed,group in audited.groupby('seed'):
            eligible=group[group.selectable].copy()
            eligible['rmse']=np.sqrt(np.maximum(0,1-eligible.val_r2)*varval)
            best=eligible.rmse.min()
            for delta in (.002,.005,.01,.02):
                choice=eligible[eligible.rmse<=best+delta].sort_values(['Length','rmse']).iloc[0]
                tolerances.append(dict(arm=arm,seed=seed,tolerance_dex=delta,Length=int(choice.Length),validation_rmse=choice.rmse,increase_over_best=choice.rmse-best))
        for row in selected[selected.budget.isin(BUDGETS)].itertuples():
            print('Structure',arm,row.seed,row.budget,'selected length',row.Length,flush=True)
            # Reuse results when the same expression persists across budgets.
            key=(arm,row.seed,row.Length)
            cached=next((d for d in structures if (d['arm'],d['seed'],d['Length'])==key),None)
            if cached is not None:
                structures.append(dict(cached,budget=row.budget));continue
            idx=np.load(OUT/f'train_indices_seed{row.seed}.npy')
            xt=mats['X_train'][idx].astype(float)
            _,gt=gradient(row.equation,xt)
            finite=np.isfinite(gt).all(axis=1)
            if not finite.any():raise ValueError('no finite gradient rows')
            _,singular,vt=np.linalg.svd(gt[finite],full_matrices=False)
            energy=float(np.sum(singular**2))
            w=vt[0]
            if np.mean(gt[finite]@w)<0:w=-w
            weights[key]=w;scores[key]=x[:,:10]@w
            base,g=gradient(row.equation,x)
            full=evaluate_equation(row.equation,x);delta=full-base;deltas[key]=delta
            linear=np.sum(g*x[:,:10],axis=1)
            rank1=(g@w)*scores[key]
            valid=np.isfinite(delta)&np.isfinite(linear)&np.isfinite(rank1)
            expr=to_sympy(row.equation,12)
            latent={sp.Symbol(f'X{i}') for i in range(1,11)}
            grads=[sp.diff(expr,v) for v in sorted(latent,key=str)]
            affine=all(not (dg.free_symbols & latent) for dg in grads)
            constant_gradient=all(not dg.free_symbols for dg in grads)
            for approx,estimate in [('linear',base+linear),('rank1',base+rank1)]:
                approximations.append(dict(arm=arm,seed=row.seed,Length=row.Length,kind=approx,
                    valid_rows=int(valid.sum()),rmse_vs_equation=float(np.sqrt(np.mean((estimate[valid]-full[valid])**2))),
                    test_r2=float(1-np.mean((estimate[valid]-y[valid])**2)/np.var(y[valid]))))
            structures.append(dict(arm=arm,seed=row.seed,budget=row.budget,Length=row.Length,
                latent_count=len(expr.free_symbols & latent),exactly_affine_in_latents=affine,
                constant_latent_coefficients=constant_gradient,
                rank1_gradient_energy=float(singular[0]**2/energy) if energy>1e-20 else np.nan,
                gradient_rank_1e6=int(np.sum(singular>singular[0]*1e-6)) if energy>1e-20 else 0,
                rank1_approx_rmse=float(np.sqrt(np.mean((rank1[valid]-delta[valid])**2))),
                linear_approx_rmse=float(np.sqrt(np.mean((linear[valid]-delta[valid])**2))),
                latent_correction_sd=float(np.std(delta[valid])),finite_reference_rows=int(valid.sum()),
                test_score_sd=float(np.std(scores[key])),equation=row.equation))
    sel=pd.concat(allsel,ignore_index=True);sel.to_csv(destination/'selected_all_budgets.csv',index=False)
    last=sel[sel.budget==29][['arm','seed','validation_rmse','test_rmse']].rename(columns={'validation_rmse':'validation_rmse_at29','test_rmse':'test_rmse_at29'})
    nearby=sel[sel.budget.isin(BUDGETS)].merge(last,on=['arm','seed'],validate='many_to_one')
    nearby['validation_rmse_cost']=nearby.validation_rmse-nearby.validation_rmse_at29
    nearby['test_rmse_cost']=nearby.test_rmse-nearby.test_rmse_at29
    nearby.to_csv(destination/'budget_sensitivity_by_seed.csv',index=False)
    summary=nearby.groupby(['arm','budget']).agg(mean_validation_rmse=('validation_rmse',complete_mean),mean_test_rmse=('test_rmse',complete_mean),mean_test_r2=('test_r2',complete_mean),min_test_r2=('test_r2','min'),max_test_r2=('test_r2','max'),mean_validation_rmse_cost=('validation_rmse_cost',complete_mean),n=('seed','nunique')).reset_index()
    summary.to_csv(destination/'budget_sensitivity_summary.csv',index=False)
    pd.DataFrame(structures).to_csv(destination/'structure_by_budget.csv',index=False)
    pd.DataFrame(approximations).to_csv(destination/'approximation_checks.csv',index=False)
    pd.DataFrame(tolerances).to_csv(destination/'rmse_tolerance_selections.csv',index=False)
    pd.DataFrame(physical).to_csv(destination/'physical_slopes_budget15.csv',index=False)
    stability=[]
    for budget in BUDGETS:
        for arm in groups:
            sub=nearby[(nearby.arm==arm)&(nearby.budget==budget)].set_index('seed')
            for sa,sb in combinations((42,43,44),2):
                ka=(arm,sa,int(sub.loc[sa,'Length']));kb=(arm,sb,int(sub.loc[sb,'Length']))
                stability.append(dict(arm=arm,budget=budget,seed_a=sa,seed_b=sb,direction_cosine=float(weights[ka]@weights[kb]),score_correlation=float(np.corrcoef(scores[ka],scores[kb])[0,1])))
    pd.DataFrame(stability).to_csv(destination/'direction_stability.csv',index=False)
    paired=[]
    for ar,br in combinations(groups,2):
        for seed in (42,43,44):
            ra=nearby[(nearby.arm==ar)&(nearby.budget==15)&(nearby.seed==seed)].iloc[0]
            rb=nearby[(nearby.arm==br)&(nearby.budget==15)&(nearby.seed==seed)].iloc[0]
            ka=(ar,seed,int(ra.Length));kb=(br,seed,int(rb.Length))
            mask=np.isfinite(deltas[ka])&np.isfinite(deltas[kb])
            paired.append(dict(arm_a=ar,arm_b=br,seed=seed,score_pearson=float(np.corrcoef(scores[ka],scores[kb])[0,1]),score_spearman=float(spearmanr(scores[ka],scores[kb]).statistic),correction_pearson=float(np.corrcoef(deltas[ka][mask],deltas[kb][mask])[0,1])))
    pd.DataFrame(paired).to_csv(destination/'paired_scores_budget15.csv',index=False)
    # Compare the new scores directly with the historical L=24 canonical score.
    historical=pd.read_csv(BASE/'validation_selected_final.csv')
    old=np.load(BASE/'z_cond+z_matrices.npz')
    oldids=np.load(BASE/'z_test_ids.npy');commonids=np.load(OUT/'test_ids.npy')
    mapping=pd.Index(oldids).get_indexer(commonids)
    assert (mapping>=0).all()
    old_comparison=[]
    for seed in (42,43,44):
        r=historical[(historical.arm=='z_cond+z')&(historical.seed==seed)&(historical.budget==24)].iloc[0]
        ix=np.random.default_rng(seed).choice(len(old['X_train']),10000,replace=False)
        _,g=gradient(r.equation,old['X_train'][ix].astype(float))
        _,singular,vt=np.linalg.svd(g,full_matrices=False);w=vt[0]
        if np.mean(g@w)<0:w=-w
        previous_score=old['X_test'][mapping,:10]@w
        for arm in groups:
            chosen=nearby[(nearby.arm==arm)&(nearby.seed==seed)&(nearby.budget==15)].iloc[0]
            new_score=scores[(arm,seed,int(chosen.Length))]
            old_comparison.append(dict(seed=seed,old_budget=24,old_chosen_length=int(r.Length),new_arm=arm,new_budget=15,new_length=int(chosen.Length),matched_rows=len(mapping),pearson=float(np.corrcoef(previous_score,new_score)[0,1]),spearman=float(spearmanr(previous_score,new_score).statistic),old_rank1_energy=float(singular[0]**2/np.sum(singular**2))))
    pd.DataFrame(old_comparison).to_csv(destination/'old_canonical_score_comparison.csv',index=False)
    use_science();fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for arm in groups:
        sub=sel[sel.arm==arm].groupby('budget').agg(val=('validation_rmse',complete_mean),test=('test_r2',complete_mean),lo=('test_r2','min'),hi=('test_r2','max')).reset_index()
        line,=axes[0].plot(sub.budget,sub.val,label=LABELS[arm]);axes[1].plot(sub.budget,sub.test,color=line.get_color(),label=LABELS[arm]);axes[1].fill_between(sub.budget,sub.lo,sub.hi,color=line.get_color(),alpha=.12)
    axes[0].set(xlabel='Expression-length budget',ylabel='Validation RMSE [dex]')
    axes[1].set(xlabel='Expression-length budget',ylabel='Test $R^2$')
    for ax in axes:ax.axvline(15,color='grey',ls=':',lw=1);ax.legend(fontsize=8)
    fig.savefig(destination/'matched_frontiers.pdf');fig.savefig(destination/'matched_frontiers.png',dpi=300);plt.close(fig)
    print(summary[summary.budget.isin([12,15,18,29])].to_string(index=False),flush=True)

if __name__=='__main__':main()
