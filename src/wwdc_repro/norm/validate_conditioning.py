from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import pandas as pd
import wwdc_repro.shared.data as data
from wwdc_repro.shared.selection import score_expression as score, audit_candidate, select_candidate
from wwdc_repro.shared.utils import sha256

ROOT = Path(os.environ.get('DATA_ROOT', 'data'))
SPECS = {
    'norm': ('sdss_II/spender_I_normalisation_flow/embeddings/norm_20261012183217_0', 'normflow_condnorm_', True),
    'znorm': ('sdss_II/spender_I_z_normalisation_flow/embeddings/znorm_20261013104946_0', 'znormflow_condnorm_', True),
    'z': ('sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0', '', False),
} # paths embedding datasets and associated equation search results. Boolean is whether to append NORM to the symbolic regression eq.


def select(rows, budget, selection="train"):
    """Compatibility adapter for the historical conditioning CSV column names. Wrapper for select_candidate(...)"""
    return select_candidate(rows, budget, length_column='Length', validation_column='val_r2', selection=selection)

def prepare_inputs(out: Path) -> None:
    """Build matrices, ordered object IDs and candidate metadata from raw inputs.

    Parameters
    ----------
    out : Path
        Destination for matrices, split IDs, candidates.csv and manifest.json.
        Embeddings/catalogues are resolved through ROOT and SPECS; archived
        fronts are read relative to the working directory.

    Notes
    -----
    Does not audit, select, evaluate equations or rerun searches. Existing
    preparation files in out are replaced; saved search fronts are unchanged.
    Uses the shared redshift data pipeline. For NORM/joint flows, the final
    embedding conditioning coordinate is passed as an explicit scaled feature;
    the ten latent coordinates remain unchanged.
    """
    out.mkdir(parents=True, exist_ok=True) # make a new output dir
    candidates, manifest = [], {} 
    for cond, (relative, prefix, append) in SPECS.items():
        print('Loading', cond, flush=True)
        frames = data.clean_target(data.load_sample(
            ROOT / relative, ROOT / 'galSpecExtra-dr8.fits', ROOT / 'sdss'))
        if append:
            for frame in frames.values():
                y = np.stack(frame.y.values).reshape(len(frame), -1)
                # The final conditioning coordinate is NORM (the joint flow also has z).
                frame['norm'] = y[:, -1]
        manifest[cond] = {'embeddings': str(ROOT / relative), 'splits': {}}
        for split, frame in frames.items():
            # Record the ids associated with the data. Count and hash allows for checking later.
            ids = frame.specObjID.astype(str).to_numpy()
            np.save(out / f'{cond}_{split}_ids.npy', ids.astype(str))
            manifest[cond]['splits'][split] = {'rows': len(ids), 'ordered_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()}
        for arm in (['cond+z'] if cond == 'znorm' else ['cond', 'cond+z']):
            # make the data matrix for the given arm
            mats = data.matrices(frames, arm, scaled_columns=('norm',) if append else ())
            label = f'{cond}_{arm}'
            np.savez(out / f'{label}_matrices.npz', **mats) # save prepared matrices for later.
            for seed in (42, 43, 44):
                # load equations
                path = Path(f'job_results_{prefix}lgm_tot_p50_{arm}_seed{seed}/pareto_fronts.csv')
                raw = pd.read_csv(path) # read the learned equations
                for i, row in raw.iterrows():
                    candidates.append(dict(
                        arm=label, seed=seed, row=i, source=str(path),
                        Length=int(row.Length),
                        equation=row.get('Equation_p60', row.get('Equation_p12', row.Equation)),
                        export_precision=60 if 'Equation_p60' in raw.columns else 12 if 'Equation_p12' in raw.columns else 0,
                        train_r2_stored=row.Train_R2,
                    )) # append candidates equation for a given front.
        del frames # clean up
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)) # save meta data
    pd.DataFrame(candidates).to_csv(out / 'candidates.csv', index=False) # Collected equations along with scores saved to csv in out.


def load_candidates(out: Path) -> pd.DataFrame:
    """Helper function to load candidate equations

    Parameters
    ----------
    out : Path
        Prepared directory containing candidates.csv produced by prepare_inputs(...)

    Returns
    -------
    pandas.DataFrame
        Candidate identity, expression and native training-score reference.
    """
    path = out / 'candidates.csv'
    if not path.exists():
        path = out / 'candidate_audit.csv'
    loaded = pd.read_csv(path)
    columns = ['arm', 'seed', 'row', 'source', 'Length', 'equation', 'train_r2_stored']
    if 'export_precision' in loaded:
        columns.append('export_precision')
    return loaded[columns].copy()


def apply_recoveries(candidates: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, list]:
    """Recover archived NORM seed-42 exports unless fresh precise exports are supplied. 

    Parameters
    ----------
    candidates : pandas.DataFrame
        Candidate metadata returned by load_candidates.
    out : Path
        Prepared directory containing rerun manifests and native fronts.

    Returns
    -------
    pandas.DataFrame
        Copy with recovered expressions, native scores and recovery flags.
    list
        Verified recovery manifests for provenance.

    Raises
    ------
    ValueError
        If a recovery does not match its source hash, row count or lengths,
        or its manifest does not confirm an exact historical-front match.
    """
    candidates = candidates.copy()
    candidates['recovered_export'] = False
    recoveries = []
    for arm in ('norm_cond', 'norm_cond+z'):
        group = candidates[(candidates.arm == arm) & (candidates.seed == 42)]
        # Fresh fronts already carry full-precision exports and native train scores.
        if not group.empty and 'export_precision' in group and (group.export_precision >= 60).all():
            continue
        directory = out / f'rerun_{arm}_seed42'
        manifest = json.loads((directory / 'manifest.json').read_text())
        if not manifest['exact_front_match']:
            raise ValueError(f'{directory}: rerun did not reproduce the saved front')
        if group.empty or group.source.nunique() != 1:
            raise ValueError(f'{arm}: expected one source front for seed 42')
        source = Path(group.source.iloc[0])
        if sha256(source) != manifest['original_sha256']:
            raise ValueError(f'{source}: recovery source checksum differs')
        fresh = pd.read_csv(directory / 'native_front.csv')
        if len(fresh) != len(group) or sorted(group.row.tolist()) != list(range(len(fresh))):
            raise ValueError(f'{directory}: candidate rows do not match the recovered front')
        for i, row in group.iterrows():
            recovered = fresh.iloc[int(row.row)]
            if int(recovered.Length) != int(row.Length):
                raise ValueError(f'{directory}: recovered length differs at row {row.row}')
            candidates.loc[i, 'equation'] = recovered.Equation_p60
            candidates.loc[i, 'train_r2_stored'] = recovered.Train_R2
            candidates.loc[i, 'recovered_export'] = True
        recoveries.append(manifest)
    return candidates, recoveries


def audit_candidates(candidates: pd.DataFrame, out: Path, selection="train") -> pd.DataFrame:
    """Check candidate equations against training and validation data.
 
    Recreate each seed's training subsample and compare the evaluated score
    with the saved training score. In validation mode, also calculate validation
    scores and require finite scores across full training and validation splits.
    
    Parameters
    ----------
    candidates : pd.DataFrame
        Candidate equations after export recovery. Includes arm, seed,
        equation, native training-score reference (train_r2_stored),
        and candidate metadata.
    out : Path
        Directory containing each arm's saved matrices in
        <arm>_matrices.npz.
    
    Returns
    -------
    pd.DataFrame
        Candidate metadata and recalculated audit results, preserving input
        row order. Adds train_r2, train_delta, val_r2, selectable and selection.
        Validation scores are NaN when training selection is used.
    
    Notes
    -----
    Eligibility requires a finite training-subsample score agreeing with
    the native reference within an absolute tolerance of 1e-6, plus finite
    full-training predictions and a finite validation score.
    """
    records = {}
    for arm, group in candidates.groupby('arm', sort=False):
        with np.load(out / f'{arm}_matrices.npz') as saved:
            keys = ('X_train', 'y_train', 'X_val', 'y_val') if selection == 'validation' else ('X_train', 'y_train')
            mats = {key: saved[key] for key in keys}
        for seed, rows in group.groupby('seed', sort=False):
            idx = np.random.default_rng(seed).choice(
                len(mats['X_train']), min(10000, len(mats['X_train'])), replace=False) # recreate the training subsample
            for i, row in rows.iterrows():
                audit = audit_candidate(row.equation, mats, idx, row.train_r2_stored,
                                        check_full_training=selection == "validation", selection=selection) # compute training and validation scores
                records[i] = dict(row.to_dict(), **{key: audit[key] for key in
                    ('train_r2', 'train_delta', 'val_r2', 'selectable', 'selection')}) # store results from audit
        print(f'{arm}: audited {len(group)} candidates', flush=True)
    columns = ['arm', 'seed', 'row', 'source', 'Length', 'equation', 'train_r2',
               'train_r2_stored', 'train_delta', 'val_r2', 'selection',
               'selectable', 'recovered_export']
    return pd.DataFrame([records[i] for i in candidates.index], columns=columns)


def freeze_selection(audit: pd.DataFrame, out: Path, selection="train") -> pd.DataFrame:
    """Select best training (default) or validation scorers at length budgets L=5–29 and save them before test scoring.

    Parameters
    ----------
    audit : pandas.DataFrame
        Audited candidates without test scores.
    out : Path
        Destination for selection_frozen_before_test.csv.

    Returns
    -------
    pandas.DataFrame
        One eligible winner per arm, seed and available budget.
    """
    frozen = []
    for (arm, _), group in audit.groupby(['arm', 'seed'], sort=False):
        for budget in range(5, 30):
            row = select(group, budget, selection=selection)
            if row is not None:
                frozen.append(dict(row.to_dict(), budget=budget))
    frozen = pd.DataFrame(frozen, columns=[*audit.columns, 'budget'])
    frozen.to_csv(out / 'selection_frozen_before_test.csv', index=False) # save validation equations before evaluating on test.
    return frozen


def evaluate_selected(frozen: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Evaluate frozen equations on full and common-object test samples.

    Parameters
    ----------
    frozen : pandas.DataFrame
        Training- or validation-selected equations; failed test predictions are not replaced.
    out : Path
        Prepared matrix and split-ID directory.

    Returns
    -------
    pandas.DataFrame
        Frozen rows with test scores, failure status and sample counts.
    """
    test_ids = {flow: np.load(out / f'{flow}_test_ids.npy') for flow in ('norm', 'znorm', 'z')}
    common = set.intersection(*(set(ids) for ids in test_ids.values())) # get the spectra common across data. This 
    # is necessary as quality cuts using NORM (finite and positive values) are used and so the counts may differ slightly
    # between datasets.
    result = []
    for arm, group in frozen.groupby('arm', sort=False):
        with np.load(out / f'{arm}_matrices.npz') as saved:
            mats = {key: saved[key] for key in ('X_test', 'y_test')} # load test arrays
        flow = arm.split('_')[0]
        mask = np.isin(test_ids[flow], list(common)) # spectra that belong to common sample.
        for _, r in group.iterrows():
            test = score(r.equation, mats['X_test'], mats['y_test']) # evaluate the test score.
            matched = score(r.equation, mats['X_test'][mask], mats['y_test'][mask])
            result.append(dict(r.to_dict(), test_r2=test, common_test_r2=matched,
                               test_status='ok' if np.isfinite(test) else 'evaluation_failure',
                               test_rows=len(mask), common_test_rows=int(mask.sum())))
    result = pd.DataFrame(result) # save test scores.
    return result


def write_outputs(result: pd.DataFrame, out: Path) -> None:
    """Write selected scores and summaries for the completed validation.

    Parameters
    ----------
    result : pandas.DataFrame
        Frozen selections with test results from evaluate_selected.
    out : Path
        Destination directory for the selected-score and frontier-summary CSV tables.
    """
    result.to_csv(out / 'validation_selected_final.csv', index=False)
    final = result[result.budget == 29]
    final.to_csv(out / 'selected_equations_budget29.csv', index=False)
    def complete_mean(s):
        return float(s.mean()) if len(s) == 3 and np.isfinite(s).all() else np.nan # compute mean score
    curves = result.groupby(['arm', 'budget']).agg(
        mean_test_r2=('test_r2', complete_mean), min_test_r2=('test_r2', 'min'), 
        max_test_r2=('test_r2', 'max'), selected_seeds=('seed', 'nunique'),
        finite_tests=('test_r2', 'count'), common_mean_test_r2=('common_test_r2', complete_mean)).reset_index()
    curves['selection'] = result.selection.iloc[0]
    curves.loc[curves.finite_tests < 3, ['mean_test_r2', 'min_test_r2', 'max_test_r2']] = np.nan
    curves.to_csv(out / 'frontier_summary.csv', index=False) # save summary across arms and seed

def run(out: Path, selection="train") -> pd.DataFrame:
    """Recover, audit, select and score saved candidate equations.

    Parameters
    ----------
    out : Path
        Prepared directory; validation products are written here.

    Returns
    -------
    pandas.DataFrame
        Validation-selected equations and held-out scores.
    """
    candidates, _ = apply_recoveries(load_candidates(out), out)
    audit = audit_candidates(candidates, out, selection=selection)
    audit.to_csv(out / 'candidate_audit_final.csv', index=False)
    frozen = freeze_selection(audit, out, selection=selection)
    result = evaluate_selected(frozen, out)
    evaluate_selected(audit, out).to_csv(out / 'front_test_scores.csv', index=False)
    write_outputs(result, out)
    return result


def main():
    """Run saved-input validation, optionally rebuilding the raw-data inputs first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('conditioning_validation_2026-10-13'))
    parser.add_argument('--selection', choices=('train', 'validation'), default='train')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true', help='Rebuild raw-data inputs, then validate')
    modes.add_argument('--prepare-only', action='store_true', help='Rebuild inputs without auditing or scoring')
    args = parser.parse_args()
    if args.prepare or args.prepare_only:
        prepare_inputs(args.out)
    if not args.prepare_only:
        run(args.out, selection=args.selection)

if __name__ == '__main__':
    main()
