"""Matched 10k total-mass MLP benchmark: preparation, fitting and CSV/LaTeX export.

``prepare --config FILE --data-root RAW --outdir CACHE``
creates the matched spectral arrays from the existing SDSS split dataset.
Alternatively, give ``fit`` --spectra-data-root RAW
to prepare its cache automatically before fitting.

Run with ``python -m wwdc_repro.redshift.stellar_mass_mlp fit --config FILE
--outdir DIR`` (or execute this file directly). Original latents are always
fitted; --spectra-dir additionally fits the full normalised observed-frame
spectra. That directory must contain {train,val,test}_spectra.npy and matching
{train,val,test}_rows.csv with specObjID and lgm_tot_p50 columns. Its rows must
exactly match the seed-specific SR draw and the existing validation/test sets.

``export --run DIR --outdir TABLES`` verifies completion hashes and exports
selected-model results without fitting. No archived scores are substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Also support running the one file without an editable package installation.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wwdc_repro.shared import mlp as shared_mlp
from wwdc_repro.shared.data import clean_target, latent_array, load_sample, spectrum_ids
from wwdc_repro.shared.search_runs import input_hashes, positive_int, search_run
from wwdc_repro.shared.utils import sha256

SPLITS = ('train', 'val', 'test')
WIDTHS = ((128, 64), (1024, 512))
RATES = (1e-3, 1e-4)


def sample_sources(paths):
    """List files defining the shared sample for both MLP input representations.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Resolved configuration paths. Requires ``galspec`` for the MPA-JHU
        catalogue, ``specgals_home`` for the two astroML catalogue files, and
        ``embeddings`` for the train/val/test Parquet directories.

    Returns
    -------
    list[pathlib.Path]
        Catalogue paths followed by embedding shards sorted by filename
        within each split. Their contents are subsequently checksummed to
        identify the inputs used for preparation or fitting.

    Raises
    ------
    FileNotFoundError
        If any split contains no embedding Parquet files. Catalogue existence
        is checked later when the files are read or checksummed.
    KeyError
        If a required configuration key is missing.

    Notes
    -----
    Both the original-latent and full-spectrum MLPs use this catalogue and
    embedding sample to obtain matching spectrum IDs, targets and splits.
    """
    files = [paths['galspec'], *(paths['specgals_home']/f'SDSSspecgalsDR8_{i}.fit.gz' for i in (1,2))]
    for split in SPLITS:
        shards = sorted((paths['embeddings']/split).glob('*.parquet'))
        if not shards:
            raise FileNotFoundError(paths['embeddings']/split)
        files.extend(shards)
    return files


def benchmark_frames(paths, seed, *, full_train=False):
    """Load the cleaned sample, optionally retaining the full training split.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Resolved configuration paths for ``embeddings``, ``galspec`` and
        ``specgals_home``, passed to the shared sample loader.
    seed : int
        Random seed for the training-row draw. Using the same seed and the
        same ordered, cleaned sample reproduces the SR training selection.

    full_train : bool, optional
        Retain all cleaned training rows for preprocessing and cache creation.
        The fitting caller selects its 10,000 model-training rows separately.

    Returns
    -------
    dict[str, pandas.DataFrame]
        DataFrames keyed by ``train``, ``val`` and ``test``, containing matched
        catalogue columns, spectrum IDs and latent representations. Training
        contains all cleaned training rows when full_train=True, otherwise
        10,000 distinct rows in sampling order; validation and test
        retain all eligible rows in their existing order.

    Raises
    ------
    ValueError
        If fewer than 10,000 eligible training rows remain, the seed is
        invalid, or the shared loading/cleaning routines reject the data.
    FileNotFoundError
        If required catalogue files or embedding shards are missing.
    KeyError
        If required configuration keys or data columns are missing.

    Notes
    -----
    The shared loader joins embeddings and catalogues by spectrum ID and
    applies the paper's sample cuts. Target cleaning removes missing and
    sentinel total stellar masses and validates the retained data. Sampling
    uses ``default_rng(seed).choice(..., replace=False)``, as in SR.
    Existing split membership is preserved; no new validation/test split is
    created. Both MLP input representations use these same selected rows.
    """
    frames = clean_target(load_sample(paths['embeddings'], paths['galspec'], paths['specgals_home']))
    if not full_train:
        indices = np.random.default_rng(seed).choice(len(frames['train']), 10000, replace=False)
        frames['train'] = frames['train'].iloc[indices].copy()
    return frames


def spectral_dataset_sources(data_root):
    """List the files in the existing SDSS spectral dataset.

    Parameters
    ----------
    data_root : pathlib.Path
        Root containing ``sdss/sdss_II_catalog``. That directory must contain
        ``dataset_dict.json`` and a ``state.json`` file in each of its
        ``train``, ``val`` and ``test`` subdirectories.

    Returns
    -------
    list[pathlib.Path]
        Sorted paths to all files recursively beneath the dataset directory,
        including metadata and spectral data. These files are subsequently
        checksummed to identify the dataset used for cache preparation.

    Raises
    ------
    FileNotFoundError
        If the overall dataset metadata or any required split metadata file
        is missing.

    Notes
    -----
    This checks the presence of the expected metadata files, not their
    contents or the readability of every spectral data file. It does not
    load spectra, download data or create new splits; preparation uses the
    existing archived split membership.
    """
    dataset = data_root / 'sdss/sdss_II_catalog'
    for required in [dataset/'dataset_dict.json', *(dataset/s/'state.json' for s in SPLITS)]:
        if not required.is_file():
            raise FileNotFoundError(f'Existing SDSS split dataset required: {required}')
    return sorted(p for p in dataset.rglob('*') if p.is_file())


def write_spectral_split(dataset, frame, out, split):
    """Prepare catalogue-matched spectra in order using the upstream dataset.

    Parameters
    ----------
    dataset : upstream SDSS dataset
        Existing split with ``return_id=True``. Its item accessor performs
        fixed-grid placement and Spender normalisation on observed spectra.
    frame : pandas.DataFrame
        Desired rows with ``specObjID`` and ``lgm_tot_p50`` columns.
    out : pathlib.Path
        Existing cache directory, owned by the preparation completion guard.
    split : str
        Split name used for output filenames.

    Returns
    -------
    dict
        Row count and SHA-256 of ordered spectrum IDs.

    Raises
    ------
    ValueError
        If IDs are duplicated, a requested spectrum is absent, the returned
        identity differs, or a spectrum has the wrong shape/nonfinite flux.
    """
    ids = spectrum_ids(pd.Series(dataset.dataset['object_id']), f'{split}.object_id').to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f'{split}: duplicate IDs in the SDSS dataset')
    lookup = {int(identity): i for i, identity in enumerate(ids)}
    wanted = frame.specObjID.to_numpy(dtype='<i8')
    if len(np.unique(wanted)) != len(wanted):
        raise ValueError(f'{split}: duplicate requested IDs')
    missing = set(wanted) - lookup.keys()
    if missing:
        raise ValueError(f'{split}: {len(missing)} requested spectra are absent from the existing split')
    if len(dataset._wave_obs) != 3921:
        raise ValueError('Upstream wavelength grid must contain 3921 pixels')
    values = np.lib.format.open_memmap(out/f'{split}_spectra.npy', mode='w+',
                                      dtype=np.float32, shape=(len(frame),3921)) # create numpy array for spectra
    for row, identity in enumerate(wanted):
        item = dataset[lookup[int(identity)]]
        returned_id = spectrum_ids(pd.Series([item[2]]), f'{split}.returned_id').iloc[0]
        flux = item[0].numpy()
        if returned_id != identity or flux.shape != (3921,) or not np.isfinite(flux).all():
            raise ValueError(f'{split}: invalid prepared spectrum at row {row}')
        values[row] = flux
        if (row+1) % 5000 == 0:
            print(f'prepare {split}: {row+1}/{len(frame)}', flush=True) # print progress
    values.flush()
    del values
    frame[['specObjID','lgm_tot_p50']].to_csv(out/f'{split}_rows.csv', index=False)
    return dict(n=len(frame), ordered_ids_sha256=hashlib.sha256(wanted.tobytes()).hexdigest())


def prepare_cache(paths, data_root, out, *, seed=42, threads=4, restart=False, raw_flux=False):
    """Prepare and record matched observed spectra for all benchmark splits.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Resolved catalogue and embedding configuration used to construct the
        benchmark sample. Includes ``data_root``, ``embeddings``, ``galspec``
        and ``specgals_home``.
    data_root : pathlib.Path
        Root containing the existing ``sdss/sdss_II_catalog`` spectral
        dataset. This may differ from the catalogue/embedding data root.
    out : pathlib.Path
        Destination cache directory, separate from the input directories.
    seed : int, optional
        Retained for the common CLI; cache preparation uses all eligible
        training rows, independently of the later seeded fitting draw.
    threads : int, optional
        Number of PyTorch CPU threads used during preparation; default is 4.
    raw_flux : bool, optional
        Preserve observed flux amplitude instead of dividing each spectrum by
        its rest-frame median. Default is False. Neither cache applies asinh.
    restart : bool, optional
        Preserve an incomplete attempt and start preparation again when True.
        This does not resume partially written arrays. Default is False.

    Returns
    -------
    pathlib.Path
        Absolute cache directory, either newly prepared or reused after its
        saved configuration and required output checksums have been verified.

    Raises
    ------
    FileNotFoundError
        If required source, catalogue, embedding or dataset files are missing.
    FileExistsError
        If an incomplete cache exists and ``restart`` is False.
    ValueError
        If output overlaps an input directory, completed-cache inputs/settings
        differ, a recorded output has changed, or sample/spectrum validation
        fails. Inconsistent wavelength grids across splits are also rejected.
    RuntimeError
        If another process holds the destination's preparation lock.

    Notes
    -----
    Preserves the archived dataset splits and matches spectra by ID in the
    benchmark's requested row order. The shared SDSS loader supplies normalised
    observed-frame fluxes.

    Writes ``{split}_spectra.npy`` and ``{split}_rows.csv`` for each split,
    plus ``wave_observed.npy``. The single
    ``search_complete.json`` record contains input fingerprints,
    preprocessing details, ordered sample IDs' checksums and required
    output checksums.
    """
    import torch
    from wwdc_repro.shared import sdss

    data_root, out = data_root.resolve(), out.resolve()
    for source in (data_root, paths['data_root'], paths['embeddings']):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError(f'Cache output must be separate from input: {source}')
    raw_files = spectral_dataset_sources(data_root) # collects filepaths from SDSS spectra dataset
    source_path = Path(sdss.__file__)
    config = dict(training_rows='full_cleaned_split', raw_flux=raw_flux, inputs=input_hashes([*sample_sources(paths), *raw_files, source_path]),
                  source_sha256=sha256(Path(__file__)))
    products = [f'{s}_{suffix}' for s in SPLITS for suffix in ('spectra.npy','rows.csv')]
    products += ['wave_observed.npy']
    metadata = {}
    with search_run(out, config, products, restart_incomplete=restart,
                    completion_metadata=metadata) as required:
        if not required:
            return out
        torch.set_num_threads(threads)
        frames = benchmark_frames(paths, seed, full_train=True) # cache the full training split for preprocessing.
        samples, wave = {}, None
        for split in SPLITS:
            # load each data split for the spectra.
            dataset = sdss.SDSS(data_root=data_root, split=split, y_catalog={'variables':{'Z':{}},'drop_variables':[]}, return_id=True, normalise=not raw_flux)
            current_wave = dataset._wave_obs.numpy()
            if wave is not None and not np.array_equal(wave, current_wave):
                raise ValueError('Inconsistent wavelength grids across splits')
            wave = current_wave
            samples[split] = write_spectral_split(dataset, frames[split], out, split) # write numpy arrays
        np.save(out/'wave_observed.npy', wave)
        normalisation = ('raw observed flux; no per-spectrum division' if raw_flux else
                         'normalised by the median usable flux in rest-frame 5300--5850 Angstrom')
        metadata.update(samples=samples, preprocessing=
            f'Shared SDSS loader: observed-frame fixed grid, {normalisation}; '
            'no asinh or pixel standardisation. Upstream masking and nonfinite-value '
            'handling are preserved. Only flux is exported, not inverse-variance weights.')
    return out


def metrics(truth, prediction):
    """Evaluate log-mass predictions with scikit-learn metrics.

    Parameters
    ----------
    truth, prediction : array-like of shape (n_samples,)
        Catalogue and predicted log10 stellar masses in the same row order.
        Values are converted to float64; no observations or outliers are dropped.

    Returns
    -------
    dict
        Sample count ``n``, coefficient of determination ``r2``, root mean
        squared error ``rmse_dex`` and mean absolute error ``mae_dex``.
        Both errors are measured in dex; R2 is dimensionless.

    Raises
    ------
    ValueError
        If shapes differ, values are nonfinite, fewer than two samples are
        supplied, or the catalogue targets are constant (undefined R2).
    """
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite or mismatched predictions/targets')
    if len(truth) < 2 or np.all(truth == truth[0]):
        raise ValueError('Constant target cannot define benchmark R2')
    return dict(n=len(truth), r2=float(r2_score(truth, prediction)),
                rmse_dex=float(root_mean_squared_error(truth, prediction)),
                mae_dex=float(mean_absolute_error(truth, prediction)))


def read_spectra(directory, frames, *, raw_flux=False):
    """Load cached spectra and verify their alignment with the benchmark sample.

    Parameters
    ----------
    directory : pathlib.Path
        Directory containing ``{split}_spectra.npy`` and ``{split}_rows.csv``
        for train, val and test. Arrays must contain 3,921-pixel
        observed-frame spectra, before asinh transformation or pixel-wise
        standardisation. CSV files must contain ``specObjID`` and
        ``lgm_tot_p50`` columns in the same order as the array rows.
    frames : dict[str, pandas.DataFrame]
        Expected catalogue sample for each split. Fitting supplies the full
        cleaned training split for preprocessing, plus the full eligible
        validation and test samples. Each frame supplies spectrum IDs and catalogue masses.

    Returns
    -------
    dict[str, numpy.memmap]
        Read-only spectral arrays keyed by split, each of shape
        ``(len(frames[split]), 3921)``. Flux values are not transformed.

    Raises
    ------
    ValueError
        If IDs or row ordering differ (including old 10k-only caches),
        masses disagree by more than 1e-6 dex,
        array shapes are incorrect, or spectral values are nonfinite.
    FileNotFoundError
        If a required cache file is missing.
    KeyError
        If an expected split or CSV/catalogue column is missing.

    Notes
    -----
    raw_flux=True requires a completion record identifying an unnormalised
    flux cache. Otherwise Spender normalisation is expected. This function checks
    sample alignment and numerical validity, but does not reconstruct spectra
    from raw SDSS data. For caches created by ``prepare``, it also verifies all
    recorded output hashes. The loader source fingerprint belongs to the
    preparation configuration and is not rechecked here. Legacy caches
    without a receipt receive only the alignment and numerical checks below.
    """
    # Legacy caches can still be supplied explicitly. Caches produced by the
    # preparation command must pass their recorded output checks.
    if (directory/'search_complete.json').exists():
        completed = json.loads((directory/'search_complete.json').read_text())
        if completed['configuration'].get('raw_flux', False) != raw_flux: # just make sure that the right spectra (raw/normalised flux is used)
            raise ValueError('Spectral cache normalisation does not match the requested input')
        for name, expected in completed['outputs'].items():
            if not (directory/name).is_file() or sha256(directory/name) != expected:
                raise ValueError(f'Prepared spectral cache missing/changed: {directory/name}')
    elif raw_flux:
        raise ValueError('Raw flux requires a prepared cache with a completion record')
    arrays = {}
    for split in SPLITS:
        rows = pd.read_csv(directory / f'{split}_rows.csv') # get the ids and stellar masses
        frame = frames[split]
        if not np.array_equal(rows.specObjID.to_numpy(), frame.specObjID.to_numpy()):
            raise ValueError(f'{split}: spectra IDs/order differ from the SR sample; '
                             'rebuild old 10k-only caches to include the full cleaned training split')
        if not np.allclose(rows.lgm_tot_p50, frame.lgm_tot_p50, rtol=0, atol=1e-6):
            raise ValueError(f'{split}: spectra targets differ from the catalogue')
        values = np.load(directory / f'{split}_spectra.npy', mmap_mode='r', allow_pickle=False)
        if values.shape != (len(frame), 3921) or not np.isfinite(values).all():
            raise ValueError(f'{split}: expected finite (n, 3921) spectra')
        arrays[split] = values
    return arrays


def fit_arm(arrays, targets, frames, out, *, arm, seed, threads=2, max_epochs=300,
            patience=25, widths=WIDTHS, rates=RATES, train_indices=None):
    """Fit and select an MLP for one input representation, then export results.

    Parameters
    ----------
    arrays : dict[str, array-like]
        Feature matrices keyed by 'train', 'val' and 'test', with shape
        (n_samples, n_features). Supply original latents or normalised spectra
        before asinh transformation and standardisation. Columns must have
        the same meaning across splits.
    targets : dict[str, array-like]
        One-dimensional catalogue log10 stellar masses for each split,
        aligned row-for-row with the corresponding feature matrix.
    frames : dict[str, pandas.DataFrame]
        Catalogue frames with a specObjID column in the same row order as
        arrays and targets. Alignment must be checked before calling.
    out : pathlib.Path
        New output directory for this representation; must not already exist.
    arm : {'orig', 'spectra', 'spectra_normalised', 'spectra_raw'}
        Representation label. 'spectra' enables training-scaled asinh before
        standardisation. 'spectra_normalised' and 'spectra_raw' omit asinh;
        'orig' also uses standardisation alone.
    seed : int
        Initialisation and shuffling seed, reused for every trial. Training
        sample selection is performed by the caller, not this function.
    train_indices : array-like of int or None, optional
        Training rows used for model fitting and training-score reporting.
        Preprocessing uses all rows of arrays['train'] and targets['train'].
        The benchmark caller supplies the same seeded 10,000 indices as SR.
    threads : int, optional
        Numerical thread limit for fitting and prediction; default is 2.
    max_epochs : int, optional
        Maximum training epochs per trial; default is 300.
    patience : int, optional
        Stop a trial after this many epochs without improved validation MSE;
        default is 25.
    widths : sequence of pairs of int, optional
        Candidate sizes for the two hidden layers. Defaults to (128, 64)
        and (1024, 512).
    rates : sequence of float, optional
        Candidate initial learning rates, crossed with every hidden-layer
        pair. Defaults to 0.001 and 0.0001. Both candidate sequences must
        be nonempty.

    Returns
    -------
    dict
        Representation, seed, selected trial settings, selection criterion,
        parameter count, and train/validation/test metrics (n, R2, RMSE and
        MAE). Error metrics are expressed in dex.

    Notes
    -----
    Delegates training and prediction to shared.mlp. Each trial uses Adam,
    ReLU hidden layers, batch size 256 and L2 regularisation alpha=1e-4.
    Input and target scaling are fitted on training rows only. The existing
    validation split selects both the epoch and the trial by minimum MSE;
    exact ties retain the first trial. Test data do not influence selection.

    Writes each trial's epoch history and trials.csv, then saves the selected
    model and its preprocessing in model.joblib before evaluating the test
    sample. After selection, every trial checkpoint is evaluated on all three
    splits for reporting in trial_metrics.csv. Only the selected model writes
    {split}_predictions.csv. Metrics are returned for the combined results
    file. Failed runs may leave partial
    outputs; completion handling belongs to the caller.
    """
    out.mkdir(parents=True)
    trials, best_loss, chosen, checkpoint = [], np.inf, None, None
    for hidden in widths:
        for rate in rates:
            name = f'{hidden[0]}-{hidden[1]}-lr{rate:g}'
            candidate, history = shared_mlp.fit_mlp(
                arrays['train'], targets['train'], arrays['val'], targets['val'],
                seed=seed, max_epochs=max_epochs, patience=patience, threads=threads,
                train_indices=train_indices, asinh=arm == 'spectra', model_parameters=dict(
                    hidden_layer_sizes=hidden, learning_rate_init=rate,
                    batch_size=256, alpha=1e-4))
            candidate['arm'] = arm
            joblib.dump(candidate, out / f'{name}.joblib')
            loss = float(history.val_mse_dex.min())
            history.to_csv(out / f'{name}_history.csv', index=False)
            row = dict(trial=name, hidden_1=hidden[0], hidden_2=hidden[1], learning_rate=rate,
                       best_epoch=candidate['best_epoch'], epochs_run=candidate['epochs_run'],
                       validation_mse_dex=loss)
            trials.append(row)
            print(f'{arm}: {name}, epoch {row["best_epoch"]}, val RMSE={loss**.5:.5f}', flush=True)
            if loss < best_loss:
                best_loss, chosen, checkpoint = loss, dict(row), candidate
    pd.DataFrame(trials).to_csv(out / 'trials.csv', index=False)
    checkpoint['arm'] = arm
    joblib.dump(checkpoint, out / 'model.joblib')
    model = checkpoint['model']['mlp']
    result = dict(arm=arm, seed=seed, selected=chosen, selection='validation',
                  n_parameters=sum(v.size for v in [*model.coefs_, *model.intercepts_]), metrics={})
    # Report training scores on the fitting subset, not the preprocessing pool.
    if train_indices is not None:
        arrays = dict(arrays, train=arrays['train'][train_indices])
        targets = dict(targets, train=targets['train'][train_indices])
        frames = dict(frames, train=frames['train'].iloc[train_indices])
    trial_metrics = []
    for trial in trials:
        candidate = joblib.load(out / f'{trial["trial"]}.joblib')
        selected = trial['trial'] == chosen['trial']
        for split in SPLITS:
            pred = shared_mlp.predict_mlp(candidate, arrays[split], threads=threads)
            scores = metrics(targets[split], pred)
            trial_metrics.append(dict(arm=arm, seed=seed, **trial, selected=selected,
                                      split=split, **scores))
            if selected:
                result['metrics'][split] = scores
                pd.DataFrame(dict(specObjID=frames[split].specObjID.to_numpy(), truth=targets[split],
                                  prediction=pred)).to_csv(out / f'{split}_predictions.csv', index=False)
    pd.DataFrame(trial_metrics).to_csv(out / 'trial_metrics.csv', index=False)
    return result


def fit(args):
    """Coordinate the matched MLP benchmark using command-line arguments.

    Load configuration, prepare spectra if requested, validate inputs and
    reuse a verified completed run when available. Otherwise, build the
    matched samples, call fit_arm for each representation, and save combined
    results and completion records. Table export is handled separately.
    """
    from wwdc_repro.workflows.paper_stages import load_config
    paths = load_config(args.config.resolve())
    spectra = args.spectra_dir or paths.get('stellar_mlp_spectra')
    raw_root = args.spectra_data_root or paths.get('stellar_mlp_data_root')
    raw_flux = args.spectrum_mode == 'raw'
    if spectra is None and raw_root is not None:
        spectra = prepare_cache(paths, raw_root, args.outdir.resolve().parent/'stellar_mass_spectra',
                                seed=args.seed, threads=args.threads, restart=args.restart_incomplete, raw_flux=raw_flux)
    spectrum_arm = {'normalised-asinh':'spectra', 'normalised':'spectra_normalised', 'raw':'spectra_raw'}[args.spectrum_mode]
    if spectra is None and args.spectrum_mode != 'normalised-asinh':
        raise ValueError('A spectrum mode requires --spectra-dir or --spectra-data-root')
    arms = ['orig', spectrum_arm] if spectra is not None else ['orig']
    spectra = spectra.resolve() if spectra is not None else None
    sources = [*sample_sources(paths), Path(shared_mlp.__file__)]
    if spectra:
        sources.extend(spectra/f'{split}_{suffix}' for split in SPLITS for suffix in ('spectra.npy','rows.csv'))
        if (spectra/'search_complete.json').exists():
            sources.extend(spectra/name for name in ('search_complete.json','wave_observed.npy'))
    # Reject accidental writes over input directories, including parent directories.
    out = args.outdir.resolve()
    for source in [paths['embeddings'], paths['galspec'].parent, *([spectra] if spectra else [])]:
        if out == source or out in source.parents or source in out.parents:
            raise ValueError(f'Output must be separate from input: {source}')
    config = dict(inputs=input_hashes(sources), seed=args.seed, n_train=10000, arms=arms, spectrum_mode=args.spectrum_mode,
                  preprocessing_training_rows='full_cleaned_split',
                  widths=WIDTHS, rates=RATES, max_epochs=300, patience=25, batch_size=256,
                  implementation='sklearn.neural_network.MLPRegressor.partial_fit',
                  optimizer='adam', alpha=1e-4, device='cpu', threads=args.threads,
                  sklearn_version=sklearn.__version__, numpy_version=np.__version__,
                  source_sha256=sha256(Path(__file__)))
    products = ['results.json', *[f'{split}_rows.csv' for split in SPLITS]]
    for arm in arms:
        products.extend(f'{arm}/{name}' for name in (
            'model.joblib', 'trials.csv', 'trial_metrics.csv', *[f'{split}_predictions.csv' for split in SPLITS]))
        products.extend(f'{arm}/{h1}-{h2}-lr{rate:g}_history.csv'
                        for h1, h2 in WIDTHS for rate in RATES)
        products.extend(f'{arm}/{h1}-{h2}-lr{rate:g}.joblib'
                        for h1, h2 in WIDTHS for rate in RATES)
    with search_run(out, config, products, restart_incomplete=args.restart_incomplete) as required:
        if required:
            frames = benchmark_frames(paths, args.seed, full_train=True)
            train_indices = np.random.default_rng(args.seed).choice(len(frames['train']), 10000, replace=False)
            targets = {s:frames[s].lgm_tot_p50.to_numpy(dtype=np.float32) for s in SPLITS}
            sample = {}
            for split in SPLITS:
                frame = frames[split].iloc[train_indices] if split == 'train' else frames[split]
                frame[['specObjID','lgm_tot_p50']].to_csv(out/f'{split}_rows.csv', index=False)
                sample[split] = dict(n=len(frame), ordered_ids_sha256=hashlib.sha256(
                    frame.specObjID.to_numpy(dtype='<i8').tobytes()).hexdigest())
            sample['preprocessing_train'] = dict(n=len(frames['train']), ordered_ids_sha256=hashlib.sha256(
                frames['train'].specObjID.to_numpy(dtype='<i8').tobytes()).hexdigest())
            # Validate every optional spectrum before fitting anything.
            spectral_arrays = read_spectra(spectra, frames, raw_flux=raw_flux) if spectra else None
            results = []
            for arm in arms:
                arrays = ({s:latent_array(frames[s], 'orig', s) for s in SPLITS}
                          if arm=='orig' else spectral_arrays)
                results.append(fit_arm(arrays, targets, frames, out/arm, arm=arm, seed=args.seed,
                                       threads=args.threads, train_indices=train_indices))
            report = dict(configuration=config, samples=sample, results=results)
            (out/'results.json').write_text(json.dumps(report, indent=2) + '\n')
    verified_results(out)
    print(f'Completed benchmark: {out}', flush=True)


def verified_results(run):
    """Refuse incomplete or tampered benchmarks, including changed predictions."""
    completion = json.loads((run/'search_complete.json').read_text())
    outputs = completion['outputs']
    if 'results.json' not in outputs:
        raise ValueError('Completion record must list results.json directly')
    for name, expected in outputs.items():
        path = run/name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Benchmark output missing/changed: {path}')
    return json.loads((run/'results.json').read_text())


def export_results(run, outdir):
    """Export selected-model CSV, LaTeX table and numerical macros; never fit."""
    report = verified_results(run)
    rows = []
    for result in report['results']:
        row = dict(arm=result['arm'], seed=result['seed'], **result['selected'],
                   n_train=result['metrics']['train']['n'], n_test=result['metrics']['test']['n'],
                   validation_r2=result['metrics']['val']['r2'],
                   test_r2=result['metrics']['test']['r2'],
                   rmse_dex=result['metrics']['test']['rmse_dex'], mae_dex=result['metrics']['test']['mae_dex'])
        rows.append(row)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/'stellar_mass_mlp.csv', index=False)
    lines, macros = [], []
    names = {'orig':('Original latents','Original'), 'spectra':('Normalised spectrum + asinh','Spectrum'),
             'spectra_normalised':('Normalised spectrum','NormalisedSpectrum'),
             'spectra_raw':('Raw-flux spectrum','RawSpectrum')}
    for row in rows:
        title, macro = names[row['arm']]
        lines.append(f"{title} & {row['validation_r2']:.4f} & {row['test_r2']:.4f} & {row['rmse_dex']:.4f} & {row['mae_dex']:.4f}"+r'\\')
        macros.append(r'\newcommand{\StellarMLP'+macro+r'Rtwo}{'+f"{row['test_r2']:.4f}"+'}')
    tex = (r'\begin{table}[htbp]'+'\n'+r'\centering\small'+'\n'+r'\begin{tabular}{lrrrr}'+'\n'+r'\toprule'+'\n'
        +r'Input & Validation $R^2$ & Test $R^2$ & RMSE (dex) & MAE (dex)\\'+'\n'+r'\midrule'+'\n'
        +'\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
        +r'\caption{Total stellar-mass MLP benchmark on the same 10,000 training spectra as the seed-'
        +str(report['configuration']['seed'])+r' symbolic searches. Checkpoints and configurations are selected by validation loss. '
        +r'No explicit redshift or normalisation factor is provided.}'+'\n'
        +r'\label{tab:stellar_mass_mlp}'+'\n'+r'\end{table}'+'\n')
    (outdir/'stellar_mass_mlp.tex').write_text(tex)
    (outdir/'stellar_mass_mlp_numbers.tex').write_text('\n'.join(macros)+'\n')
    (outdir/'stellar_mass_mlp_provenance.json').write_text(json.dumps(dict(
        run=str(run.resolve()), results_sha256=sha256(run/'results.json'),
        configuration=report['configuration'], samples=report['samples']), indent=2)+'\n')
    print(pd.DataFrame(rows)[['arm','validation_r2','test_r2','rmse_dex']].to_string(index=False), flush=True)


def export_grid(runs, outdir):
    """Export all trial/split scores from verified runs, deduplicating shared arms.

    Each run must include trial_metrics.csv. Repeated representations (such
    as the latent control) must have identical metrics. Selection flags were
    fixed using validation MSE before any test scores were computed.
    """
    arms = {}
    for run in runs:
        report = verified_results(run)
        for result in report['results']:
            arm = result['arm']
            frame = pd.read_csv(run/arm/'trial_metrics.csv')
            if arm in arms:
                pd.testing.assert_frame_equal(arms[arm], frame)
            else:
                arms[arm] = frame
    combined = pd.concat(arms.values(), ignore_index=True)
    outdir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outdir/'stellar_mass_mlp_grid.csv', index=False)
    macro_names = {'orig':'Original', 'spectra':'Spectrum',
                   'spectra_normalised':'NormalisedSpectrum', 'spectra_raw':'RawSpectrum'}
    macros = []
    for arm, frame in arms.items():
        selected = frame[(frame['split'] == 'test') & frame.selected]
        if len(selected) != 1:
            raise ValueError(f'{arm}: expected one validation-selected test result')
        macros.append(r'\newcommand{\StellarMLP'+macro_names[arm]+r'Rtwo}{'+
                      f'{selected.iloc[0].r2:.4f}'+'}')
    (outdir/'stellar_mass_mlp_numbers.tex').write_text('\n'.join(macros)+'\n')
    names = {'orig':'Original latents', 'spectra':'Normalised spectrum + asinh',
             'spectra_normalised':'Normalised spectrum', 'spectra_raw':'Raw-flux spectrum'}
    lines = [r'\begin{table}[htbp]', r'\centering\normalsize',
             r'\renewcommand{\arraystretch}{1.15}',
             r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}', r'\toprule',
             r'Hidden layers & Learning rate & Test $R^2$ & RMSE (dex) & MAE (dex)\\',r'\midrule']
    display_order = ('orig', 'spectra_normalised', 'spectra', 'spectra_raw')
    for group_index, arm in enumerate(arm for arm in display_order if arm in arms):
        frame = arms[arm]
        if group_index:
            lines.extend([r'\addlinespace[0.3em]', r'\midrule'])
        lines.append(r'\multicolumn{5}{@{}l}{\textbf{'+names[arm]+r'}}\\')
        lines.append(r'\addlinespace[0.2em]')
        test = frame[frame['split'] == 'test']
        previous_widths = None
        for _, row in test.iterrows():
            widths = (int(row.hidden_1), int(row.hidden_2))
            if previous_widths is not None and widths != previous_widths:
                lines.append(r'\addlinespace[0.3em]')
            previous_widths = widths
            r2 = f'{row.r2:.4f}'
            mantissa, exponent = f'{row.learning_rate:.6e}'.split('e')
            coefficient = '' if float(mantissa) == 1 else f'{float(mantissa):g}'+r'\times '
            rate = '$'+coefficient+f'10^{{{int(exponent)}}}'+'$'
            lines.append(f'$({widths[0]},\\,{widths[1]})$ & {rate} & '
                         f'{r2} & {row.rmse_dex:.4f} & {row.mae_dex:.4f}'+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular*}',
        r'\caption{Test performance for each of the trained MLPs, using the matched seed-42 sample of 10,000 spectra and evaluated on the test spectra. Each model uses its minimum-validation-MSE checkpoint.}',
        r'\label{tab:stellar_mass_mlp}', r'\end{table}'])
    (outdir/'stellar_mass_mlp_grid.tex').write_text('\n'.join(lines)+'\n')
    print(f'Exported {len(combined)} trial/split rows to {outdir}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fitting = sub.add_parser('fit')
    fitting.add_argument('--config', type=Path, default=Path('reproduction.json'))
    fitting.add_argument('--outdir', type=Path, required=True)
    fitting.add_argument('--spectra-dir', type=Path)
    fitting.add_argument('--spectra-data-root', type=Path, help='Root containing sdss/sdss_II_catalog; prepare spectra if no cache is supplied')
    fitting.add_argument('--spectrum-mode', choices=('normalised-asinh','normalised','raw'),
                         default='normalised-asinh', help='Flux preprocessing; all modes use training-only pixel standardisation')
    fitting.add_argument('--seed', type=int, default=42)
    fitting.add_argument('--threads', type=positive_int, default=4)
    fitting.add_argument('--restart-incomplete', action='store_true')
    preparing = sub.add_parser('prepare', help='Create the matched spectral cache without fitting')
    preparing.add_argument('--config', type=Path, default=Path('reproduction.json'))
    preparing.add_argument('--data-root', type=Path, help='Root containing existing SDSS split dataset')
    preparing.add_argument('--outdir', type=Path, required=True)
    preparing.add_argument('--raw-flux', action='store_true', help='Preserve observed flux amplitude in the cache')
    preparing.add_argument('--seed', type=int, default=42)
    preparing.add_argument('--threads', type=positive_int, default=4)
    preparing.add_argument('--restart-incomplete', action='store_true')
    exporting = sub.add_parser('export')
    exporting.add_argument('--run', type=Path, required=True)
    exporting.add_argument('--outdir', type=Path, required=True)
    grid = sub.add_parser('export-grid', help='Export every trial from one or more completed runs')
    grid.add_argument('--runs', type=Path, nargs='+', required=True)
    grid.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'fit':
        fit(args)
    elif args.command == 'prepare':
        from wwdc_repro.workflows.paper_stages import load_config
        paths = load_config(args.config.resolve())
        raw_root = args.data_root or paths.get('stellar_mlp_data_root', paths['data_root'])
        prepare_cache(paths, raw_root, args.outdir,
                      seed=args.seed, threads=args.threads, restart=args.restart_incomplete, raw_flux=args.raw_flux)
    elif args.command == 'export-grid':
        export_grid([run.resolve() for run in args.runs], args.outdir.resolve())
    else:
        export_results(args.run.resolve(), args.outdir.resolve())


if __name__ == '__main__':
    main()
