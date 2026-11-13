"""Three paper commands: fit equations/models, render plots, and export tables.

Uses existing modules. All downstream products come from the requested run;
this workflow never imports archived equations or reference model scores.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from wwdc_repro.shared.search_runs import search_run, input_hashes, positive_int
from wwdc_repro.shared.utils import sha256

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 43, 44)
ARMS = ('z', 'orig', 'uncond', 'cond', 'cond+z', 'orig+z')


def load_config(path):
    """Resolve explicit inputs; default embedding locations follow the released data layout."""
    raw = json.loads(path.read_text())
    allowed = {'data_root', 'embeddings', 'norm_embeddings', 'joint_embeddings',
               'galspec', 'specgals_home', 'cmd_cache', 'norm_generation'}
    if not isinstance(raw, dict) or 'data_root' not in raw or set(raw) - allowed:
        raise ValueError(f'Configuration needs data_root; allowed keys: {sorted(allowed)}')
    paths = {}
    for name, value in raw.items():
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty path')
            p = Path(value).expanduser()
            paths[name] = (path.parent / p).resolve()
    base = paths['data_root']
    defaults = dict(embeddings=base / 'sdss_II/spender_I_flow_v2/spender_I_flow_v2/embeddings/7655991_0',
                    norm_embeddings=base / 'sdss_II/spender_I_normalisation_flow/embeddings/norm_20261012183217_0',
                    joint_embeddings=base / 'sdss_II/spender_I_z_normalisation_flow/embeddings/znorm_20261013104946_0',
                    galspec=base / 'galSpecExtra-dr8.fits', specgals_home=base / 'sdss')
    for key, value in defaults.items():
        paths.setdefault(key, value)
    # Preparation modules currently resolve the catalogue pair through DATA_ROOT.
    if paths['galspec'] != defaults['galspec'] or paths['specgals_home'] != defaults['specgals_home']:
        raise ValueError('For fresh fitting, place/link catalogues at data_root/galSpecExtra-dr8.fits and data_root/sdss')
    return paths


def source_files(paths, stage):
    """Check the stage's external prerequisites before running commands."""
    files = [paths['galspec'], *(paths['specgals_home'] / f'SDSSspecgalsDR8_{i}.fit.gz' for i in (1, 2))]
    for name in ('embeddings', 'norm_embeddings', 'joint_embeddings'):
        for split in ('train', 'val', 'test'):
            shards = sorted((paths[name] / split).glob('*.parquet'))
            if not shards:
                raise FileNotFoundError(f'No embeddings: {paths[name] / split}')
            files.extend(shards)
    norm = paths['norm_embeddings']
    files.append(norm.parent.parent / 'ckpts' / f'{norm.name}.yaml')
    if stage == 'plots':
        if not {'cmd_cache', 'norm_generation'} <= paths.keys():
            raise ValueError('plots requires cmd_cache and norm_generation to produce all manuscript figures')
        files.append(paths['cmd_cache'])
        files.extend(paths['norm_generation'] / name for name in ('generated_spectra.npz', 'observed_spectra.npz'))
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    return files


def build_steps(stage, paths, out, *, selection='train', threads=8, restart=False):
    """Return explicit module commands and expected products in dependency order."""
    work = out / 'work/conditioning'
    python = ['uv', 'run', '--frozen', 'python', '-m']
    suite = ['uv', 'run', '--script', '--locked', str(ROOT / 'experiments/conditioning/suite.py'),
             'run', '--workdir', str(work)]
    common = ['--embeddings', str(paths['embeddings']), '--galspec', str(paths['galspec']),
              '--specgals-home', str(paths['specgals_home'])]
    selection_args = ['--selection', selection]
    fit_args = ['--threads', str(threads)] + (['--restart-incomplete'] if restart else [])
    steps = []

    def add(name, module, args=(), products=(), owned=(), search=False, pinned=False):
        command = suite + [module, *args] if pinned else python + [module, *args]
        steps.append(dict(name=name, command=command, cwd=str(ROOT if pinned else work),
                          products=list(products), owned=list(owned), search=search))

    if stage == 'equations':
        for kind, arms, embeddings, prefix in (
            ('redshift', ARMS, paths['embeddings'], ''),
            ('norm', ('cond', 'cond+z'), paths['norm_embeddings'], 'normflow_condnorm_'),
            ('znorm', ('cond+z',), paths['joint_embeddings'], 'znormflow_condnorm_'),
        ):
            for arm in arms:
                for seed in SEEDS:
                    args = ['--conditioning', kind, '--embed-type', arm, '--seed', str(seed),
                            '--embeddings', str(embeddings), '--galspec', str(paths['galspec']),
                            '--specgals-home', str(paths['specgals_home']), *fit_args]
                    if kind != 'redshift':
                        args.append('--append-norm')
                    folder = f'job_results_{prefix}lgm_tot_p50_{arm}_seed{seed}'
                    add(f'fit-{kind}-{arm}-{seed}', 'run_total_mass', args,
                        [f'{folder}/pareto_fronts.csv', f'{folder}/search_complete.json'], search=True, pinned=True)
        add('prepare-norm', 'wwdc_repro.norm.validate_conditioning', ['--prepare-only'],
            ['norm_analysis/*_matrices.npz', 'norm_analysis/*_ids.npy', 'norm_analysis/candidates.csv', 'norm_analysis/manifest.json'], ['norm_analysis'])
        add('score-norm', 'wwdc_repro.norm.validate_conditioning', selection_args,
            ['norm_analysis/candidate_audit_final.csv', 'norm_analysis/validation_selected_final.csv'])
        add('prepare-controls', 'wwdc_repro.norm.matched_conditioning_controls', ['prepare'],
            ['norm_analysis/matched_controls/*.*'], ['norm_analysis/matched_controls'])
        for arm in ('original', 'zflow'):
            for seed in SEEDS:
                add(f'fit-control-{arm}-{seed}', 'matched_conditioning_controls',
                    ['fit', '--arm', arm, '--seed', str(seed), *selection_args, *fit_args],
                    [f'norm_analysis/matched_controls/{arm}_seed{seed}/*.csv'], search=True, pinned=True)
        add('prepare-ablation', 'wwdc_repro.norm.norm_input_ablation', ['prepare'], ['norm_ablation/*.*'], ['norm_ablation'])
        for arm in ('original', 'norm'):
            for seed in SEEDS:
                add(f'fit-ablation-{arm}-{seed}', 'norm_input_ablation',
                    ['fit', '--arm', arm, '--seed', str(seed), *selection_args, *fit_args],
                    [f'norm_ablation/{arm}_seed{seed}/*.csv'], search=True, pinned=True)
        add('prepare-fibre', 'wwdc_repro.fibre_mass.fibre_mass_conditioning', ['prepare'], ['fibre_mass/*.*'], ['fibre_mass'])
        for arm in ('original', 'zflow', 'norm', 'znorm'):
            for seed in SEEDS:
                add(f'fit-fibre-{arm}-{seed}', 'fibre_mass_conditioning',
                    ['fit', '--arm', arm, '--seed', str(seed), *selection_args, *fit_args],
                    [f'fibre_mass/{arm}_seed{seed}/*.csv'], search=True, pinned=True)
        add('score-redshift', 'wwdc_repro.redshift.cli', [*common, '--fronts', str(work), '--outdir', str(out / 'work/redshift/analysis'),
                                                       '--fresh-fronts', *selection_args],
            [str(out / 'work/redshift/analysis/selected.csv')], [str(out / 'work/redshift/analysis')])
        add('score-ablation', 'wwdc_repro.norm.norm_input_ablation', ['compare_results', *selection_args],
            ['norm_ablation/selected_all_budgets.csv', 'norm_ablation/summary.csv'])
        add('score-fibre', 'wwdc_repro.fibre_mass.analyse_fibre', selection_args,
            ['fibre_mass/original_control_comparison/*.csv'])
        add('fit-norm-probes', 'wwdc_repro.norm.norm_conditioning_probe', products=['norm_probe/linear_results.csv', 'norm_probe/coordinate_correlations.csv', 'norm_probe/mlp_sklearn/*.csv'], owned=['norm_probe'])
        add('fit-fibre-mlp', 'wwdc_repro.fibre_mass.fibre_mass_mlp_control', products=['fibre_mass/mlp_sklearn/*.csv'], owned=['fibre_mass/mlp_sklearn'])
    else:
        destination = out / ('figures/redshift' if stage == 'plots' else 'tables/redshift')
        only = ['frontier', 'seed-frontiers', 'latent', 'anatomy', 'zero-latent', 'cmd'] if stage == 'plots' else ['weights', 'front-tables', 'diagnostic-tables']
        add(f'{stage}-redshift', 'wwdc_repro.redshift.figures', [*common, '--selected', str(out / 'work/redshift/analysis/selected.csv'),
            '--fronts', str(work), '--outdir', str(destination), '--only', *only,
            *(['--cmd-cache', str(paths['cmd_cache'])] if stage == 'plots' else [])],
            [str(destination / ('*.pdf' if stage == 'plots' else '*.tex'))], [str(destination)])
        if stage == 'plots':
            add('plot-norm-frontier', 'wwdc_repro.norm.plot_norm_frontier',
                ['--outdir', str(out / 'figures/normalisation')], [str(out / 'figures/normalisation/norm_frontier.pdf')])
            add('plot-norm-cmd', 'wwdc_repro.norm.plot_norm_cmd_full', products=['norm_cmd/*.pdf'])
            add('plot-norm-sequence', 'wwdc_repro.norm.generate_norm_sequence',
                ['--input', str(paths['norm_generation']), '--outdir', str(out / 'figures/normalisation')],
                [str(out / 'figures/normalisation/norm_conditional_sequence.pdf')])
        else:
            add('tables-conditioning', 'wwdc_repro.workflows.paper_tables',
                ['--outdir', str(out / 'tables/conditioning')], [str(out / 'tables/conditioning/*.csv'), str(out / 'tables/conditioning/*.tex')])
    return steps


def products(step, work):
    """Require every expected product pattern and fingerprint its files."""
    files = []
    for pattern in step['products']:
        import glob
        matches = [Path(p) for p in glob.glob(str(work / pattern)) if Path(p).is_file()]
        if not matches:
            raise FileNotFoundError(f'{step["name"]} did not produce {pattern}')
        files.extend(matches)
    return input_hashes(files)


def verify_products(hashes):
    for name, digest in hashes.items():
        if not Path(name).is_file() or sha256(Path(name)) != digest:
            raise ValueError(f'Missing/changed run output: {name}')


def run_steps(steps, out, configuration, env, *, restart=False):
    """Execute sequentially with verified step receipts; retain subprocess logs."""
    work = out / 'work/conditioning'
    work.mkdir(parents=True, exist_ok=True)
    logs = out / 'logs' / configuration['stage']
    logs.mkdir(parents=True, exist_ok=True)
    def execute(step):
        logfile = logs / f'{step["name"]}.log'
        with logfile.open('a') as stream:
            try:
                subprocess.run(step['command'], cwd=step['cwd'], env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'{step["name"]} failed; see {logfile}') from exc

    all_products = {}
    for step in steps:
        print(step['name'], flush=True)
        state = out / '.steps' / step['name']
        # Search modules manage their own per-seed restart/completion records.
        if step['search']:
            execute(step)
            all_products.update(products(step, work))
            continue
        with search_run(state, configuration, ('products.json',), restart_incomplete=restart) as required:
            if required:
                for name in step['owned']:
                    directory = work / name
                    if directory.exists():
                        if not restart:
                            raise FileExistsError(f'{directory} already exists; use --restart-incomplete to preserve and rebuild this step')
                        import uuid
                        backup = out / 'incomplete_attempts'
                        backup.mkdir(exist_ok=True)
                        directory.rename(backup / f'{directory.name}-{uuid.uuid4().hex}')
                execute(step)
                receipt = products(step, work)
                (state / 'products.json').write_text(json.dumps(receipt, indent=2) + '\n')
            receipt = json.loads((state / 'products.json').read_text())
            verify_products(receipt)
            all_products.update(receipt)
    return all_products


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('equations', 'plots', 'tables'))
    parser.add_argument('--config', type=Path, default=Path('reproduction.json'))
    parser.add_argument('--outdir', type=Path, default=Path('outputs/paper'))
    parser.add_argument('--selection', choices=('train', 'validation'), default='train')
    parser.add_argument('--threads', type=positive_int, default=8)
    parser.add_argument('--restart-incomplete', action='store_true')
    parser.add_argument('--check', action='store_true', help='Validate inputs and print commands without executing')
    args = parser.parse_args(argv)
    paths, out = load_config(args.config.resolve()), args.outdir.resolve()
    for name, source in paths.items():
        if out == source or source in out.parents or out in source.parents:
            raise ValueError(f'Output directory must be separate from {name}: {source}')
    sources = source_files(paths, args.stage)
    configuration = dict(stage=args.stage, selection=args.selection, threads=args.threads,
                         inputs=input_hashes(sources))
    if args.stage != 'equations':
        completed = out / 'equations.json'
        if not completed.is_file():
            raise FileNotFoundError(f'Run equations first: {completed}')
        record = json.loads(completed.read_text())
        if record['configuration']['selection'] != args.selection:
            raise ValueError('Selection policy differs from the equation run')
        for name, digest in record['configuration']['inputs'].items():
            if configuration['inputs'].get(name) != digest:
                raise ValueError(f'Inputs differ from equation run: {name}')
        verify_products(record['products'])
        configuration['equations_sha256'] = sha256(completed)
    steps = build_steps(args.stage, paths, out, selection=args.selection, threads=args.threads, restart=args.restart_incomplete)
    if args.check:
        for step in steps:
            print(f"{step['name']}: {shlex.join(step['command'])}")
        return
    env = dict(os.environ, DATA_ROOT=str(paths['data_root']), MPLBACKEND='Agg',
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / 'src'), os.environ.get('PYTHONPATH')])),
               WWDC_Z_EMBEDDINGS=str(paths['embeddings']), WWDC_NORM_EMBEDDINGS=str(paths['norm_embeddings']),
               WWDC_ZNORM_EMBEDDINGS=str(paths['joint_embeddings']),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # uv must resolve this repository's project even when the modules run in work/.
    env['UV_PROJECT'] = str(ROOT)
    if args.stage == 'plots':
        cache = out / 'work/conditioning/plot_cmd_latent_results/vagc_join.parquet'
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths['cmd_cache'], cache)
    result = run_steps(steps, out, configuration, env, restart=args.restart_incomplete)
    if args.stage == 'plots':
        target = out / 'figures/normalisation/norm_cmd.pdf'
        shutil.copy2(out / 'work/conditioning/norm_cmd/norm_cmd_full.pdf', target)
        result[str(target)] = sha256(target)
    (out / f'{args.stage}.json').write_text(json.dumps(dict(configuration=configuration, products=result), indent=2) + '\n')
    print(f'Completed {args.stage}: {out}', flush=True)
