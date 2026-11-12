# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.4.4",
#   "pandas==3.0.2",
#   "scikit-learn==1.8.0",
#   "sympy==1.14.0",
#   "scipy==1.17.1",
#   "matplotlib==3.10.9",
#   "scienceplots==2.2.1",
#   "astroML==1.0.2.post1",
#   "astropy==7.2.0",
#   "pyarrow==24.0.0",
#   "PyYAML==6.0.3",
#   "tqdm==4.67.3",
#   "pyoperon==0.4.0",
#   "torch==2.11.0",
#   "torchvision==0.26.0",
#   "lightning==2.6.1",
#   "spender==0.2.9",
#   "flow-matching==1.0.10",
#   "torchdiffeq==0.2.5",
#   "datasets==4.8.5",
#   "timm==1.0.26",
#   "python-dotenv==1.2.2",
# ]
# ///
#!/usr/bin/env python3
"""Run packaged search and conditioning workflows in a data working directory.

Only the standard library is needed for export, list and verify.
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parents[2]
from wwdc_repro.shared.utils import sha256 as digest
from wwdc_repro.shared.paths import ARCHIVE_PATHS, working_path
WORKFLOWS = {
    'validate_conditioning': 'Recover, audit, freeze training or validation choices and score held-out equations',
    'rerun_conditioning_exports': 'Recover high-precision NORM equation exports',
    'matched_conditioning_controls': 'Prepare or fit matched original/z-flow SR controls',
    'norm_input_ablation': 'Section 6: explicit-NORM ablation fits and table',
    'run_total_mass': 'Regenerate redshift-conditioned or original candidate fronts',
    'run_normflow': 'Section 6: regenerate NORM-conditioned candidate fronts',
    'latent_mlp': 'Train sklearn log-NORM probes from prepared latent-only matrices',
    'norm_conditioning_probe': 'Section 6: latent-only NORM probe table',
    'generate_norm_sequence': 'Section 6: counterfactual spectral sequence',
    'plot_norm_frontier': 'Section 6: four-arm SR frontier figure',
    'plot_norm_cmd_full': 'Section 6: prepare additive S and render the full-sample CMD',
    'fibre_mass_conditioning': 'Appendix: prepare or fit original and conditioned fibre-mass SR',
    'fibre_mass_mlp_control': 'Appendix: train matched fibre-mass MLPs',
    'analyse_fibre': 'Appendix: performance table and equation structure',
}
SCRIPT_MODULES = {
    'matched_conditioning_controls': 'wwdc_repro.norm.matched_conditioning_controls',
    'run_normflow': 'wwdc_repro.norm.run_normflow',
    'run_total_mass': 'wwdc_repro.workflows.run_total_mass',
    'latent_mlp': 'wwdc_repro.norm.latent_mlp',
    'validate_conditioning': 'wwdc_repro.norm.validate_conditioning',
    'rerun_conditioning_exports': 'wwdc_repro.norm.rerun_conditioning_exports',
    'norm_input_ablation': 'wwdc_repro.norm.norm_input_ablation',
    'norm_conditioning_probe': 'wwdc_repro.norm.norm_conditioning_probe',
    'plot_norm_frontier': 'wwdc_repro.norm.plot_norm_frontier',
    'plot_norm_cmd_full': 'wwdc_repro.norm.plot_norm_cmd_full',
    'generate_norm_sequence': 'wwdc_repro.norm.generate_norm_sequence',
    'utils': 'wwdc_repro.shared.fitting',
    'conditioning_data': 'wwdc_repro.shared.conditioning_data',
    'fibre_mass_mlp_control': 'wwdc_repro.fibre_mass.fibre_mass_mlp_control',
    'analyse_fibre': 'wwdc_repro.fibre_mass.analyse_fibre',
    'fibre_mass_conditioning': 'wwdc_repro.fibre_mass.fibre_mass_conditioning',
}

ROOTS = ('norm_analysis', 'fibre_mass', 'norm_ablation')
SUFFIXES = {'.csv', '.json', '.npz', '.npy', '.pkl', '.parquet', '.tex', '.md'}


def export(source, workdir):
    '''
    Copy prepared conditioning inputs into a new working directory.
 
    Copy files with supported suffixes from the configured archive roots,
    the CMD catalogue cache, and matching saved Pareto-front CSVs. Write
    conditioning_inputs.json with the copied paths, sizes and SHA-256 hashes.
    
    Parameters
    ----------
    source : Path
        Root directory of the prepared conditioning archive.
    workdir : Path
        New destination directory, separate from and outside source.
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If workdir is source or lies inside it.
    FileExistsError
        If workdir already exists.
    FileNotFoundError
        If a required archive root or CMD cache is missing.
    OSError
        If directories cannot be created or files cannot be copied.
    
    Notes
    -----
    Copies files without modifying the source archive. An interrupted
    export may leave a partially populated working directory.
    '''
    source, workdir = source.resolve(), workdir.resolve()
    # check the source and workdirs
    if source == workdir or source in workdir.parents:
        raise ValueError('The work directory must be separate from the source archive')
    if workdir.exists():
        raise FileExistsError(f'Refusing to overwrite {workdir}')
    paths = []
    # build a list of files to copy
    for root in ROOTS:
        # A historical archive may also contain a same-named scripts folder.
        folder = source / ARCHIVE_PATHS["directories"][root]
        if not folder.is_dir():
            folder = source / root
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in SUFFIXES)
    cache = source/'plot_cmd_latent_results/vagc_join.parquet'
    if not cache.is_file():
        raise FileNotFoundError(cache)
    paths.append(cache)
    # Preserve the exact saved candidate pools needed for validation/recovery.
    for folder in source.glob('job_results_*lgm_tot_p50_*seed*'):
        # collect the saved equation fronts.
        if ('normflow_condnorm_' in folder.name or
            folder.name.startswith(('job_results_lgm_tot_p50_cond_', 'job_results_lgm_tot_p50_cond+z_'))):
            paths.extend(folder.glob('pareto_fronts.csv'))
    workdir.mkdir(parents=True)
    records = []
    for p in sorted(set(paths)):
        relative = p.relative_to(source)
        destination = working_path(relative)
        target = workdir / destination
        if target.exists():
            raise FileExistsError(f"Archive paths collide at {destination}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        records.append(dict(path=str(destination), source_path=str(relative), bytes=target.stat().st_size, sha256=digest(target)))
    (workdir/'conditioning_inputs.json').write_text(json.dumps(dict(
        schema=2, source=str(source), files=records,
        note='Prepared matrices, frozen fronts and reference results; raw catalogues, embeddings and model weights remain external.'), indent=2)+'\n')
    # record input metadata
    print(f'Copied {len(records)} files ({sum(r["bytes"] for r in records)/1e6:.1f} MB) to {workdir}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List packaged scripts without importing heavy dependencies')
    p = sub.add_parser('export', help='Copy archived inputs/results into a new, writable work directory')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--workdir', type=Path, required=True)
    p = sub.add_parser('verify', help='Check all exported input hashes')
    p.add_argument('--workdir', type=Path, required=True)
    p = sub.add_parser('run', help='Run a packaged script inside an isolated work directory')
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--upstream', type=Path, help='wwdc_spectra checkout, for conditional generation')
    p.add_argument('script', choices=sorted(WORKFLOWS))
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'list':
        for name, description in WORKFLOWS.items():
            print(f'{name:38} {description}')
    elif args.command == 'export':
        export(args.source, args.workdir)
    elif args.command == 'verify':
        records = json.loads((args.workdir/'conditioning_inputs.json').read_text())['files']
        failures = [r['path'] for r in records if not (args.workdir/r['path']).is_file() or digest(args.workdir/r['path']) != r['sha256']]
        if failures:
            raise ValueError('Missing or changed inputs: '+', '.join(failures))
        print(f'Verified {len(records)} input files')
    else:
        work = args.workdir.resolve()
        if not work.is_dir():
            parser.error('Create the working directory or export inputs there first')

        env = dict(os.environ, MPLBACKEND='Agg')
        # Run the shared package directly; output folders contain data, not staged code.
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (
            str(REPRO_ROOT / 'src'), env.get('PYTHONPATH', ''))))
        if args.data_root:
            env['DATA_ROOT'] = str(args.data_root.resolve(strict=True))
        if args.upstream:
            upstream = args.upstream.resolve(strict=True)
            if not (upstream/'src/wwdc_spectra').is_dir():
                parser.error('--upstream must contain src/wwdc_spectra')
            env['WWDC_UPSTREAM'] = str(upstream)
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        command = [sys.executable, '-m', SCRIPT_MODULES[args.script], *arguments]
        print('Running:', ' '.join(command), flush=True)
        raise SystemExit(subprocess.call(command, cwd=work, env=env))

if __name__ == '__main__':
    main()
