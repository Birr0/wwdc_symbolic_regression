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
"""Stage and run the checkpoint-first conditioning reproduction experiments.

Only the standard library is needed for export, list and verify.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOTS = ('conditioning_validation_2026-10-13', 'fibre_mass_conditioning_2026-10-14',
         'norm_input_ablation_2026-10-16', 'shared_structure_audit_2026-10-14')
SUFFIXES = {'.csv', '.json', '.npz', '.npy', '.pkl', '.parquet', '.tex', '.md'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def check_code():
    for row in json.loads((HERE/'source_manifest.json').read_text())['files']:
        if digest(HERE/'scripts'/row['file']) != row['packaged_sha256']:
            raise ValueError(f"Packaged source changed: {row['file']}; review and update the source manifest")

def export(source, workdir):
    source, workdir = source.resolve(), workdir.resolve()
    if source == workdir or source in workdir.parents:
        raise ValueError('The work directory must be separate from the source archive')
    if workdir.exists():
        raise FileExistsError(f'Refusing to overwrite {workdir}')
    paths = []
    for root in ROOTS:
        folder = source/root
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in SUFFIXES)
    cache = source/'plot_cmd_latent_results/vagc_join.parquet'
    if not cache.is_file():
        raise FileNotFoundError(cache)
    paths.append(cache)
    # Preserve the exact saved candidate pools needed for validation/recovery.
    for folder in source.glob('job_results_*lgm_tot_p50_*seed*'):
        if ('normflow_condnorm_' in folder.name or
            folder.name.startswith(('job_results_lgm_tot_p50_cond_', 'job_results_lgm_tot_p50_cond+z_'))):
            paths.extend(folder.glob('pareto_fronts.csv'))
    workdir.mkdir(parents=True)
    records = []
    for p in sorted(set(paths)):
        relative = p.relative_to(source)
        target = workdir/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        records.append(dict(path=str(relative), bytes=target.stat().st_size, sha256=digest(target)))
    (workdir/'conditioning_inputs.json').write_text(json.dumps(dict(
        schema=1, source=str(source), files=records,
        note='Prepared matrices, frozen fronts and reference results; raw catalogues, embeddings and model weights remain external.'), indent=2)+'\n')
    print(f'Copied {len(records)} files ({sum(r["bytes"] for r in records)/1e6:.1f} MB) to {workdir}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List packaged scripts without importing heavy dependencies')
    p = sub.add_parser('export', help='Copy archived inputs/results into a new, writable work directory')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--workdir', type=Path, required=True)
    p = sub.add_parser('verify', help='Check packaged source and all exported input hashes')
    p.add_argument('--workdir', type=Path, required=True)
    p = sub.add_parser('run', help='Run a packaged script inside an isolated work directory')
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--upstream', type=Path, help='wwdc_spectra checkout, for conditional generation')
    p.add_argument('script', choices=sorted(p.stem for p in (HERE/'scripts').glob('*.py')))
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    check_code()
    if args.command == 'list':
        print('\n'.join(p.stem for p in sorted((HERE/'scripts').glob('*.py'))))
    elif args.command == 'export':
        export(args.source, args.workdir)
    elif args.command == 'verify':
        records = json.loads((args.workdir/'conditioning_inputs.json').read_text())['files']
        failures = [r['path'] for r in records if not (args.workdir/r['path']).is_file() or digest(args.workdir/r['path']) != r['sha256']]
        if failures:
            raise ValueError('Missing or changed inputs: '+', '.join(failures))
        print(f'Verified source and {len(records)} input files')
    else:
        work = args.workdir.resolve()
        if not (work/'conditioning_inputs.json').is_file():
            parser.error('Export inputs into a separate work directory first')
        # Scripts retain dated artifact paths and local imports. Stage code beside
        # their inputs so subprocesses and source hashing have identical semantics.
        for path in (HERE/'scripts').glob('*.py'):
            target = work/path.name
            if target.exists() and digest(target) != digest(path):
                raise FileExistsError(f'Changed staged code: {target}; retain it separately before updating')
            shutil.copy2(path, target)
        env = dict(os.environ, MPLBACKEND='Agg')
        if args.data_root:
            env['DATA_ROOT'] = str(args.data_root.resolve(strict=True))
        if args.upstream:
            upstream = args.upstream.resolve(strict=True)
            if not (upstream/'src/wwdc_spectra').is_dir():
                parser.error('--upstream must contain src/wwdc_spectra')
            env['WWDC_UPSTREAM'] = str(upstream)
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        command = [sys.executable, str(work/(args.script+'.py')), *arguments]
        print('Running:', ' '.join(command), flush=True)
        raise SystemExit(subprocess.call(command, cwd=work, env=env))

if __name__ == '__main__':
    main()
