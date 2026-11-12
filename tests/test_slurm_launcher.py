"""Check SLURM argument routing without submitting a cluster job."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SlurmLauncherTests(unittest.TestCase):
    def test_seed_threads_and_shared_module_are_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            stub = home / 'srun'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\npwd > "$CAPTURE.cwd"\n')
            stub.chmod(0o755)
            env = dict(os.environ, PATH=f'{home}:{os.environ["PATH"]}',
                       SLURM_SUBMIT_DIR=str(ROOT), SLURM_ARRAY_TASK_ID='1',
                       SLURM_CPUS_PER_TASK='4', SEARCH_PYTHON=sys.executable,
                       SEARCH_WORKDIR=str(home / 'work'), CAPTURE=str(home / 'args'))
            subprocess.run(['bash', str(ROOT / 'experiments/slurm/search.sbatch'),
                            'wwdc_repro.fibre_mass.fibre_mass_conditioning', 'fit',
                            '--arm', 'original', '--restart-incomplete'], env=env, check=True)
            self.assertEqual((home / 'args').read_text().splitlines(), [sys.executable, '-m',
                             'wwdc_repro.fibre_mass.fibre_mass_conditioning', 'fit', '--arm',
                             'original', '--restart-incomplete', '--seed', '43', '--threads', '4'])
            self.assertEqual((home / 'args.cwd').read_text().strip(), str(home / 'work'))
