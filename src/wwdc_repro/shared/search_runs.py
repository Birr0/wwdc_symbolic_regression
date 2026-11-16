"""Shared completion checks and explicit restart handling for symbolic searches."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import uuid

from wwdc_repro.shared.utils import sha256


def input_hashes(paths):
    """Calculate SHA-256 hashes for search input files.
 
    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Input files used by the search.
    
    Returns
    -------
    dict[str, str]
        Absolute file paths mapped to their SHA-256 hashes.
    
    Raises
    ------
    OSError
        If an input file cannot be opened or read.
    
    Notes
    -----
    Hashes cover file contents only. Absolute paths are also recorded, so
    moving an unchanged file changes the returned dictionary.
    """
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


@contextmanager
def search_run(directory, configuration, outputs, *, restart_incomplete=False, completion_metadata=None):
    """Manage completion checks and restarts for one symbolic search.
 
    Parameters
    ----------
    directory : str or pathlib.Path
        Output directory for one representation and seed.
    configuration : dict
        Search settings and input hashes used to identify a matching run.
    outputs : sequence of str
        Required output filenames, relative to directory.
    restart_incomplete : bool, optional
        Preserve an incomplete or unverified attempt in incomplete_attempts/
        and start again. Default is False.
    completion_metadata : dict or None, optional
        Additional metadata to save in the completion record. The caller may
        populate this dictionary during execution; it is serialised only on
        successful completion and does not participate in configuration matching.
    
    Yields
    ------
    bool
        True if the caller should run the search; False if matching,
        verified outputs already exist.
    
    Raises
    ------
    RuntimeError
        If another process holds the lock for this search.
    FileExistsError
        If an incomplete or unverified directory exists and restarting
        has not been requested.
    ValueError
        If a completed run has different settings, inputs or required
        outputs, or a completed output is missing or changed.
    OSError
        If files cannot be read, written or moved.
    
    Notes
    -----
    Holds a process lock throughout the search. Records completion only
    after the caller exits successfully and all required outputs exist.
    Restarting begins a new search; it does not resume an Operon checkpoint.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    configuration = json.loads(json.dumps(configuration, default=str))
    with (directory.parent / f'.{directory.name}.lock').open('a') as lock:
        try:
            # prevent multiple processes writing to the same directory
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Search already running: {directory}') from exc
        marker = directory / 'search_complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['configuration'] != configuration:
                raise ValueError(f'Inputs/settings differ for {directory}; use a new output directory')
            if set(saved['outputs']) != set(outputs):
                raise ValueError(f'Completion output list differs for {directory}')
            for name, digest in saved['outputs'].items(): # verify each saved output
                path = directory / name
                if not path.is_file() or sha256(path) != digest:
                    raise ValueError(f'Completed output missing or changed: {path}')
            print(f'SKIP verified complete search: {directory}', flush=True)
            yield False
            return
        if directory.exists(): # check if a directory already exists if there is no completion record
            if not restart_incomplete:
                raise FileExistsError(f'Incomplete/unverified search: {directory}; use --restart-incomplete to preserve and restart it')
            archived = directory.parent / 'incomplete_attempts'
            archived.mkdir(exist_ok=True)
            directory.rename(archived / f'{directory.name}-{uuid.uuid4().hex}')
        directory.mkdir()
        (directory / 'search_started.json').write_text(json.dumps(configuration, indent=2) + '\n')
        yield True
        hashes = {name: sha256(directory / name) for name in outputs}
        temporary = directory / '.search_complete.tmp'
        record = dict(configuration=configuration, outputs=hashes)
        if completion_metadata is not None:
            record['metadata'] = completion_metadata
        temporary.write_text(json.dumps(record, indent=2) + '\n') # record completion
        temporary.replace(marker)


def add_search_options(parser):
    """Expose the same resource and restart options on every fitting CLI."""
    parser.add_argument('--threads', type=positive_int, default=8, help='Operon threads per search (default: 8)')
    parser.add_argument('--restart-incomplete', action='store_true', help='Preserve incomplete/unverified attempts and restart them')


def positive_int(value):
    import argparse
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number
