"""Shared standard-library utilities for reproduction scripts."""

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    '''
    Create a sha256 hash of a file to check the same file has been used.

    Parameters
    ----------
    path: Path 
        The path to the file 
    
    Returns
    -------
    hash: str 
        Hash of the file
    '''
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

