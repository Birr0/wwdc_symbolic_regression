"""Translate historical archive paths to stable working-directory names.

Dates identify immutable input archives only; active producers use stable paths.
"""
import json
from importlib.resources import files
from pathlib import Path

ARCHIVE_PATHS = json.loads(files('wwdc_repro.resources').joinpath('archive_paths.json').read_text())


def working_path(relative: Path | str) -> Path:
    """Map archived directory components while preserving file names."""
    aliases = {old: new for new, old in ARCHIVE_PATHS['directories'].items()}
    return Path(*(aliases.get(part, part) for part in Path(relative).parts))
