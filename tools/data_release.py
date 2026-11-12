"""Shared manifest and configuration helpers for the paper's HF data release."""

import json
import sys
from pathlib import Path
import shutil

PROFILES = {
    "saved": {"analysis", "conditioning", "generation"},
    "full": {"analysis", "conditioning", "generation", "models", "spectra", "pretrained"},
}
REDSHIFT = "sdss_II/spender_I_flow_v2/spender_I_flow_v2"
# Support direct execution without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wwdc_repro.shared.utils import sha256 as digest



def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe release path: {value}")
    return path


def rows_for(manifest, profile):
    rows = manifest["files"]
    names = [str(safe_relative(row["path"])) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate release paths")
    unknown = {row["component"] for row in rows} - PROFILES["full"]
    if unknown:
        raise ValueError(f"Unknown release components: {sorted(unknown)}")
    return [row for row in rows if row["component"] in PROFILES[profile]]


def verify(folder, manifest, profile="full"):
    rows = rows_for(manifest, profile)
    failures = []
    for row in rows:
        path = folder / safe_relative(row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            failures.append(row["path"])
    if failures:
        raise ValueError(f"Release files missing or changed ({len(failures)}): {failures[:8]}")
    return len(rows)


def write_configuration(folder, profile="saved"):
    """Write paths for fresh stages (full profile) or historical analysis (saved)."""
    folder = folder.resolve()
    config = {
        "embeddings": str(folder / "data" / REDSHIFT / "embeddings/7655991_0"),
        "galspec": str(folder / "data/galSpecExtra-dr8.fits"),
        "specgals_home": str(folder / "data/sdss"),
        "conditioning_archive": str(folder / "conditioning"),
        "norm_generation": str(folder / "generation"),
    }
    if profile == "full":
        config.pop("conditioning_archive")
        config.update(
            data_root=str(folder / "data"),
            norm_embeddings=str(folder / "data/sdss_II/spender_I_normalisation_flow/embeddings/norm_20261012183217_0"),
            joint_embeddings=str(folder / "data/sdss_II/spender_I_z_normalisation_flow/embeddings/znorm_20261013104946_0"),
            cmd_cache=str(folder / "conditioning/plot_cmd_latent_results/vagc_join.parquet"),
        )
    destination = folder / "reproduction.json"
    if destination.exists() and json.loads(destination.read_text()) != config:
        raise ValueError(f"Refusing to replace a different path configuration: {destination}")
    destination.write_text(json.dumps(config, indent=2) + "\n")
    return destination


def copy_local_bundle(source, destination, manifest, profile):
    """Test/reuse a locally staged release through the same integrity checks."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Local download destination must be separate from the source bundle")
    verify(source, manifest, profile)
    for row in rows_for(manifest, profile):
        relative = safe_relative(row["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != row["sha256"]:
                raise ValueError(f"Different existing download: {target}")
        else:
            shutil.copy2(source / relative, target)
    shutil.copy2(source / "release.json", destination / "release.json")
    verify(destination, manifest, profile)
    return write_configuration(destination, profile)
