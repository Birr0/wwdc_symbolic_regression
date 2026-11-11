"""Rebuild the historical CMD cache from explicitly pinned mirror snapshots."""

import argparse
import importlib.resources
import json
import shutil
from pathlib import Path

import pandas as pd
from wwdc_repro.shared.utils import sha256


def download():
    """Acquire only the five pinned shards; verify their recorded checksums."""
    parser = argparse.ArgumentParser(description=download.__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use existing Hub cache without network",
    )
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    sources = json.loads(
        importlib.resources.files("wwdc_repro.resources")
        .joinpath("catalogue_sources.json")
        .read_text()
    )
    for name, directory in (
        ("crossmatch", "vac-cross"),
        ("photometry", "vac-photometry"),
    ):
        source = sources[name]
        snapshot = Path(
            snapshot_download(
                repo_id=source["repo_id"],
                repo_type="dataset",
                revision=source["revision"],
                allow_patterns=[r["file"] for r in source["files"]],
                local_files_only=args.local_files_only,
            )
        )
        for row in source["files"]:
            path = snapshot / row["file"]
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Downloaded source hash mismatch: {path}")
            destination = args.outdir / directory / row["file"]
            if destination.exists():
                if sha256(destination) != row["sha256"]:
                    raise ValueError(
                        f"Refusing to overwrite different input: {destination}"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    print(f"Verified pinned catalogue shards under {args.outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crossmatch-dir", type=Path, required=True)
    parser.add_argument("--photometry-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".manifest.json").exists():
        parser.error("Output or manifest already exists")
    sources = json.loads(
        importlib.resources.files("wwdc_repro.resources")
        .joinpath("catalogue_sources.json")
        .read_text()
    )
    records = []
    frames = {}
    for name, root in (
        ("crossmatch", args.crossmatch_dir),
        ("photometry", args.photometry_dir),
    ):
        pieces = []
        for entry in sources[name]["files"]:
            path = root / entry["file"]
            if not path.is_file() or sha256(path) != entry["sha256"]:
                parser.error(f"Missing or different pinned source shard: {path}")
            pieces.append(pd.read_parquet(path))
            records.append({"path": str(path.resolve()), "sha256": entry["sha256"]})
        frames[name] = pd.concat(pieces, ignore_index=True)
    cross = frames["crossmatch"][["VAC_ID", "object_id"]]
    photo = frames["photometry"][
        ["VAC_ID", "ABSMAG_u", "ABSMAG_g", "ABSMAG_r", "MTOL_r", "MASS"]
    ]
    joined = cross.merge(
        photo, on="VAC_ID", how="inner", validate="many_to_one", sort=False
    )
    joined["specObjID"] = joined.object_id.astype("int64")
    ambiguous = int(
        joined.loc[joined.specObjID.duplicated(False), "specObjID"].nunique()
    )
    # Fixed shard order in catalogue_sources.json and original row order make
    # the historical first-match convention deterministic and inspectable.
    joined = joined.drop_duplicates("specObjID", keep="first")
    if len(joined) != 525505 or ambiguous != 13:
        raise ValueError("Pinned catalogue row/ambiguity counts differ from the audit")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(args.out, index=False)
    manifest = {
        "sources": sources,
        "inputs": records,
        "rows": len(joined),
        "ambiguous_spectrum_ids": ambiguous,
        "match_policy": "first row in fixed shard and source-row order; historical paper reproduction",
        "output_sha256": sha256(args.out),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Wrote {len(joined)} unique-spectrum cache rows; {ambiguous} IDs had multiple matches"
    )
