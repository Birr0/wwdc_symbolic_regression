"""Download a pinned paper data release, verify it and write reproduction.json."""

import argparse
import json
from pathlib import Path

from tools.data_release import copy_local_bundle, digest, rows_for, verify, write_configuration

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("data/paper"))
    parser.add_argument("--profile", choices=["saved", "full"], default="saved",
                        help="saved: analysis; full: also spectra, checkpoints, embeddings and Spender cache")
    parser.add_argument("--release-lock", type=Path, default=ROOT / "artifacts/data_release.json")
    parser.add_argument("--local-bundle", type=Path, help="Use a locally staged bundle instead of HF; useful before publication")
    args = parser.parse_args()
    try:
        if args.local_bundle:
            manifest = json.loads((args.local_bundle / "release.json").read_text())
            config = copy_local_bundle(args.local_bundle, args.outdir, manifest, args.profile)
        else:
            release = json.loads(args.release_lock.read_text())
            if release.get("status") != "published" or not release.get("revision") or not release.get("manifest_sha256"):
                raise ValueError("The HF release has not been published/pinned yet. See docs/DATA_RELEASE.md.")
            from huggingface_hub import hf_hub_download, snapshot_download
            path = Path(hf_hub_download(release["repo_id"], "release.json", repo_type="dataset", revision=release["revision"]))
            if digest(path) != release["manifest_sha256"]:
                raise ValueError("Downloaded release manifest does not match the pinned SHA-256")
            manifest = json.loads(path.read_text())
            rows = rows_for(manifest, args.profile)
            print(f"Downloading {len(rows)} files ({sum(r['bytes'] for r in rows)/1e9:.2f} GB).", flush=True)
            snapshot_download(release["repo_id"], repo_type="dataset", revision=release["revision"],
                              local_dir=args.outdir, allow_patterns=["release.json", "README.md", *[r["path"] for r in rows]])
            verify(args.outdir, manifest, args.profile)
            config = write_configuration(args.outdir)
        print(f"Verified {args.profile} inputs. Run: python3 reproduce.py --config {config}")
        if args.profile == "full":
            print(f"Training data root: {args.outdir.resolve() / 'data'}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
