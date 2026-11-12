"""Prepare, verify and upload the exact paper inputs to a Hugging Face dataset."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.data_release import REDSHIFT, digest, safe_relative, verify


def inventory(data_root, archive, generation, torch_home):
    """Use the audited file allowlists, never recursively upload a working tree."""
    files = {}

    def add(source, destination, component, expected=None):
        destination = str(safe_relative(destination))
        source = source.resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Not a file: {source}")
        checksum, size = digest(source), source.stat().st_size
        if expected and (checksum != expected["sha256"] or size != expected["bytes"]):
            raise ValueError(f"Audited input changed: {source}")
        row = {"path": destination, "component": component, "bytes": size, "sha256": checksum}
        if destination in files and files[destination][1] != row:
            # The analysis list already contains the redshift embedding shards.
            old_source, old = files[destination]
            if old_source != source or old["sha256"] != checksum or old["bytes"] != size:
                raise ValueError(f"Conflicting release input: {destination}")
            return
        files[destination] = (source, row)

    for row in json.loads((ROOT / "artifacts/analysis_inputs.json").read_text()):
        add(data_root / row["file"], "data/" + row["file"], "analysis", row)
    for row in json.loads((ROOT / "artifacts/conditioning/raw_inputs.json").read_text())["files"]:
        add(data_root / row["path"], "data/" + row["path"], "models", row)
    for row in json.loads((ROOT / "artifacts/conditioning/prepared_inputs.json").read_text())["files"]:
        add(archive / row["path"], "conditioning/" + row["path"], "conditioning", row)
    saved = data_root / "sdss/sdss_II_catalog"
    if not (saved / "dataset_dict.json").is_file():
        raise ValueError(f"Archived spectrum DatasetDict required: {saved}")
    spectrum_files = [saved / "dataset_dict.json"]
    for split in ("train", "val", "test"):
        folder = saved / split
        spectrum_files.extend([folder / "state.json", folder / "dataset_info.json"])
        state = json.loads((folder / "state.json").read_text())
        for row in state["_data_files"]:
            spectrum_files.append(folder / safe_relative(row["filename"]))
    for path in spectrum_files:
        add(path, "data/" + str(path.relative_to(data_root)), "spectra")
    norm_caches = (
        "e15d9586ef33858629a065073510488f4a1e7af26d02a0898ba9994d02a06e6e.npy",
        "a929b6b7fb344d5b75eec0bf0de4800c034e0f462021ef95a18ccd27d82729d7.npy",
        "02eb144c0e925240855c69eea7499a91a6f8d90c52a28c577db68ce003a8d281.npy",
    )
    for name in norm_caches:
        path = data_root / "sdss/normalisation_cache" / name
        add(path, "data/" + str(path.relative_to(data_root)), "models")
    for name in ("generated_spectra.npz", "observed_spectra.npz", "metadata.json"):
        add(generation / name, "generation/" + name, "generation")
    spender = torch_home / "hub/checkpoints/spender.sdss.paperI-08798cbc.pt"
    add(spender, "pretrained/torch/hub/checkpoints/" + spender.name, "pretrained")
    return files


def prepare(args):
    if args.outdir.exists():
        raise FileExistsError(f"Use a new bundle directory: {args.outdir}")
    print("Hashing and checking exact paper inputs, including spectrum shards…", flush=True)
    files = inventory(args.data_root.resolve(), args.archive.resolve(), args.generation.resolve(), args.torch_home.resolve())
    totals = defaultdict(lambda: {"files": 0, "bytes": 0})
    args.outdir.mkdir(parents=True)
    for name, (source, row) in sorted(files.items()):
        destination = args.outdir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(source, destination)
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        totals[row["component"]]["files"] += 1
        totals[row["component"]]["bytes"] += row["bytes"]
    manifest = {
        "schema": 1, "release": "symbolic-regression-v1", "repo_id": args.repo_id,
        "components": dict(totals), "files": [row for _, row in sorted(files.values(), key=lambda item: item[1]["path"])],
        "parent_rows": {"train": 422481, "val": 52810, "test": 52811},
        "normalisation": "Median usable flux density over rest-frame 5300–5850 Angstrom; training-only log10 scaling",
        "training_provenance": "Current observed upstream source and archived configuration; not proof of exact historical source or bit-identical retraining.",
    }
    (args.outdir / "release.json").write_text(json.dumps(manifest, indent=2) + "\n")
    table = "\n".join(f"| {key} | {value['files']} | {value['bytes']/1e9:.3f} |" for key, value in sorted(totals.items()))
    card = f"""---
pretty_name: WWDC symbolic-regression reproduction inputs
tags:
- astronomy
- symbolic-regression
- reproducibility
---

# WWDC symbolic-regression reproduction inputs

Exact archived inputs for [wwdc_symbolic_regression](https://github.com/Birr0/wwdc_symbolic_regression).
Training and embedding code: [wwdc_spectra](https://github.com/Birr0/wwdc_spectra).
`release.json` records every released file's size and SHA-256.

| Component | Files | GB |
|---|---:|---:|
{table}

Use `download_data.py` in the reproduction repository. Its `saved` profile
downloads analysis, conditioning and saved generation products; `full` also
downloads archived spectra, model products and the pretrained Spender cache.
The downloader pins an immutable HF revision, verifies SHA-256 values and writes
the input-path file: `full` supports `reproduce.py equations`, `plots` and
`tables`; `saved` supports the historical `python3 reproduce.py` workflow.

The spectra retain the archived 422,481 / 52,810 / 52,811 parent rows and row order.
Splits separate spectrum IDs; they are not guaranteed to separate physical galaxies.
Prepared inputs include frozen expressions, matched rows and training indices.
Some prepared frames use Python pickle: load only the files matching this release's hashes.
Generation arrays permit replotting; they do not establish fresh checkpoint inference.

## Sources and attribution

- Original spectra: [Multimodal Universe SDSS pipeline](https://github.com/MultimodalUniverse/MultimodalUniverse/tree/main/scripts/sdss), preserving the author's saved splits.
- Analysis catalogues: astroML SDSSspecgals DR8 files and SDSS DR8 galSpecExtra, identified by checksums in the code repository.
- CMD photometry: the pinned NYU VAGC mirror inputs; original crossmatch provenance is limited.
- Pretrained encoder: Spender `sdss_I`, cached as `spender.sdss.paperI-08798cbc.pt`.
- Flow checkpoints, embeddings and derived analysis products: the WWDC paper workflows.

Repository code licensing does not relicense third-party catalogues, spectra or
pretrained weights. Preserve the original sources' attribution and applicable terms.
No blanket dataset licence is assigned by this card.

Exact historical training source is not fully established. The training repository
provides archived-configuration recipes and observed source; it does not promise
bit-identical retraining across systems. Checkpoints and original embeddings are
included so numerical reproduction does not depend on retraining.
"""
    (args.outdir / "README.md").write_text(card)
    print(json.dumps({"files": len(files), "GB": sum(row["bytes"] for _, row in files.values()) / 1e9,
                      "components": dict(totals), "bundle": str(args.outdir.resolve())}, indent=2))


def upload(args):
    from huggingface_hub import HfApi, get_token, hf_hub_download
    token = get_token()
    if not token:
        raise ValueError("No Hugging Face credential is configured. Run `hf auth login` in your normal terminal; never put a token in Git.")
    manifest = json.loads((args.folder / "release.json").read_text())
    verify(args.folder, manifest)
    api = HfApi(token=token)
    repo_id = manifest["repo_id"]
    private = args.visibility == "private"
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if bool(info.private) != private:
        raise ValueError("Existing repository visibility differs; no upload was attempted")
    # Exact allowlist prevents release-local caches or unrelated files being sent.
    expected = ["release.json", "README.md", *[row["path"] for row in manifest["files"]]]
    api.upload_large_folder(repo_id, folder_path=args.folder, repo_type="dataset", allow_patterns=expected)
    revision = api.repo_info(repo_id, repo_type="dataset").sha
    remote = Path(hf_hub_download(repo_id, "release.json", repo_type="dataset", revision=revision, token=token))
    checksum = digest(args.folder / "release.json")
    if digest(remote) != checksum:
        raise ValueError("Remote release manifest differs after upload; release lock was not updated")
    published = set(api.list_repo_files(repo_id, repo_type="dataset", revision=revision))
    if set(expected) - published:
        raise ValueError("Remote release is missing files; release lock was not updated")
    lock = {"status": "published", "repo_id": repo_id, "revision": revision, "manifest_sha256": checksum,
            "visibility": args.visibility, "files": len(manifest["files"])}
    args.lock_out.parent.mkdir(parents=True, exist_ok=True)
    args.lock_out.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"Uploaded and pinned {repo_id}@{revision}. Commit {args.lock_out} to publish the download target.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--generation", type=Path, required=True)
    p.add_argument("--torch-home", type=Path, default=Path.home() / ".cache/torch")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--repo-id", default="Birr001/wwdc-symbolic-regression")
    p.add_argument("--copy", action="store_true", help="Copy bytes; default uses hardlinks on the same filesystem")
    p = sub.add_parser("verify")
    p.add_argument("--folder", type=Path, required=True)
    p = sub.add_parser("upload")
    p.add_argument("--folder", type=Path, required=True)
    p.add_argument("--visibility", choices=["public", "private"], required=True)
    p.add_argument("--lock-out", type=Path, default=ROOT / "artifacts/data_release.json")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "upload":
            upload(args)
        else:
            manifest = json.loads((args.folder / "release.json").read_text())
            print(f"Verified {verify(args.folder, manifest)} release files")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
