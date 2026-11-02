# Hugging Face paper data release

The local release bundle is prepared and verified. **It has not been uploaded.**
This session has no HF credential and its network sandbox blocks sockets.
The intended dataset is `Birr001/wwdc-symbolic-regression`; repository visibility
is supplied explicitly to the upload command. `artifacts/data_release.json`
remains unpublished until the upload succeeds and pins an immutable HF revision.

## For readers, once published

```bash
uv sync --frozen --extra catalogues
uv run --frozen --extra catalogues python download_data.py
python3 reproduce.py --config data/paper/reproduction.json
```

The downloader verifies every selected file's SHA-256 and creates the path
configuration automatically. Its default `saved` profile is about 1.78 GB.
For the neural training/inference inputs as well:

```bash
uv run --frozen --extra catalogues python download_data.py --profile full
```

The full profile is about 35.95 GB. It preserves the exact spectrum splits,
row order and model products; a new raw download/random split is not equivalent.
Training code is in [wwdc_spectra](https://github.com/Birr0/wwdc_spectra), with
the walkthrough in `docs/PAPER_TRAINING.md`.

| Component | Files | Approx. GB | Included in |
|---|---:|---:|---|
| Analysis embeddings and catalogues | 1,037 | 0.616 | saved, full |
| Prepared NORM/fibre matrices, candidates and summaries | 293 | 1.157 | saved, full |
| Saved spectral-sequence arrays and metadata | 3 | 0.004 | saved, full |
| Checkpoints, companion YAMLs, NORM/joint embeddings and normalisation cache | 15 | 1.094 | full |
| Archived spectrum DatasetDict | 74 | 33.043 | full |
| Exact Spender pretrained cache | 1 | 0.038 | full |

The 1,423-file allowlist and checksums are in
[`data_release_manifest.json`](../artifacts/data_release_manifest.json).
The manifest makes the package contents reviewable before publication.
The dataset card preserves source attribution and does not assign a blanket
licence to third-party spectra, catalogues or pretrained weights.

## For the author: finish publication

The prepared folder on the author machine is:

```text
/home/birr0/Desktop/wwdc_release/symbolic-regression-v1
```

From the reproduction repository in a normal network-enabled terminal:

Push the prepared training commit from `wwdc_spectra` and the downloader commit
from this repository first, so the pinned source is available to readers.

```bash
uv sync --frozen --extra catalogues
uv run --frozen --extra catalogues hf auth login
uv run --frozen --extra catalogues python tools/release_data.py upload \
  --folder /home/birr0/Desktop/wwdc_release/symbolic-regression-v1 \
  --visibility public
```

Use `--visibility private` if the release is intended to require reader access.
The tool refuses a visibility mismatch for an existing dataset and never changes
its visibility. Authentication stays outside the repository; no token is needed
in command arguments. Large uploads use the HF resumable uploader; rerun the same
command after an interruption.

After uploading, the tool checks the remote manifest and complete file list,
then writes the immutable HF revision and manifest SHA to
`artifacts/data_release.json`. It leaves that lock unchanged after a failed or
incomplete upload. Commit and push that file to enable the reader download command:

```bash
git add artifacts/data_release.json
git commit -m "Pin published paper data release"
git push origin main
```

Update the pending-publication notices in this page and the main README after
the remote upload/download has been verified.

Then test a fresh download into a separate directory and run `reproduce.py` using
the generated JSON. This remote round trip remains unverified until publication.

## Rebuild or verify the bundle

The preparation command checks the existing audit manifests and stages only
explicitly selected files. It never uploads a whole local data or source tree.

```bash
python3 tools/release_data.py prepare \
  --data-root /path/to/original-data \
  --archive /path/to/original-symbolic-regression-archive \
  --generation /path/to/norm_generation_random_seed42_five \
  --torch-home /path/to/torch-cache \
  --outdir /path/to/new-release-folder \
  --repo-id Birr001/wwdc-symbolic-regression
python3 tools/release_data.py verify --folder /path/to/new-release-folder
```

Staging uses hardlinks on the same filesystem to avoid duplicating 36 GB; the
staging folder is not an independent backup and should be treated as read-only.
Use `--copy` for independent byte copies. Do not modify source or staged files
after verification. The uploader rechecks every file before publishing.

The local saved-profile download/copy and generated configuration were tested
against this bundle. You can repeat that check before publication:

```bash
python3 download_data.py --local-bundle /path/to/release-folder --outdir /path/to/new-local-download
python3 reproduce.py --config /path/to/new-local-download/reproduction.json --check
```
