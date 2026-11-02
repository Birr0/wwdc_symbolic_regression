# NORM conditioning and fibre-mass experiments

This suite packages the analysis behind the new NORM subsection, joint-conditioning
comparison and fibre-mass appendix. Start with [the walkthrough](../../docs/CONDITIONING.md).
The scripts preserve the existing calculations and dated artifact names; `suite.py`
provides a portable working directory and explicit data/upstream paths.

- `scripts/`: scoring, search, probes, structural diagnostics and figure generation.
- `source_manifest.json`: original and packaged source SHA-256 hashes, identifying adaptations.
- `suite.py.lock`: resolved experiment environment; use `uv run --script --locked suite.py`.
- `environment.json`: observed original direct dependency versions.
- `../../artifacts/conditioning/`: reference tables, external-input manifests and upstream patch.

`python experiments/conditioning/suite.py list` needs only Python's standard library.
Do not execute every listed module: some are shared helpers. Use the workflow in
the walkthrough. Full-size matrices, photometry, embeddings and checkpoints are
external inputs. This addition does not change the earlier redshift-only pipeline
or the separate exploratory `normalisation_flow.py` training implementation.
