# Normalisation-only flow experiment

**Superseded:** the user requested the native `wwdc_spectra` workflow. The
standalone run was stopped after epoch 22; its checkpoints are retained. Use
`/home/birr0/Desktop/wwdc_spectra/src/conf/experiment/spender_I_normalisation_flow/train.yaml`
and that repository's `docs/normalisation_flow.md` for the active experiment,
which runs the standard trainer, spectrum loader, Spender encoder and W&B logger.

Requested 2026-10-12. This is a new exploratory model, not part of the archived
paper results. No paper checkpoint, embedding, or symbolic front is overwritten.

`normalisation_flow.py` imports the clean upstream `wwdc_spectra` implementation
at `0895b4ed9dbe6815cbe641fd1f936c6d53197aa5`. It trains the original 10-dimensional
Spender latent flow, with three 1024-wide layers, from a fresh seed-42
initialisation. The only conditioning input is

```text
(log10(N) - mean_train(log10(N))) / std_train(log10(N))
```

`N` is the exact factor returned by upstream `SDSS.prepare_spectrum`: the
`torch.median` over usable pixels in the rest-frame 5300–5850 Å window, including
the upstream spectrum mask and skyline mask. The older aperture-analysis cache
is not used: it uses NumPy's median and a different mask. The rest-frame window
itself uses the recorded redshift, although redshift is not an explicit flow
conditioning input.

The original archived train/validation/test membership is preserved. Original
Spender embeddings are joined to raw spectra one-to-one by exact 64-bit spectrum
ID, and split overlap is checked. Nonpositive/nonfinite normalisations and
nonfinite original embeddings are excluded and counted. There are no additional
stellar-mass catalogue, mask-ratio, or redshift cuts for flow training, matching
the parent flow sample rather than the downstream SR sample. This is the full
flow training split, not the 10,000-row SR fitting subsample. Embeddings are not
standardised; only the conditioning scalar is standardised on retained training
rows. The fixed encoder need not run again.

Training uses the upstream conditional/unconditional batch-doubled flow-matching
objective, AdamW at 1e-4, batch size 512, gradient clipping at 1, and mixed FP16.
Early stopping monitors validation loss (patience 10, min_delta 1e-4), with a
100-epoch cap. As in the upstream implementation, validation flow loss samples
fresh base noise and times; it is not a deterministic density estimate. Test
data are prepared but never passed to the trainer or checkpoint selection.

## Commands

From the reproduction repository root, using an environment with the upstream
training dependencies and a GPU-compatible PyTorch installation:

```sh
python experiments/normalisation_flow.py prepare \
  --upstream /path/to/pinned/wwdc_spectra \
  --data-root /path/to/local_data \
  --prepared outputs/normalisation-flow/data --workers 8

python experiments/normalisation_flow.py train \
  --upstream /path/to/pinned/wwdc_spectra \
  --data-root /path/to/local_data \
  --prepared outputs/normalisation-flow/data \
  --outdir outputs/normalisation-flow/pilot --epochs 1 --pilot-batches 32

python experiments/normalisation_flow.py train \
  --upstream /path/to/pinned/wwdc_spectra \
  --data-root /path/to/local_data \
  --prepared outputs/normalisation-flow/data \
  --outdir outputs/normalisation-flow/seed42
```

The pilot checks the same architecture, optimiser and dataset with limited
batches. The full run starts afresh from seed 42 rather than continuing the
pilot. Outputs include the exact retained IDs, per-row normalisation, source and
embedding hashes, training-only conditioning transform, CSV loss history, best
and last checkpoints, and completion metadata. This training environment is
separate from the lightweight SR reproduction environment; no new locked
training environment is claimed here.

After training, evaluate dependence on normalisation and redshift separately,
then compare stellar-mass prediction on matched downstream samples using
original latents, new latents alone, and new latents with normalisation restored.
Lower flow-matching loss alone does not establish successful removal of
normalisation dependence, or preservation of mass-prediction performance.

## Local run, 2026-10-12

Outputs: `outputs/normalisation-flow-2026-10-12/`. The environment used was the
parent workspace `.venv` (GPU PyTorch 2.11.0+cu130); exact package versions are
saved in the run's `config.json`. The pinned upstream checkout is locally at
`/tmp/wwdc-spectra-reproduction-check`.

| Split | Parent rows | Retained | Excluded normalisation |
| --- | ---: | ---: | ---: |
| Train | 422,481 | 411,742 | 10,739 |
| Validation | 52,810 | 51,493 | 1,317 |
| Test | 52,811 | 51,481 | 1,330 |

No original embeddings were nonfinite. In training, 10,667 exclusions had zero
normalisation and 72 had negative normalisation. The 32-batch GPU pilot completed
with finite losses. The full run was launched separately, and its first epoch
took 14.57 seconds with validation flow-matching loss 0.269845. This is an
initial training diagnostic, not a downstream prediction score. At that speed,
the 100-epoch cap is approximately 25 minutes; early stopping can finish sooner.

`training.log` records ongoing progress. `seed42/result.json` is written only on
successful completion, and names the best validation checkpoint. `launch.json`
records the exact command and process ID. The status at the time of this note
was **running**, not completed or evaluated.
