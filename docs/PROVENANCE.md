# Provenance and boundaries

## Embeddings and checkpoint

The upstream implementation is
[`Birr0/wwdc_spectra`](https://github.com/Birr0/wwdc_spectra), inspected at
commit [`0895b4ed9dbe6815cbe641fd1f936c6d53197aa5`](https://github.com/Birr0/wwdc_spectra/tree/0895b4ed9dbe6815cbe641fd1f936c6d53197aa5).
Its README describes the Hydra-driven training command in
`src/wwdc_spectra/training/train.py` and the embedding-generation command in
`src/wwdc_spectra/inference/embed.py`. The resulting parquet files contain
`orig`, `cond`, and `uncond` representations. The checkpoint and its exact
configuration still need a public artifact reference and hash here. Merely
linking the upstream code does not establish the historical checkpoint's
training run or a successful fresh embedding run.

## Downstream numerical analysis

The local source of this package was the reviewed analysis under
`examples/spectra/symbolic_regression/` and the pre-submission audit under
`papers/wwdc_symbolic_regression/pre-submission/audit/` in the `flower`
workspace. The data join follows `val_rescore.py`: saved split embeddings,
astroML SDSS specgals, then galSpecExtra; it applies `mask_ratio <= 0.5`,
`z <= 0.3`, and valid `lgm_tot_p50`. The astroML source itself has `z > 0.02`.

The candidates are saved `pareto_fronts.csv` files. This package evaluates
their printed expressions, rejects training-score failures, selects on the
held-out validation split, and reports saved test scores. The saved fronts
are inputs, not regenerated search outputs. Search settings and runtime should
be documented, but exact replay from a seed is not claimed.

## Work remaining for the public artifact

- Record public locations, licences, revision IDs and checksums for all inputs.
- Add the checkpoint-to-embeddings command and broaden its numerical comparison.
- Integrate the six other active figures into clean parameterised commands.
- Define the CMD mirror version and deterministic match/quality rule.
- Check a fresh clone on a separate machine and record the resulting manifest.
