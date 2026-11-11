# Package layout

- `shared/`: common data preparation, equation evaluation, selection, symbolic
  fitting, MLPs, CMD helpers, styles and file hashing.
- `redshift/`: redshift-conditioned analysis, figures and tables.
- `norm/`: NORM-conditioned analysis, controls, figures and spectral generation.
- `fibre_mass/`: fibre-mass preparation, fitting and comparisons.
- `workflows/`: commands coordinating searches and reproduction.
- `resources/`: packaged reference results and catalogue metadata.

Topic-specific code calls `shared/`; it does not duplicate preprocessing or
fitting. `workflows/run_total_mass.py` serves both redshift and NORM searches.

Small root-level launchers preserve existing installed CLI commands. Direct
execution uses the topic path, for example:
`python -m wwdc_repro.norm.generate_norm_sequence --help`.
The existing `experiments/conditioning/suite.py` command is also supported.
