# Documentation

Start with [the reproduction walkthrough](QUICKSTART.md). The pages below are
technical reference; you do not need to read them to run the normal workflow.

This repository covers the symbolic-regression analysis. Embedding training is
documented in the upstream [wwdc_spectra](https://github.com/Birr0/wwdc_spectra)
project; our checkpoint wrapper uses a pinned revision.

| I want to… | Start here |
|---|---|
| Review every paper figure and table | [Paper review inventory](PAPER_REVIEW.md) |
| Reproduce the saved-equation analysis | [Walkthrough](QUICKSTART.md) |
| Run checkpoint inference or train models/probes | [Advanced reproduction](ADVANCED.md) |
| Understand the modules and scientific decisions | [Code guide](CODE_GUIDE.md) |
| Inspect catalogue joins, filtering, and scaling | [Data preparation review](DATA_REVIEW.md) |
| Inspect equation evaluation and model selection | [Equation/selection review](EQUATION_SELECTION_REVIEW.md) |
| Obtain and verify the inputs | [Inputs](INPUTS.md) |
| Download the HF bundle or finish publication | [Data release](DATA_RELEASE.md) |
| Recreate or inspect a plot | [Figure calculations](FIGURES.md) |
| Find the purpose of every experiment script | [Script guide](SCRIPTS.md) |
| Reproduce Section 6 (spectral normalisation) and fibre controls | [Conditioning walkthrough](CONDITIONING.md) |
| Rerun a symbolic search or inspect MLP experiments | [Experiments](../experiments/README.md) |
| Understand validation selection and test failures | [Selection audit](SELECTION_AUDIT.md) |
| See what has actually been checked | [Validation evidence](VALIDATION.md) |
| Check provenance and limitations | [Provenance](PROVENANCE.md) |
| Prepare the public release | [Release work list](REPOSITORY_PREPARATION.md) |
| See what the review release includes | [Release status](RELEASE_STATUS.md) |

The release work list records unfinished work. Historical verification reports
describe the versions and inputs they checked, rather than guaranteeing every
subsequent change.
