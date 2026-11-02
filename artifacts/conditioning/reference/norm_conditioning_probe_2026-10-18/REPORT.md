# Residual log10(NORM) information in the representations

Target: log10(NORM / one input flux-density unit), matching the flow conditioning transform. R² and Pearson correlations refer to log NORM, not linear NORM. Ten latent inputs only; neither z nor NORM appended. Existing matched splits and exactly the same 10,000 training rows for each of seeds 42–44. Coordinate correlations use the full test set. OLS has no tuned parameters. MLP: 64/32 ReLU, Adam 0.001, batch 1024, at most 500 epochs, patience 40; checkpoint selected on existing full validation set before test evaluation. Input/target scaling fitted to each 10k training subset.

| Representation | Max absolute coordinate Pearson r | Max absolute coordinate Spearman rho | Linear mean test R² | MLP mean test R² | MLP seed range | MLP RMSE [dex in NORM] |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.4925 | 0.4834 | 0.3325 | 0.5021 | 0.5000–0.5054 | 0.18245 |
| zflow | 0.3924 | 0.3958 | 0.4507 | 0.5352 | 0.5311–0.5376 | 0.17629 |
| norm | 0.0668 | 0.0681 | 0.0137 | 0.0308 | 0.0298–0.0318 | 0.25455 |
| znorm | 0.0771 | 0.0782 | 0.0210 | 0.0423 | 0.0412–0.0432 | 0.25304 |

These are predictive probes on the filtered total-mass sample, not proofs of independence or estimates of all recoverable information. A weak coordinate correlation does not rule out nonlinear prediction. NORM conditioning aims to reduce dependence on NORM, but finite probe performance cannot prove full statistical independence. Seed variation includes training subset and model initialization, not independent flow training. Raw R² may be negative when worse than the test-mean reference.

Epoch-cap counts: {'norm': 0, 'original': 0, 'zflow': 0, 'znorm': 0}. All saved MLP predictions and validation checkpoint epochs verified.

Reproduce: `../../../.venv/bin/python norm_conditioning_probe.py`. Preparation refuses overwrite. Training loop reused from `d4000_mlp.py`.
