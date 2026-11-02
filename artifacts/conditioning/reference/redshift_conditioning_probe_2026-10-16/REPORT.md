# Residual redshift information in the representations

Ten latent inputs only; neither z nor NORM appended. Existing matched splits and exactly the same 10,000 training rows for each of seeds 42–44. Coordinate correlations use the full test set. OLS has no tuned parameters. MLP: 64/32 ReLU, Adam 0.001, batch 1024, at most 500 epochs, patience 40; checkpoint selected on existing full validation set before test evaluation. Input/target scaling fitted to each 10k training subset.

| Representation | Max absolute coordinate Pearson r | Max absolute coordinate Spearman rho | Linear mean test R² | MLP mean test R² | MLP seed range | MLP RMSE in z |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.5285 | 0.5635 | 0.4148 | 0.5814 | 0.5801–0.5831 | 0.03354 |
| zflow | 0.0567 | 0.0556 | 0.0063 | 0.0168 | 0.0154–0.0184 | 0.05140 |
| norm | 0.3423 | 0.3502 | 0.4544 | 0.5451 | 0.5416–0.5470 | 0.03496 |
| znorm | 0.0982 | 0.0969 | 0.0244 | 0.0377 | 0.0365–0.0390 | 0.05085 |

These are predictive probes on the filtered total-mass sample, not proofs of independence or estimates of all recoverable information. A weak coordinate correlation does not rule out nonlinear prediction. NORM conditioning does not explicitly impose independence from redshift. Seed variation includes training subset and model initialization, not independent flow training. Raw R² may be negative when worse than the test-mean reference.

Epoch-cap counts: {'norm': 0, 'original': 0, 'zflow': 0, 'znorm': 0}. All saved MLP predictions and validation checkpoint epochs verified.

Reproduce: `../../../.venv/bin/python redshift_conditioning_probe.py`. Preparation refuses overwrite. Training loop reused from `d4000_mlp.py`.
