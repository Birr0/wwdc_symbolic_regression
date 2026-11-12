"""Run latent-only log10(NORM) probes with the shared sklearn MLP trainer."""

from pathlib import Path
import argparse

from wwdc_repro.shared.mlp import run_mlp_experiment

BASE = Path("norm_probe")
ARMS = ("original", "zflow", "norm")


def run(base=BASE, outdir=None, arms=ARMS):
    """Predict log NORM from ten latents using the saved matched splits."""
    base = Path(base)
    return run_mlp_experiment(
        base,
        Path(outdir) if outdir is not None else base / "mlp_sklearn",
        arms=arms,
        n_features=10,
        target="log10(NORM); standardized on fitted training subset",
        inputs="ten latent coordinates only; standardized on the 10000 training rows per seed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    run(args.base, args.outdir)


if __name__ == "__main__":
    main()
