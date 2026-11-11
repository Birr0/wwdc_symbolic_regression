"""Run symbolic-regression searches using NORM-conditioned embeddings.

Reuse ``run_total_mass.run_cell`` for fitting and scoring. With ``--append-norm``,
append the stored standardised log10(NORM) to the conditioned latents; ``cond+z``
also appends raw redshift.

Save ``pareto_fronts.csv`` and ``summary_metrics.csv`` under
``job_results_normflow_{feature}_{embed_type}_seed{seed}/``, using the prefix
``normflow_condnorm`` when NORM is appended. Refuse to overwrite existing runs.

Usage:
    python -m wwdc_repro.norm.run_normflow --embed-type cond+z --seed 42
    python -m wwdc_repro.norm.run_normflow --append-norm
"""

from wwdc_repro.workflows.run_total_mass import main as run_searches


def main() -> None:
    """Run the shared search CLI with NORM-conditioned embedding defaults."""
    run_searches(conditioning="norm")


if __name__ == "__main__":
    main()
