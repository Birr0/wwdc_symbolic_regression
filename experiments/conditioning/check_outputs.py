"""Compare regenerated science tables with the small committed reference tables."""
from pathlib import Path
import argparse
import json
import pandas as pd

TABLES = (
    'conditioning_validation_2026-10-13/validation_selected_final.csv',
    'norm_input_ablation_2026-10-16/summary.csv',
    'fibre_mass_conditioning_2026-10-14/original_control_comparison/performance.csv',
    'norm_joint_s_comparison_2026-10-16/comparison.csv',
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[2]/'artifacts/conditioning/reference'
    results = []
    for relative in TABLES:
        expected = pd.read_csv(reference/relative)
        actual = pd.read_csv(args.workdir/relative)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-12)
        results.append(dict(path=relative, rows=len(actual), agrees=True))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
