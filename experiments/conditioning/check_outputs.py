"""Compare regenerated science tables with the small committed reference tables."""
from pathlib import Path
import argparse
import json
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wwdc_repro.shared.paths import working_path, ARCHIVE_PATHS

TABLES = ARCHIVE_PATHS['table_references']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[2]/'artifacts/conditioning/reference'
    results = []
    for relative in TABLES:
        expected = pd.read_csv(reference/relative)
        actual = pd.read_csv(args.workdir / working_path(relative))
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=1e-12)
        results.append(dict(path=relative, rows=len(actual), agrees=True))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
