# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.4.4",
#   "pandas==3.0.2",
#   "scikit-learn==1.8.0",
#   "sympy==1.14.0",
#   "scipy==1.17.1",
#   "matplotlib==3.10.9",
#   "scienceplots==2.2.1",
#   "astroML==1.0.2.post1",
#   "astropy==7.2.0",
#   "pyarrow==24.0.0",
#   "PyYAML==6.0.3",
#   "tqdm==4.67.3",
#   "pyoperon==0.4.0",
#   "torch==2.11.0",
#   "torchvision==0.26.0",
#   "lightning==2.6.1",
#   "spender==0.2.9",
#   "flow-matching==1.0.10",
#   "torchdiffeq==0.2.5",
#   "datasets==4.8.5",
#   "timm==1.0.26",
#   "python-dotenv==1.2.2",
# ]
# ///
"""Compatibility launcher for wwdc_repro.workflows.conditioning_suite."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wwdc_repro.workflows.conditioning_suite import WORKFLOWS, SCRIPT_MODULES, export, main

if __name__ == "__main__":
    main()
