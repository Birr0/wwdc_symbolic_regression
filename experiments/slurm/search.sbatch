#!/usr/bin/env bash
#SBATCH --job-name=wwdc-sr
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/search-%A_%a.log
set -euo pipefail

# Submit from the reproduction repository. Configure a pinned search environment
# and a shared working directory containing any required prepared matrices.
: "${SEARCH_PYTHON:?Set SEARCH_PYTHON to the absolute Python path in your search environment}"
: "${SEARCH_WORKDIR:?Set SEARCH_WORKDIR to an absolute search working directory}"
repo_root="${SLURM_SUBMIT_DIR:?Submit with sbatch from the reproduction repository}"
[[ -d "$repo_root/src/wwdc_repro" ]] || { echo "Submit from the reproduction repository" >&2; exit 1; }
[[ "$SEARCH_PYTHON" = /* && -x "$SEARCH_PYTHON" ]] || { echo "SEARCH_PYTHON must be an executable absolute path" >&2; exit 1; }
[[ "$SEARCH_WORKDIR" = /* ]] || { echo "SEARCH_WORKDIR must be absolute" >&2; exit 1; }
(( $# >= 1 )) || { echo "Usage: sbatch search.sbatch module [arguments...]" >&2; exit 1; }
seeds=(42 43 44)
index="${SLURM_ARRAY_TASK_ID:?Missing array index}"
[[ "$index" =~ ^[0-2]$ ]] || { echo "Array index must be 0, 1 or 2" >&2; exit 1; }
module="$1"
shift
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$SEARCH_WORKDIR"
cd "$SEARCH_WORKDIR"
srun "$SEARCH_PYTHON" -m "$module" "$@" --seed "${seeds[$index]}" --threads "${SLURM_CPUS_PER_TASK:-8}"
