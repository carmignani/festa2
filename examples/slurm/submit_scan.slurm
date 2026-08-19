#!/bin/bash

#SBATCH --job-name=festa2_amp
#SBATCH --partition=asd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# IMPORTANT: update wall time when changing N_TURNS
#SBATCH --time=03:00:00

#SBATCH --mem=2G

set -euo pipefail


if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch submit_scan.slurm <scan_dir>"
    exit 1
fi


SCAN_DIR="$1"


echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Scan directory: ${SCAN_DIR}"
echo "Start time: $(date)"


# Activate FESTA2 Python environment
source /mnt/multipath-shares/machfs/carmigna/env/pynic/bin/activate


# Prevent numerical libraries from using extra threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


python -u run_frequency.py \
    "${SLURM_ARRAY_TASK_ID}" \
    "${SCAN_DIR}"


echo "End time: $(date)"
