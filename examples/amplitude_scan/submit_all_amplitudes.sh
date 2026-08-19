#!/bin/bash

AMPLITUDES=(
    01 02 03 04 05 06 07 08 09 10
    15 20 25 30 35 40 45 50
)

for AMP in "${AMPLITUDES[@]}"; do

    SCAN_DIR="amp_${AMP}urad"

    echo "Submitting ${SCAN_DIR}"

    sbatch \
        --array=0-1000%200 \
        --output="${SCAN_DIR}/logs/festa2_%A_%a.out" \
        --error="${SCAN_DIR}/logs/festa2_%A_%a.err" \
        submit_scan.slurm \
        "${SCAN_DIR}"

done
