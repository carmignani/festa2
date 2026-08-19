# ESRF-EBS example

This example reproduces a resonant depolarization frequency scan for the
ESRF-EBS storage ring.

## Contents

* `betamodel.mat` : ESRF-EBS AT lattice required locally (not distributed
with this repository).
- `prepare_scan.py` : generate the scan input (`scan_input.pkl`).

## Typical workflow

```bash
python prepare_scan.py
```

Submit the scan using the Slurm example:

```bash
../slurm/submit_scan.slurm
```

Then use the analysis scripts in `examples/amplitude_scan/` to merge and
analyze the simulation results.