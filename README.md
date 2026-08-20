# FESTA

**Fast Electron Spin Tracking for Accelerators**

Version 2.0

FESTA is an open-source Python framework for spin tracking simulations
and resonant depolarization studies in electron storage rings.

FESTA combines orbital tracking with Accelerator Toolbox (pyAT) and
spin-orbit quantities computed with Xsuite to simulate spin dynamics and
resonant depolarization with radiation damping and quantum excitation.

---

## Features

- Spin tracking of particle ensembles
- Orbit Angle Matrix (OAM) computation from Xsuite
- Fast-ring orbital tracking with python Accelerator Toolbox (pyAT)
- Radiation damping and quantum excitation
- RF shaker simulations
- Frequency scans for resonant depolarization studies
- Parallel execution on HPC clusters
- Analysis tools for polarization spectra and beam dynamics

---

## Current applications

FESTA is currently used to study resonant depolarization in electron
storage rings. Current and planned applications include:

- ESRF-EBS
- ESRF (2015 lattice)
- SOLEIL

---

## Repository structure

```
src/
    festa2/          Python package

scripts/
    Simulation and analysis scripts

examples/
    Example simulations
```

---

## Installation

Coming soon.

---

## History

FESTA Version 2.0 is a complete rewrite of the original FESTA code.

The new implementation combines Accelerator Toolbox (AT) and Xsuite to
provide a flexible framework for spin tracking simulations in modern
electron storage rings.
