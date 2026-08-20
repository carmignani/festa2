#!/usr/bin/env python3

"""
Prepare the input data for a FESTA2 frequency scan.

This script:
- loads the AT lattice;
- creates the corresponding Xsuite line;
- computes the OAM and the closed-orbit spin matrix;
- creates an equilibrium particle distribution;
- initializes the beam polarization;
- defines the shaker-frequency scan;
- computes the equivalent electron energies;
- writes all data to scan_input.pkl.
"""

import pickle
import time
from pathlib import Path

import at
import numpy as np
from scipy.constants import physical_constants

import os
import subprocess
from datetime import datetime

import festa2


# ----------------------------------------------------------------------
# User configuration
# ----------------------------------------------------------------------

LATTICE_FILE = "betamodel.mat"
LATTICE_KEY = "betamodel"

OUTPUT_FILE = "scan_input.pkl"

NPARTICLES = 400
INITIAL_POLARIZATION = 0.92
RANDOM_SEED = 12345

N_EQUILIBRATION_TURNS = 30_000

N_TURNS = 1_000_000
SAVE_EVERY = 100
AVERAGE_BLOCKS = False

SHAKER_AMPLITUDE = 2e-6
SHAKER_PHASE = 0.0

TUNE_OFFSET_MIN = -0.060
TUNE_OFFSET_MAX = 0.060
N_FREQUENCIES = 1001

TRACKING_SEED = 12345

def main() -> None:
    start = time.perf_counter()

    lattice_path = Path(LATTICE_FILE)

    if not lattice_path.exists():
        raise FileNotFoundError(
            f"Lattice file not found: {lattice_path.resolve()}"
        )

    # ------------------------------------------------------------------
    # Load and prepare the lattice
    # ------------------------------------------------------------------

    print("Loading lattice...")

    ring_at = at.load_lattice(
        LATTICE_FILE,
        key=LATTICE_KEY,
    )

    energy_eV = ring_at.energy
    
    ring_xs, ring_at_rot = festa2.xs_from_at(
        ring_at,
        energy_eV,
    )

    # ------------------------------------------------------------------
    # Machine tunes
    # ------------------------------------------------------------------
    
    machine_tunes = ring_at_rot.get_tune(method="linopt")
    
    qx = float(machine_tunes[0])
    qy = float(machine_tunes[1])
    
    print(f"Horizontal tune Qx : {qx:.10f}")
    print(f"Vertical tune Qy   : {qy:.10f}")
    
    # ------------------------------------------------------------------
    # Find frequencies
    # ------------------------------------------------------------------

    rf_frequency_Hz = ring_at.rf_frequency
    harmonic_number = 992
    revolution_frequency_Hz = rf_frequency_Hz / harmonic_number

    # ------------------------------------------------------------------
    # Compute spin maps
    # ------------------------------------------------------------------

    print("Computing OAM and R0...")

    OAM, R0 = festa2.compute_oam(ring_xs)

    # ------------------------------------------------------------------
    # Prepare AT fast ring and closed orbit
    # ------------------------------------------------------------------

    print("Creating AT fast ring...")

    _, frr = at.fast_ring(ring_at_rot)
    closed_orbit = frr.find_orbit6()[0]

    # ------------------------------------------------------------------
    # Reference spin tune
    # ------------------------------------------------------------------

    anomalous_magnetic_moment = float(
        ring_xs.particle_ref.anomalous_magnetic_moment[0]
    )

    gamma0 = float(
        ring_xs.particle_ref.gamma0[0]
    )

    nusp = gamma0 * anomalous_magnetic_moment

    # ------------------------------------------------------------------
    # Prepare equilibrium orbital distribution
    # ------------------------------------------------------------------

    print(
        f"Equilibrating {NPARTICLES} particles for "
        f"{N_EQUILIBRATION_TURNS} turns..."
    )

    part = festa2.Particles(
        n=NPARTICLES,
        nusp=nusp,
    )

    part.coord[:] = closed_orbit[:, None]

    for _ in range(N_EQUILIBRATION_TURNS):
        r_out, info, loss = frr.track(part.coord, 1)
        part.coord = r_out[:, :, 0, 0]

    # ------------------------------------------------------------------
    # Initialize polarization
    # ------------------------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    actual_polarization = part.init_spin_polarized(
        polarization=INITIAL_POLARIZATION,
        rng=rng,
        shuffle=True,
    )

    print(
        f"Requested polarization: {INITIAL_POLARIZATION}"
    )
    print(
        f"Actual polarization:    {actual_polarization}"
    )

    # ------------------------------------------------------------------
    # Frequency scan
    # ------------------------------------------------------------------

    tune_offsets = np.linspace(
        TUNE_OFFSET_MIN,
        TUNE_OFFSET_MAX,
        N_FREQUENCIES,
    )

    spin_tunes_unwrapped = nusp + tune_offsets

    # The shaker tune used turn-by-turn is reduced modulo 1.
    tunes = np.mod(spin_tunes_unwrapped, 1.0)

    # Random seed used for tracking at each frequency
    tracking_seeds = np.full(
        len(tunes),
        TRACKING_SEED,
        dtype=object,
    )

    # ------------------------------------------------------------------
    # Equivalent energies from nu_s = a * gamma
    # ------------------------------------------------------------------

    electron_rest_energy_GeV = (
        physical_constants[
            "electron mass energy equivalent in MeV"
        ][0]
        / 1000.0
    )

    gammas = (
        spin_tunes_unwrapped
        / anomalous_magnetic_moment
    )

    energies_GeV = (
        gammas * electron_rest_energy_GeV
    )

    reference_energy_GeV = (
        gamma0 * electron_rest_energy_GeV
    )

    energy_offsets_GeV = (
        energies_GeV - reference_energy_GeV
    )

    relative_energy_offsets = (
        energy_offsets_GeV / reference_energy_GeV
    )

    shaker_frequency_Hz = tunes * revolution_frequency_Hz
    shaker_frequency_kHz = shaker_frequency_Hz / 1e3

    frequency_offset_Hz = tune_offsets * revolution_frequency_Hz
    frequency_offset_kHz = frequency_offset_Hz / 1e3
    
    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    
    metadata = {
        "creation_time": datetime.now().isoformat(timespec="seconds"),
        "hostname": os.uname().nodename,
    }

    package_dir = Path(festa2.__file__).resolve().parent
    repo_dir = package_dir.parent.parent
    
    try:
        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
        ).strip()
    except Exception:
        metadata["git_commit"] = None

    # ------------------------------------------------------------------
    # Save configuration
    # ------------------------------------------------------------------

    scan_input = {
        # Frequency axes
        "tunes": tunes,
        "tune_offsets": tune_offsets,
        "spin_tunes_unwrapped": spin_tunes_unwrapped,

        # Seed
        "tracking_seeds": tracking_seeds,

        # Machine frequencies
        "rf_frequency_Hz": rf_frequency_Hz,
        "harmonic_number": harmonic_number,
        "revolution_frequency_Hz": revolution_frequency_Hz,
        "shaker_frequency_Hz": shaker_frequency_Hz,
        "shaker_frequency_kHz": shaker_frequency_kHz,
        "frequency_offset_Hz": frequency_offset_Hz,
        "frequency_offset_kHz": frequency_offset_kHz,

        # Machine tunes
        "horizontal_tune": qx,
        "vertical_tune": qy,

        # Equivalent energy axes
        "energies_GeV": energies_GeV,
        "energy_offsets_GeV": energy_offsets_GeV,
        "relative_energy_offsets": relative_energy_offsets,
        "reference_energy_GeV": reference_energy_GeV,

        # Beam and spin model
        "part_initial": part,
        "OAM": OAM,
        "R0": R0,
        "closed_orbit": closed_orbit,
        "nusp": nusp,
        "gamma0": gamma0,
        "anomalous_magnetic_moment":
            anomalous_magnetic_moment,

        # Tracking configuration
        "nturns": N_TURNS,
        "save_every": SAVE_EVERY,
        "average_blocks": AVERAGE_BLOCKS,

        # Shaker configuration
        "shaker_amplitude": SHAKER_AMPLITUDE,
        "shaker_phase": SHAKER_PHASE,

        # Beam initialization
        "nparticles": NPARTICLES,
        "initial_polarization":
            INITIAL_POLARIZATION,
        "actual_initial_polarization":
            actual_polarization,
        "random_seed": RANDOM_SEED,
        "n_equilibration_turns":
            N_EQUILIBRATION_TURNS,

        # Lattice information
        "lattice_file": LATTICE_FILE,
        "lattice_key": LATTICE_KEY,

        "metadata": metadata,
    }

    output_path = Path(OUTPUT_FILE)
    temporary_path = output_path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        pickle.dump(
            scan_input,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    temporary_path.replace(output_path)

    elapsed = time.perf_counter() - start

    print()
    print(f"Number of frequencies : {len(tunes)}")
    print(f"Tune range            : "
          f"{tunes.min():.8f} to {tunes.max():.8f}")
    print(f"Offset range          : "
          f"{tune_offsets[0]:+.6f} to "
          f"{tune_offsets[-1]:+.6f}")
    print(f"Energy range [GeV]    : "
          f"{energies_GeV[0]:.6f} to "
          f"{energies_GeV[-1]:.6f}")
    print(f"Output file           : {output_path}")
    print(f"Preparation time      : {elapsed:.3f} s")


if __name__ == "__main__":
    main()
