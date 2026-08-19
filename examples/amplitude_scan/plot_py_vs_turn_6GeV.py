#!/usr/bin/env python3

"""
Plot vertical polarization versus turn number at the nominal
6 GeV spin resonance for selected shaker amplitudes.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

AMPLITUDES_URAD = [
    1,
    2,
    5,
    10,
    20,
    50,
]

SAVE_PLOT = True
SHOW_PLOT = False
FIGURE_DPI = 200

OUTPUT_FILE = "py_vs_turn_6GeV_selected_amplitudes.png"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for amplitude_urad in AMPLITUDES_URAD:

        scan_dir = Path(
            f"amp_{amplitude_urad:02d}urad"
        )

        merged_file = (
            scan_dir
            / "scan_merged.pkl"
        )

        if not merged_file.exists():
            raise FileNotFoundError(
                f"Missing file: {merged_file}"
            )

        with merged_file.open("rb") as f:
            data = pickle.load(f)

        scan_input = data["scan_input"]
        scan_results = data["scan_results"]

        tune_offsets = np.asarray(
            scan_input["tune_offsets"]
        )

        energies_GeV = np.asarray(
            scan_input["energies_GeV"]
        )

        # Find the simulated frequency corresponding
        # to the nominal 6 GeV spin tune
        i_6gev = np.argmin(
            np.abs(tune_offsets)
        )

        result = scan_results[i_6gev]

        turns = np.asarray(
            result["turns"]
        )

        py = np.asarray(
            result["py"]
        )

        print(
            f"{amplitude_urad:5.1f} urad  "
            f"index={i_6gev:4d}  "
            f"energy={energies_GeV[i_6gev]:.9f} GeV  "
            f"Py_final={py[-1]:.6f}  "
            f"Py_min={np.min(py):.6f}"
        )

        ax.plot(
            turns,
            py,
            label=(
                f"{amplitude_urad:g} urad"
            ),
        )

    ax.set_xlabel(
        "Turn number"
    )

    ax.set_ylabel(
        r"$P_y$"
    )

    ax.set_title(
        "Vertical polarization at nominal spin tune (6 GeV)"
    )

    ax.grid(True)

    ax.legend(
        title="Shaker amplitude"
    )

    fig.tight_layout()

    if SAVE_PLOT:
        fig.savefig(
            OUTPUT_FILE,
            dpi=FIGURE_DPI,
        )

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
