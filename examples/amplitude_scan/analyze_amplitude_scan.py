#!/usr/bin/env python3

"""
Analyze a FESTA2 shaker-amplitude scan.

The script reads all:
    amp_XXurad/scan_merged.pkl

and produces:
    - 2D map of minimum vertical polarization
    - 2D map of minimum polarization magnitude
    - 2D map of maximum vertical 3-sigma envelope
    - Py_min at 6 GeV versus shaker amplitude
    - |P|_min at 6 GeV versus shaker amplitude
"""

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SAVE_PLOTS = True
SHOW_PLOTS = False
FIGURE_DPI = 200


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def amplitude_from_directory(path):
    """
    Extract shaker amplitude in microradians from a directory name
    like amp_05urad or amp_50urad.
    """

    match = re.match(
        r"amp_(\d+)urad$",
        path.name,
    )

    if match is None:
        raise ValueError(
            f"Cannot extract amplitude from directory name: "
            f"{path.name}"
        )

    return float(match.group(1))


def load_all_scans():
    """
    Load all merged amplitude scans.

    Returns
    -------
    dict
        Common scan axes and 2D observables.
    """

    scan_dirs = sorted(
        Path(".").glob("amp_*urad"),
        key=amplitude_from_directory,
    )

    if not scan_dirs:
        raise RuntimeError(
            "No amp_*urad directories found."
        )

    amplitudes_urad = []

    py_min_all = []
    p_abs_min_all = []

    max_3sigma_all = []
    max_abs_y_mean_all = []
    max_y_std_all = []

    tunes = None
    tune_offsets = None
    energies_GeV = None

    horizontal_tune = None
    vertical_tune = None
    nusp = None

    for scan_dir in scan_dirs:

        merged_file = (
            scan_dir
            / "scan_merged.pkl"
        )

        if not merged_file.exists():
            raise FileNotFoundError(
                f"Missing merged file: "
                f"{merged_file}"
            )

        with merged_file.open("rb") as f:
            data = pickle.load(f)

        scan_results = data["scan_results"]
        scan_input = data["scan_input"]

        amplitude_urad = (
            amplitude_from_directory(
                scan_dir
            )
        )

        amplitudes_urad.append(
            amplitude_urad
        )

        current_tunes = np.asarray(
            scan_input["tunes"]
        )

        current_tune_offsets = np.asarray(
            scan_input["tune_offsets"]
        )

        current_energies_GeV = np.asarray(
            scan_input["energies_GeV"]
        )

        # ----------------------------------------------------------
        # First scan defines the common axes
        # ----------------------------------------------------------

        if tunes is None:

            tunes = current_tunes
            tune_offsets = current_tune_offsets
            energies_GeV = current_energies_GeV

            horizontal_tune = float(
                scan_input["horizontal_tune"]
            )

            vertical_tune = float(
                scan_input["vertical_tune"]
            )

            nusp = float(
                scan_input["nusp"]
            )

        else:

            if not np.allclose(
                current_tunes,
                tunes,
            ):
                raise ValueError(
                    f"Tune grid differs in "
                    f"{scan_dir}"
                )

            if not np.allclose(
                current_tune_offsets,
                tune_offsets,
            ):
                raise ValueError(
                    f"Tune-offset grid differs in "
                    f"{scan_dir}"
                )

        # ----------------------------------------------------------
        # Extract observables
        # ----------------------------------------------------------

        py_min = np.array([
            r["py_min"]
            for r in scan_results
        ])

        p_abs_min = np.array([
            r["p_abs_min"]
            for r in scan_results
        ])

        max_3sigma = np.array([
            r["max_3sigma_envelope"]
            for r in scan_results
        ])

        max_abs_y_mean = np.array([
            r["max_abs_y_mean"]
            for r in scan_results
        ])

        max_y_std = np.array([
            r["max_y_std"]
            for r in scan_results
        ])

        py_min_all.append(
            py_min
        )

        p_abs_min_all.append(
            p_abs_min
        )

        max_3sigma_all.append(
            max_3sigma
        )

        max_abs_y_mean_all.append(
            max_abs_y_mean
        )

        max_y_std_all.append(
            max_y_std
        )

    return {
        "amplitudes_urad":
            np.asarray(
                amplitudes_urad
            ),

        "tunes":
            tunes,

        "tune_offsets":
            tune_offsets,

        "energies_GeV":
            energies_GeV,

        "horizontal_tune":
            horizontal_tune,

        "vertical_tune":
            vertical_tune,

        "nusp":
            nusp,

        "py_min":
            np.asarray(
                py_min_all
            ),

        "p_abs_min":
            np.asarray(
                p_abs_min_all
            ),

        "max_3sigma_envelope":
            np.asarray(
                max_3sigma_all
            ),

        "max_abs_y_mean":
            np.asarray(
                max_abs_y_mean_all
            ),

        "max_y_std":
            np.asarray(
                max_y_std_all
            ),
    }


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def print_summary(q):

    print()
    print("Amplitude scan summary")
    print("----------------------")

    i_6gev = np.argmin(
        np.abs(
            q["tune_offsets"]
        )
    )

    print(
        f"6 GeV index      : {i_6gev}"
    )

    print(
        f"6 GeV tune       : "
        f"{q['tunes'][i_6gev]:.10f}"
    )

    print(
        f"6 GeV energy     : "
        f"{q['energies_GeV'][i_6gev]:.9f} GeV"
    )

    print()

    for i, amp in enumerate(
        q["amplitudes_urad"]
    ):

        j = np.argmin(
            q["p_abs_min"][i]
        )

        print(
            f"{amp:5.1f} urad  "
            f"min |P| = "
            f"{q['p_abs_min'][i, j]:.5f}  "
            f"at tune = "
            f"{q['tunes'][j]:.6f}  "
            f"Py_min@6GeV = "
            f"{q['py_min'][i, i_6gev]:.5f}  "
            f"|P|_min@6GeV = "
            f"{q['p_abs_min'][i, i_6gev]:.5f}  "
            f"max 3sigma = "
            f"{1e3 * q['max_3sigma_envelope'][i, j]:.3f} mm"
        )


# ----------------------------------------------------------------------
# Generic 2D map
# ----------------------------------------------------------------------

def plot_map(
    tunes,
    amplitudes_urad,
    values,
    title,
    colorbar_label,
    filename,
    spin_tune=None,
    vertical_resonance=None,
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    mesh = ax.pcolormesh(
        tunes,
        amplitudes_urad,
        values,
        shading="auto",
    )

    ax.set_xlabel(
        "Shaker tune"
    )

    ax.set_ylabel(
        "Shaker amplitude [urad]"
    )

    ax.set_title(
        title
    )

    if spin_tune is not None:

        ax.axvline(
            spin_tune,
            linestyle="--",
            linewidth=1.5,
            label=rf"$\nu_s = {spin_tune:.4f}$",
        )

    if vertical_resonance is not None:

        ax.axvline(
            vertical_resonance,
            linestyle=":",
            linewidth=1.5,
            label=rf"$1-Q_y = {vertical_resonance:.4f}$",
        )

    if (
        spin_tune is not None
        or vertical_resonance is not None
    ):
        ax.legend()

    colorbar = fig.colorbar(
        mesh,
        ax=ax,
    )

    colorbar.set_label(
        colorbar_label
    )

    fig.tight_layout()

    if SAVE_PLOTS:

        fig.savefig(
            filename,
            dpi=FIGURE_DPI,
        )

    return fig


# ----------------------------------------------------------------------
# 6 GeV curves
# ----------------------------------------------------------------------

def plot_py_at_6gev(q):

    i_6gev = np.argmin(
        np.abs(
            q["tune_offsets"]
        )
    )

    py_at_6gev = (
        q["py_min"][:, i_6gev]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        q["amplitudes_urad"],
        py_at_6gev,
        "o-",
    )

    ax.set_xlabel(
        "Shaker amplitude [urad]"
    )

    ax.set_ylabel(
        r"$P_{y,\min}$ at 6 GeV"
    )

    ax.set_title(
        "Minimum vertical polarization at nominal spin tune"
    )

    ax.grid(True)

    fig.tight_layout()

    if SAVE_PLOTS:

        fig.savefig(
            "py_min_at_6GeV_vs_amplitude.png",
            dpi=FIGURE_DPI,
        )

    return fig


def plot_pabs_at_6gev(q):

    i_6gev = np.argmin(
        np.abs(
            q["tune_offsets"]
        )
    )

    p_abs_at_6gev = (
        q["p_abs_min"][:, i_6gev]
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        q["amplitudes_urad"],
        p_abs_at_6gev,
        "o-",
    )

    ax.set_xlabel(
        "Shaker amplitude [urad]"
    )

    ax.set_ylabel(
        r"$|P|_{\min}$ at 6 GeV"
    )

    ax.set_title(
        "Minimum polarization magnitude at nominal spin tune"
    )

    ax.grid(True)

    fig.tight_layout()

    if SAVE_PLOTS:

        fig.savefig(
            "p_abs_min_at_6GeV_vs_amplitude.png",
            dpi=FIGURE_DPI,
        )

    return fig


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    q = load_all_scans()

    print_summary(q)

    spin_tune = np.mod(
        q["nusp"],
        1.0,
    )

    vertical_resonance = np.mod(
        1.0 - q["vertical_tune"],
        1.0,
    )

    # --------------------------------------------------------------
    # Map 1: minimum vertical polarization
    # --------------------------------------------------------------

    plot_map(
        tunes=q["tunes"],
        amplitudes_urad=q["amplitudes_urad"],
        values=q["py_min"],
        title="Minimum vertical polarization",
        colorbar_label=r"$P_{y,\min}$",
        filename="map_py_min.png",
        spin_tune=spin_tune,
        vertical_resonance=vertical_resonance,
    )

    # --------------------------------------------------------------
    # Map 2: minimum polarization magnitude
    # --------------------------------------------------------------

    plot_map(
        tunes=q["tunes"],
        amplitudes_urad=q["amplitudes_urad"],
        values=q["p_abs_min"],
        title="Minimum polarization magnitude",
        colorbar_label=r"$|P|_{\min}$",
        filename="map_p_abs_min.png",
        spin_tune=spin_tune,
        vertical_resonance=vertical_resonance,
    )

    # --------------------------------------------------------------
    # Map 3: maximum vertical 3-sigma envelope
    # Convert m -> mm
    # --------------------------------------------------------------

    plot_map(
        tunes=q["tunes"],
        amplitudes_urad=q["amplitudes_urad"],
        values=1e3 * q[
            "max_3sigma_envelope"
        ],
        title="Maximum vertical 3-sigma envelope",
        colorbar_label=(
            r"$|\langle y\rangle| + 3\sigma_y$ [mm]"
        ),
        filename="map_vertical_3sigma.png",
        spin_tune=spin_tune,
        vertical_resonance=vertical_resonance,
    )

    # --------------------------------------------------------------
    # Curves at nominal 6 GeV spin tune
    # --------------------------------------------------------------

    plot_py_at_6gev(q)

    plot_pabs_at_6gev(q)

    # --------------------------------------------------------------
    # Show or close
    # --------------------------------------------------------------

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
