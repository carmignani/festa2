#!/usr/bin/env python3

"""
Fit the characteristic depolarization time at 6 GeV
for all shaker amplitudes.

Model:

    Py(N) = 0.92 * exp(-N / tau)

The fit is optionally restricted to points with Py > PY_FIT_MIN,
to avoid fitting the late-time oscillations around zero.
"""

import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

SAVE_PLOTS = True
SHOW_PLOTS = False
FIGURE_DPI = 200

INITIAL_POLARIZATION = 0.92

# Only fit points above this Py value.
# Set to None to fit the full curve.
PY_FIT_MIN = 0.10

# Selected amplitudes to show in the detailed fit plot
PLOT_AMPLITUDES_URAD = [
    1,
    2,
    5,
    10,
    20,
    50,
]


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


# ----------------------------------------------------------------------
# Depolarization model
# ----------------------------------------------------------------------

def depolarization_model(turns, tau):
    """
    Exponential depolarization model:

        Py(turn) = P0 * exp(-turn / tau)

    with fixed P0 = INITIAL_POLARIZATION.
    """

    return (
        INITIAL_POLARIZATION
        * np.exp(-turns / tau)
    )


# ----------------------------------------------------------------------
# Fit one curve
# ----------------------------------------------------------------------

def fit_one_curve(turns, py):
    """
    Fit Py(turn) with:

        Py(turn) = 0.92 * exp(-turn / tau)

    Returns
    -------
    dict
        Fit results and diagnostic quantities.
    """

    turns = np.asarray(
        turns,
        dtype=float,
    )

    py = np.asarray(
        py,
        dtype=float,
    )

    # --------------------------------------------------------------
    # Select fit region
    # --------------------------------------------------------------

    if PY_FIT_MIN is None:

        mask = np.ones(
            py.shape,
            dtype=bool,
        )

    else:

        mask = (
            py > PY_FIT_MIN
        )

    turns_fit = turns[mask]
    py_fit = py[mask]

    if len(turns_fit) < 3:
        raise RuntimeError(
            "Too few points available for exponential fit."
        )

    # --------------------------------------------------------------
    # Initial guess
    # --------------------------------------------------------------

    # Estimate tau from the last fitted point if possible
    last_py = py_fit[-1]
    last_turn = turns_fit[-1]

    if (
        last_py > 0
        and last_py < INITIAL_POLARIZATION
        and last_turn > 0
    ):

        tau_guess = (
            -last_turn
            / np.log(
                last_py
                / INITIAL_POLARIZATION
            )
        )

    else:

        tau_guess = (
            0.5 * turns[-1]
        )

    tau_guess = max(
        tau_guess,
        1.0,
    )

    # --------------------------------------------------------------
    # Fit
    # --------------------------------------------------------------

    popt, pcov = curve_fit(
        depolarization_model,
        turns_fit,
        py_fit,
        p0=[tau_guess],
        bounds=(
            [1.0],
            [1e10],
        ),
        maxfev=50000,
    )

    tau = popt[0]

    tau_err = np.sqrt(
        np.diag(pcov)
    )[0]

    # --------------------------------------------------------------
    # Evaluate fit
    # --------------------------------------------------------------

    fitted_full = depolarization_model(
        turns,
        tau,
    )

    fitted_region = depolarization_model(
        turns_fit,
        tau,
    )

    residuals = (
        py_fit
        - fitted_region
    )

    rms_residual = np.sqrt(
        np.mean(
            residuals**2
        )
    )

    return {
        "tau":
            tau,

        "tau_err":
            tau_err,

        "turns_fit":
            turns_fit,

        "py_fit":
            py_fit,

        "fitted_full":
            fitted_full,

        "rms_residual":
            rms_residual,

        "n_fit_points":
            len(turns_fit),

        "last_fit_turn":
            turns_fit[-1],
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    scan_dirs = sorted(
        Path(".").glob("amp_*urad"),
        key=amplitude_from_directory,
    )

    if not scan_dirs:
        raise RuntimeError(
            "No amp_*urad directories found."
        )

    amplitudes = []
    taus = []
    tau_errors = []
    rms_residuals = []

    # --------------------------------------------------------------
    # Figure with selected Py curves and fits
    # --------------------------------------------------------------

    fig_curves, ax_curves = plt.subplots(
        figsize=(11, 7)
    )

    print()
    print("Depolarization fits at 6 GeV")
    print("----------------------------")

    for scan_dir in scan_dirs:

        amplitude = (
            amplitude_from_directory(
                scan_dir
            )
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

        # ----------------------------------------------------------
        # Nominal 6 GeV point:
        # tune_offset = 0
        # ----------------------------------------------------------

        i_6gev = np.argmin(
            np.abs(
                tune_offsets
            )
        )

        result = (
            scan_results[
                i_6gev
            ]
        )

        turns = np.asarray(
            result["turns"],
            dtype=float,
        )

        py = np.asarray(
            result["py"],
            dtype=float,
        )

        # ----------------------------------------------------------
        # Fit
        # ----------------------------------------------------------

        fit = fit_one_curve(
            turns,
            py,
        )

        amplitudes.append(
            amplitude
        )

        taus.append(
            fit["tau"]
        )

        tau_errors.append(
            fit["tau_err"]
        )

        rms_residuals.append(
            fit["rms_residual"]
        )

        print(
            f"{amplitude:5.1f} urad  "
            f"tau = "
            f"{fit['tau']:12.1f} +/- "
            f"{fit['tau_err']:10.1f} turns  "
            f"Nfit = "
            f"{fit['n_fit_points']:5d}  "
            f"last turn = "
            f"{fit['last_fit_turn']:10.0f}  "
            f"RMS = "
            f"{fit['rms_residual']:.3e}"
        )

        # ----------------------------------------------------------
        # Plot selected amplitudes
        # ----------------------------------------------------------

        if amplitude in PLOT_AMPLITUDES_URAD:

            ax_curves.plot(
                turns,
                py,
                linewidth=1,
                label=(
                    f"{amplitude:g} urad data"
                ),
            )

            ax_curves.plot(
                turns,
                fit["fitted_full"],
                "--",
                linewidth=1.5,
                label=(
                    f"{amplitude:g} urad fit"
                ),
            )

    amplitudes = np.asarray(
        amplitudes
    )

    taus = np.asarray(
        taus
    )

    tau_errors = np.asarray(
        tau_errors
    )

    rms_residuals = np.asarray(
        rms_residuals
    )

    # --------------------------------------------------------------
    # Finish selected-curves plot
    # --------------------------------------------------------------

    ax_curves.set_xlabel(
        "Turn number"
    )

    ax_curves.set_ylabel(
        r"$P_y$"
    )

    ax_curves.set_title(
        "Depolarization at nominal spin tune (6 GeV)"
    )

    ax_curves.grid(True)

    ax_curves.legend(
        ncol=2,
        fontsize=9,
    )

    fig_curves.tight_layout()

    if SAVE_PLOTS:

        fig_curves.savefig(
            "depolarization_fits_6GeV.png",
            dpi=FIGURE_DPI,
        )

    # --------------------------------------------------------------
    # Tau versus shaker amplitude
    # --------------------------------------------------------------

    fig_tau, ax_tau = plt.subplots(
        figsize=(9, 6)
    )

    ax_tau.errorbar(
        amplitudes,
        taus,
        yerr=tau_errors,
        marker="o",
        linestyle="-",
        capsize=3,
    )

    ax_tau.set_xlabel(
        "Shaker amplitude [urad]"
    )

    ax_tau.set_ylabel(
        r"Depolarization time $\tau$ [turns]"
    )

    ax_tau.set_title(
        "Characteristic depolarization time at 6 GeV"
    )

    ax_tau.grid(True)

    fig_tau.tight_layout()

    if SAVE_PLOTS:

        fig_tau.savefig(
            "depolarization_tau_vs_amplitude.png",
            dpi=FIGURE_DPI,
        )

    # --------------------------------------------------------------
    # Log-log tau versus amplitude
    # --------------------------------------------------------------

    fig_log, ax_log = plt.subplots(
        figsize=(9, 6)
    )

    ax_log.errorbar(
        amplitudes,
        taus,
        yerr=tau_errors,
        marker="o",
        linestyle="none",
        capsize=3,
    )

    ax_log.set_xscale(
        "log"
    )

    ax_log.set_yscale(
        "log"
    )

    ax_log.set_xlabel(
        "Shaker amplitude [urad]"
    )

    ax_log.set_ylabel(
        r"Depolarization time $\tau$ [turns]"
    )

    ax_log.set_title(
        "Depolarization time scaling at 6 GeV"
    )

    ax_log.grid(
        True,
        which="both",
    )

    fig_log.tight_layout()

    if SAVE_PLOTS:

        fig_log.savefig(
            "depolarization_tau_vs_amplitude_loglog.png",
            dpi=FIGURE_DPI,
        )

    fit_results = {
        "amplitudes_urad": amplitudes,
        "taus": taus,
        "tau_errors": tau_errors,
        "rms_residuals": rms_residuals,
        "initial_polarization": INITIAL_POLARIZATION,
        "py_fit_min": PY_FIT_MIN,
    }
    
    with open("depolarization_tau_results.pkl", "wb") as f:
        pickle.dump(
            fit_results,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        
        print()
        print("Saved fit results to depolarization_tau_results.pkl")

    # --------------------------------------------------------------
    # RMS residual versus amplitude
    # --------------------------------------------------------------

    fig_rms, ax_rms = plt.subplots(
        figsize=(9, 6)
    )

    ax_rms.plot(
        amplitudes,
        rms_residuals,
        "o-",
    )

    ax_rms.set_xlabel(
        "Shaker amplitude [urad]"
    )

    ax_rms.set_ylabel(
        "Fit RMS residual"
    )

    ax_rms.set_title(
        "Quality of exponential depolarization fit"
    )

    ax_rms.grid(True)

    fig_rms.tight_layout()

    if SAVE_PLOTS:

        fig_rms.savefig(
            "depolarization_fit_rms_vs_amplitude.png",
            dpi=FIGURE_DPI,
        )


        
    # --------------------------------------------------------------
    # Show or close
    # --------------------------------------------------------------

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
