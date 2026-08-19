#!/usr/bin/env python3

import pickle

import matplotlib.pyplot as plt
import numpy as np


with open("depolarization_tau_results.pkl", "rb") as f:
    data = pickle.load(f)

amplitudes = np.asarray(
    data["amplitudes_urad"]
)

taus = np.asarray(
    data["taus"]
)

tau_errors = np.asarray(
    data["tau_errors"]
)


# Log-log fit
log_A = np.log(amplitudes)
log_tau = np.log(taus)

slope, intercept = np.polyfit(
    log_A,
    log_tau,
    1,
)

n = -slope
C = np.exp(intercept)

tau_fit = (
    C * amplitudes**(-n)
)

print(f"C = {C:.6e}")
print(f"n = {n:.6f}")


plt.figure(figsize=(9, 6))

plt.errorbar(
    amplitudes,
    taus,
    yerr=tau_errors,
    marker="o",
    linestyle="none",
    capsize=3,
    label="Simulation",
)

plt.plot(
    amplitudes,
    tau_fit,
    "--",
    label=rf"$\tau = {C:.2e} A^{{-{n:.3f}}}$",
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Shaker amplitude [urad]")
plt.ylabel(r"Depolarization time $\tau$ [turns]")

plt.grid(
    True,
    which="both",
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "tau_powerlaw_fit.png",
    dpi=200,
)
