"""
Frequency-scan utilities for FESTA2.
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .tracking import simulate_one_frequency

def _simulate_frequency_worker(
    tune,
    frr,
    part_initial,
    shaker_initial,
    OAM,
    R0,
    closed_orbit,
    nturns,
    save_every,
    average_blocks,
):
    """
    Worker function used by the parallel frequency scan.
    """
    result = simulate_one_frequency(
        tune=tune,
        frr=frr,
        part_initial=part_initial,
        shaker_initial=shaker_initial,
        OAM=OAM,
        R0=R0,
        closed_orbit=closed_orbit,
        nturns=nturns,
        save_every=save_every,
        average_blocks=average_blocks,
    )

    # Avoid transferring the complete final particle ensemble
    # from every process unless it is really needed.
    result.pop("part_final", None)

    return result

def scan_frequencies(
    tunes,
    frr,
    part_initial,
    shaker_initial,
    OAM,
    R0,
    closed_orbit,
    nturns,
    save_every=1,
    average_blocks=False,
    n_workers=None,
):
    """
    Run a parallel scan over shaker frequencies.

    Every frequency starts from an independent copy of the same initial
    particle ensemble.

    Parameters
    ----------
    tunes : array_like
        Shaker tunes to simulate.
    frr
        AT fast-ring tracking object.
    part_initial : Particles
        Common initial particle ensemble.
    shaker_initial : Shaker
        Common shaker configuration. Its tune is replaced for each run.
    OAM : numpy.ndarray
        Orbit Angle Matrix with shape (3, 6), in AT coordinate ordering.
    R0 : numpy.ndarray
        Closed-orbit one-turn spin rotation matrix with shape (3, 3).
    closed_orbit : numpy.ndarray
        AT closed-orbit coordinates with shape (6,).
    nturns : int
        Number of tracking turns per frequency.
    save_every : int, optional
        Save polarization every `save_every` turns.
    average_blocks : bool, optional
        If True, save the average polarization over each block.
    n_workers : int or None, optional
        Number of parallel worker processes. If None, Python chooses
        according to the available CPUs.

    Returns
    -------
    list of dict
        Results in the same order as `tunes`.
    """
    tunes = np.asarray(tunes, dtype=float)

    if tunes.ndim != 1:
        raise ValueError("tunes must be a one-dimensional array")

    if n_workers is not None and n_workers < 1:
        raise ValueError("n_workers must be at least 1")

    # Useful for debugging and validation
    if n_workers == 1:
        return [
            _simulate_frequency_worker(
                tune,
                frr,
                part_initial,
                shaker_initial,
                OAM,
                R0,
                closed_orbit,
                nturns,
                save_every,
                average_blocks,
            )
            for tune in tunes
        ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _simulate_frequency_worker,
                tune,
                frr,
                part_initial,
                shaker_initial,
                OAM,
                R0,
                closed_orbit,
                nturns,
                save_every,
                average_blocks,
            )
            for tune in tunes
        ]

        # Calling result() in submission order preserves tune ordering.
        results = [future.result() for future in futures]

    return results
