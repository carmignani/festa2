"""
Spin-orbit tracking routines for FESTA2.
"""

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from .shaker import apply_shaker

def track_1_turn(frr, part, OAM, R0, closed_orbit, seed=None):
    # Orbit deviation at the beginning of the turn
    co = closed_orbit.reshape(6, 1)
    dx = part.coord - co
    # Spin tracking
    omega = OAM @ dx
    Rextra = Rotation.from_rotvec(omega.T).as_matrix()
    R = Rextra @ R0
    spin_new = np.zeros_like(part.spin)
    for n in range(part.N):
        spin_new[:, n] = R[n] @ part.spin[:, n]
    part.spin = spin_new
    # Orbit tracking
    r_out, info, loss = frr.track(part.coord, 1, seed=seed)
    part.coord = r_out[:, :, 0, 0]

    return part

def simulate_one_frequency(
    tune,
    frr,
    part_initial,
    shaker_initial,
    OAM,
    R0,
    closed_orbit,
    nturns,
    save_every=1,
    average_blocks=False,
    seed=None,
):
    """
    Track a particle ensemble at one shaker frequency.

    The initial particle distribution is copied before tracking, allowing
    the same initial ensemble to be reused for every frequency in a scan.

    Parameters
    ----------
    tune : float
        Shaker excitation tune in oscillations per turn.
    frr
        AT ring or tracking object.
    part_initial : Particle
        Initial particle ensemble. It must contain orbital coordinates
        and spin vectors. This object is not modified.
    shaker_initial : Shaker
        Initial shaker configuration. This object is not modified.
    OAM : numpy.ndarray
        Orbit Angle Matrix with shape (3, 6), using the AT coordinate
        ordering.
    R0 : numpy.ndarray
        Closed-orbit one-turn spin rotation matrix with shape (3, 3).
    closed_orbit : numpy.ndarray
        Closed-orbit coordinates with shape (6,).
    nturns : int
        Total number of turns.
    save_every : int, optional
        Number of turns between saved samples. Default is 1.
    average_blocks : bool, optional
        If False, save instantaneous quantities every `save_every` turns.
        If True, save the mean quantities over consecutive blocks of
        `save_every` turns. Default is False.
    seed : int or None, optional
        Random seed passed to the first AT tracking call.

    Returns
    -------
    result : dict
        Dictionary containing polarization and vertical orbital quantities
        versus turn number, together with summary values and the final
        particle ensemble.
    """

    if nturns < 1:
        raise ValueError("nturns must be at least 1")

    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    # ------------------------------------------------------------------
    # Independent copies for this frequency
    # ------------------------------------------------------------------

    part = deepcopy(part_initial)
    shaker = deepcopy(shaker_initial)

    shaker.tune = tune

    # ------------------------------------------------------------------
    # Histories
    # ------------------------------------------------------------------

    turns = []

    polarization_history = []

    y_mean_history = []
    y_std_history = []

    # ------------------------------------------------------------------
    # Initial values at turn 0
    # ------------------------------------------------------------------

    initial_polarization = np.mean(
        part.spin,
        axis=1,
    )

    initial_y_mean = np.mean(part.y)
    initial_y_std = np.std(part.y)

    if not average_blocks:

        turns.append(0)

        polarization_history.append(
            initial_polarization
        )

        y_mean_history.append(
            initial_y_mean
        )

        y_std_history.append(
            initial_y_std
        )

    # ------------------------------------------------------------------
    # Variables used only for block averaging
    # ------------------------------------------------------------------

    block_polarization_sum = np.zeros(3)

    block_y_mean_sum = 0.0
    block_y_std_sum = 0.0

    block_count = 0

    # ------------------------------------------------------------------
    # Tracking loop
    # ------------------------------------------------------------------

    for turn in range(nturns):

        part = apply_shaker(
            part=part,
            shaker=shaker,
            turn=turn,
        )

        part = track_1_turn(
            frr=frr,
            part=part,
            OAM=OAM,
            R0=R0,
            closed_orbit=closed_orbit,
            seed=seed if turn == 0 else None,
        )

        # --------------------------------------------------------------
        # Beam quantities after this turn
        # --------------------------------------------------------------

        polarization = np.mean(
            part.spin,
            axis=1,
        )

        y_mean = np.mean(part.y)
        y_std = np.std(part.y)

        # --------------------------------------------------------------
        # Block averaging
        # --------------------------------------------------------------

        if average_blocks:

            block_polarization_sum += polarization

            block_y_mean_sum += y_mean
            block_y_std_sum += y_std

            block_count += 1

            if block_count == save_every:

                polarization_history.append(
                    block_polarization_sum
                    / block_count
                )

                y_mean_history.append(
                    block_y_mean_sum
                    / block_count
                )

                y_std_history.append(
                    block_y_std_sum
                    / block_count
                )

                # End turn of this averaging block
                turns.append(turn + 1)

                block_polarization_sum[:] = 0.0

                block_y_mean_sum = 0.0
                block_y_std_sum = 0.0

                block_count = 0

        # --------------------------------------------------------------
        # Instantaneous saving
        # --------------------------------------------------------------

        elif (turn + 1) % save_every == 0:

            polarization_history.append(
                polarization
            )

            y_mean_history.append(
                y_mean
            )

            y_std_history.append(
                y_std
            )

            turns.append(
                turn + 1
            )

    # ------------------------------------------------------------------
    # Keep the final incomplete averaging block
    # ------------------------------------------------------------------

    if average_blocks and block_count > 0:

        polarization_history.append(
            block_polarization_sum
            / block_count
        )

        y_mean_history.append(
            block_y_mean_sum
            / block_count
        )

        y_std_history.append(
            block_y_std_sum
            / block_count
        )

        turns.append(
            nturns
        )

    # ------------------------------------------------------------------
    # Convert histories to arrays
    # ------------------------------------------------------------------

    turns = np.asarray(
        turns,
        dtype=int,
    )

    polarization_history = np.asarray(
        polarization_history,
        dtype=float,
    ).T

    y_mean_history = np.asarray(
        y_mean_history,
        dtype=float,
    )

    y_std_history = np.asarray(
        y_std_history,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Polarization components
    # ------------------------------------------------------------------

    px = polarization_history[0, :]
    py = polarization_history[1, :]
    pz = polarization_history[2, :]

    # ------------------------------------------------------------------
    # Polarization magnitude
    # ------------------------------------------------------------------

    p_abs = np.linalg.norm(
        polarization_history,
        axis=0,
    )

    # ------------------------------------------------------------------
    # Summary indices
    # ------------------------------------------------------------------

    i_min = np.argmin(py)

    i_p_abs_min = np.argmin(
        p_abs
    )

    # ------------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------------

    return {
        "tune": tune,

        "turns": turns,

        # Polarization versus turn
        "polarization": polarization_history,
        "px": px,
        "py": py,
        "pz": pz,
        "p_abs": p_abs,

        # Vertical orbit versus turn
        "y_mean": y_mean_history,
        "y_std": y_std_history,

        # Initial polarization
        "polarization_initial":
            initial_polarization,

        "py_initial":
            initial_polarization[1],

        "p_abs_initial":
            np.linalg.norm(
                initial_polarization
            ),

        # Initial vertical orbit
        "y_mean_initial":
            initial_y_mean,

        "y_std_initial":
            initial_y_std,

        # Final polarization
        "polarization_final":
            polarization_history[:, -1],

        "py_final":
            py[-1],

        "p_abs_final":
            p_abs[-1],

        # Final vertical orbit
        "y_mean_final":
            y_mean_history[-1],

        "y_std_final":
            y_std_history[-1],

        # Minimum vertical polarization
        "py_min":
            py[i_min],

        "turn_at_py_min":
            turns[i_min],

        # Minimum polarization magnitude
        "p_abs_min":
            p_abs[i_p_abs_min],

        "turn_at_p_abs_min":
            turns[i_p_abs_min],

        # Useful vertical-orbit summary quantities
        "max_abs_y_mean":
            np.max(
                np.abs(y_mean_history)
            ),

        "max_y_std":
            np.max(
                y_std_history
            ),

        "max_3sigma_envelope":
            np.max(
                np.abs(y_mean_history)
                + 3.0 * y_std_history
            ),

        # Final ensemble
        "part_final": part,

        # Saving configuration
        "save_every":
            save_every,

        "average_blocks":
            average_blocks,
    }