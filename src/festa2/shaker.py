"""
Shaker excitation for FESTA2.
"""

import numpy as np
from scipy.spatial.transform import Rotation

class Shaker:
    """
    Vertical AC dipole (shaker).

    Parameters
    ----------
    tune : float
        Excitation tune (oscillations per turn).
    amplitude : float
        Peak kick angle applied to the orbit [rad].
    phase : float, optional
        Initial phase [rad]. Default is 0.
    """

    def __init__(self, tune, amplitude, phase=0.0):
        self.tune = tune
        self.amplitude = amplitude
        self.phase = phase

def apply_shaker(part, shaker, turn):
    """
    Apply one kick from a vertical AC dipole.

    The shaker applies
      - a vertical kick to the orbital motion (py);
      - the corresponding spin rotation.

    Parameters
    ----------
    part : Particle
        Particle object containing coord and spin.
    shaker : Shaker
        Shaker parameters.
    turn : int
        Turn number.
    Ggamma : float
        Spin precession factor (typically G*gamma).

    Returns
    -------
    part
        Updated particle object.
    """

    # Orbit kick (vertical)
    theta = shaker.amplitude * np.sin(
        2 * np.pi * shaker.tune * turn + shaker.phase
    )

    part.coord[3, :] += theta
    
    # Spin rotation angle
    nusp=part.nusp
    psi = (1.0 + nusp) * theta

    # Rotation around the horizontal x axis
    omega = np.zeros((part.N, 3))
    omega[:, 0] = psi
    # omega[:, 0] = 0

    Rkick = Rotation.from_rotvec(omega).as_matrix()

    spin_new = np.zeros_like(part.spin)

    for n in range(part.N):
        spin_new[:, n] = Rkick[n] @ part.spin[:, n]

    part.spin = spin_new

    return part
