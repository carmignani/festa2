"""
Spin-map construction utilities for FESTA2.
"""

from copy import deepcopy

import at
import numpy as np
import xpart as xp
from scipy.spatial.transform import Rotation

def spin_rotation_matrix(p):
    """
    Return the 3×3 one-turn spin rotation matrix.

    The input consists of three particles with identical phase-space
    coordinates and initial spin vectors aligned with the Cartesian
    axes (x, y, z). The returned matrix maps the initial spin vector
    onto the final spin vector after one turn.

    Parameters
    ----------
    p : xtrack.Particles
        Three particles with initial spin vectors along the x, y and z
        directions.

    Returns
    -------
    numpy.ndarray
        A 3×3 orthogonal rotation matrix describing the one-turn spin
        transport.
    """
    return np.array([
        [p.spin_x[0], p.spin_x[1], p.spin_x[2]],
        [p.spin_y[0], p.spin_y[1], p.spin_y[2]],
        [p.spin_z[0], p.spin_z[1], p.spin_z[2]],
    ])

def initialize_p(ring_xs,twiss):
    """
    Create three reference particles for spin tracking.

    The particles are initialized on the closed orbit with identical
    phase-space coordinates and spin vectors aligned with the Cartesian
    axes. They are used to reconstruct the one-turn spin rotation matrix.

    Parameters
    ----------
    ring_xs : ring in xsuite format
    twiss : xtrack.TwissTable
        Twiss table used to initialize the reference particles.

    Returns
    -------
    xtrack.Particles
        Three particles with spin vectors initially along x, y and z.
    """
    p = xp.generate_matched_gaussian_bunch(
        line=ring_xs,
        nemitt_x=0,
        nemitt_y=0,
        sigma_z=0,
        num_particles=3)
    # Initialize orbit in closed orbit
    p.x=twiss.x[0]
    p.px=twiss.px[0]
    p.y=twiss.y[0]
    p.py=twiss.py[0]
    p.zeta=twiss.zeta[0]
    p.delta=twiss.delta[0]
    # Initialize spin along 3 unit vectors
    p.spin_x = [1, 0, 0]
    p.spin_y = [0, 1, 0]
    p.spin_z = [0, 0, 1]
    return(p)


def compute_oam(ring_xs, eps=1e-6):
    """
    Compute the Orbit Angle Matrix (OAM).

    The OAM is the Jacobian of the additional one-turn spin rotation
    with respect to the six phase-space coordinates
    (x, px, y, py, zeta, delta). It is evaluated numerically using
    centered finite differences of the rotation vector.

    Parameters
    ----------
    ring_xs : xtrack.Line
        Xsuite line used for spin tracking.
    eps : float, optional
        Step size used for the centered finite-difference derivative.
        The default is 1e-6.

    Returns
    -------
    numpy.ndarray
        A (3, 6) matrix whose columns give the derivatives of the
        spin rotation vector with respect to
        (x, px, y, py, zeta, delta).
    """
    twiss = ring_xs.twiss(spin=True, radiation_analysis=True, polarization_analysis=True)
    p = initialize_p(ring_xs,twiss)
    OAM = np.zeros((3,6))
    ring_xs.track(p,num_turns=1)
    R = spin_rotation_matrix(p)
    
    for ii in range(6):
        p=initialize_p(ring_xs,twiss)
        if ii==0:
            p.x+=eps
        if ii==1:
            p.px+=eps
        if ii==2:
            p.y+=eps
        if ii==3:
            p.py+=eps
        if ii==4:
            p.zeta+=eps
        if ii==5:
            p.delta+=eps
        ring_xs.track(p,num_turns=1)
        Rplus = spin_rotation_matrix(p)
        
        p=initialize_p(ring_xs,twiss)
        if ii==0:
            p.x-=eps
        if ii==1:
            p.px-=eps
        if ii==2:
            p.y-=eps
        if ii==3:
            p.py-=eps
        if ii==4:
            p.zeta-=eps
        if ii==5:
            p.delta-=eps
        ring_xs.track(p,num_turns=1)
        Rminus = spin_rotation_matrix(p)
        
        dR_plus  = Rplus  @ R.T
        dR_minus = Rminus @ R.T
        omega_plus  = Rotation.from_matrix(dR_plus).as_rotvec()
        omega_minus = Rotation.from_matrix(dR_minus).as_rotvec()
        OAM[:,ii] = (omega_plus - omega_minus)/(2*eps)
        
    # columns 4 and 5 have to be switched because of different AT-XS coord system
    # Xsuite: x, px, y, py, zeta, delta
    # AT:     x, px, y, py, delta, ct
    OAM_at = OAM[:, [0, 1, 2, 3, 5, 4]]
    return OAM_at, R

