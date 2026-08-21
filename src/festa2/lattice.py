# lattice.py

from copy import deepcopy
import at


def xs_from_at(ring_at, energy_eV, shaker_index):
    """
    Rotate the AT lattice to start at the shaker,
    create the Xsuite lattice, configure the electron
    reference particle, radiation, and spin tracking.
    """
    
    ring_rot = deepcopy(ring_at)
    ring_rot.rotate(shaker_index)
    ring_xs = at.line_from_lattice(
        ring_rot,
        match_model=True,
    )

    ring_xs.set_particle_ref(
        "electron",
        energy0=energy_eV,
        anomalous_magnetic_moment=0.001159652181643,
    )

    ring_xs.configure_radiation("mean")
    ring_xs.configure_spin("auto")

    return ring_xs, ring_rot
