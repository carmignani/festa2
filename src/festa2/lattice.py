# lattice.py

import numpy as np
from copy import deepcopy
import at

def xs_from_at(ring_at, energy_eV):
    """
    rotate at lattice to start from the shaker
    create the xsuite lattice
    set the particles to be electrons, with correct anomalous magnetic moment
    rad on
    allow spins
    """ 
    ishaker = np.where(ring_at.get_bool_index('*Shaker*'))
    ring_rot = deepcopy(ring_at)
    ring_rot.rotate(ishaker[0][0])
    ring_xs = at.line_from_lattice(ring_rot, match_model=True)
    ring_xs.set_particle_ref(
        'electron',
        energy0=energy_eV,
        anomalous_magnetic_moment=0.001159652181643
    )
    ring_xs.configure_radiation('mean')
    ring_xs.configure_spin('auto')
    
    return ring_xs, ring_rot
