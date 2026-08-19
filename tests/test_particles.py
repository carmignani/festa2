import numpy as np
import festa2


def test_particles_initialization():
    part = festa2.Particles(n=10, nusp=0.6)

    assert part.coord.shape == (6, 10)
    assert part.spin.shape == (3, 10)

    assert np.all(part.coord == 0.0)

    assert np.all(part.sx == 0.0)
    assert np.all(part.sy == 1.0)
    assert np.all(part.sz == 0.0)

    assert part.N == 10
    assert part.nusp == 0.6

def test_init_spin_polarized():
    part = festa2.Particles(n=100, nusp=0.6)

    polarization = part.init_spin_polarized(
        polarization=0.92,
        shuffle=False,
    )

    assert np.isclose(polarization, 0.92)
    assert np.isclose(np.mean(part.sy), 0.92)

    assert np.all(part.sx == 0.0)
    assert np.all(part.sz == 0.0)

    assert np.sum(part.sy == 1.0) == 96
    assert np.sum(part.sy == -1.0) == 4
