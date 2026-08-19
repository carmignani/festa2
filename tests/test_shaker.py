import numpy as np
import festa2


def test_zero_amplitude_shaker_does_nothing():
    part = festa2.Particles(n=10, nusp=0.6)

    coord_initial = part.coord.copy()
    spin_initial = part.spin.copy()

    shaker = festa2.Shaker(
        tune=0.62,
        amplitude=0.0,
        phase=0.0,
    )

    festa2.apply_shaker(
        part,
        shaker,
        turn=123,
    )

    assert np.allclose(part.coord, coord_initial)
    assert np.allclose(part.spin, spin_initial)

def test_shaker_preserves_spin_norm():
    part = festa2.Particles(n=10, nusp=0.6)

    shaker = festa2.Shaker(
        tune=0.62,
        amplitude=2e-6,
        phase=0.37,
    )

    festa2.apply_shaker(
        part,
        shaker,
        turn=17,
    )

    spin_norm = np.linalg.norm(part.spin, axis=0)

    assert np.allclose(spin_norm, 1.0)
