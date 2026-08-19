import numpy as np
import festa2

from festa2.tracking import track_1_turn


class IdentityRing:
    """Dummy ring that leaves orbital coordinates unchanged."""

    def track(self, coord, nturns, seed=None):
        r_out = coord[:, :, None, None].copy()
        return r_out, None, None


def test_identity_spin_map_preserves_spin():
    part = festa2.Particles(n=10, nusp=0.6)

    spin_initial = part.spin.copy()
    coord_initial = part.coord.copy()

    frr = IdentityRing()

    OAM = np.zeros((3, 6))
    R0 = np.eye(3)
    closed_orbit = np.zeros(6)

    track_1_turn(
        frr=frr,
        part=part,
        OAM=OAM,
        R0=R0,
        closed_orbit=closed_orbit,
    )

    assert np.allclose(part.spin, spin_initial)
    assert np.allclose(part.coord, coord_initial)

def test_simulate_one_frequency_identity_case():
    part = festa2.Particles(n=10, nusp=0.6)

    shaker = festa2.Shaker(
        tune=0.62,
        amplitude=0.0,
        phase=0.0,
    )

    frr = IdentityRing()

    OAM = np.zeros((3, 6))
    R0 = np.eye(3)
    closed_orbit = np.zeros(6)

    result = festa2.simulate_one_frequency(
        tune=0.62,
        frr=frr,
        part_initial=part,
        shaker_initial=shaker,
        OAM=OAM,
        R0=R0,
        closed_orbit=closed_orbit,
        nturns=10,
        save_every=2,
        average_blocks=False,
    )

    # turn 0 + turns 2, 4, 6, 8, 10
    assert np.array_equal(
        result["turns"],
        np.array([0, 2, 4, 6, 8, 10]),
    )

    # Polarization must remain exactly along +y
    assert np.allclose(result["px"], 0.0)
    assert np.allclose(result["py"], 1.0)
    assert np.allclose(result["pz"], 0.0)
    assert np.allclose(result["p_abs"], 1.0)

    # No vertical motion
    assert result["max_abs_y_mean"] == 0.0
    assert result["max_y_std"] == 0.0
    assert result["max_3sigma_envelope"] == 0.0

    # Initial particles must not be modified
    assert np.allclose(part.coord, 0.0)
    assert np.allclose(part.spin[1], 1.0)
