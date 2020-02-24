import numpy as np

from quantum_systems import ODQD, GeneralOrbitalSystem


def test_copy():
    odho = GeneralOrbitalSystem(2, ODQD(12, 11, 201))

    odho_2 = odho.copy_system()

    assert id(odho) != id(odho_2)
    assert id(odho.h) != id(odho_2.h)

    np.testing.assert_allclose(odho.h, odho_2.h)

    odho.h[0, 0] = 10

    assert not np.equal(odho.h, odho_2.h).all()
