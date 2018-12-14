import numpy as np

from quantum_systems import CustomSystem


def test_setters():
    n = 2
    l = 10

    h = np.random.random((l, l))
    u = np.random.random((l, l, l, l))
    s = np.random.random((l, l))
    dipole_moment = np.random.random((3, l, l))

    cs = CustomSystem(n, l)

    cs.set_h(h, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.set_s(s, add_spin=True)
    cs.set_dipole_moment(dipole_moment, add_spin=True)

    assert True
