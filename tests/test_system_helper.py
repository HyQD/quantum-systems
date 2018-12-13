import numpy as np

from quantum_systems.system_helper import (
    get_antisymmetrized_one_body_elements,
    get_antisymmetrized_two_body_elements,
)

from quantum_systems.new_system_helper import (
    add_spin_h,
    add_spin_u,
    antisymmetrize_u
)


def test_spin_expansion_h():
    _h = np.complex128(np.random.random((10, 10)))
    h_c = get_antisymmetrized_one_body_elements(_h)
    h_n = add_spin_h(_h)

    np.testing.assert_allclose(h_c, h_n)


def test_spin_expansion_u():
    np.random.seed(2018)

    _u = np.complex128(np.random.random((2, 2, 2, 2)))
    u_c = get_antisymmetrized_two_body_elements(_u)
    u_n = antisymmetrize_u(add_spin_u(_u))

    np.testing.assert_allclose(u_c, u_n)
