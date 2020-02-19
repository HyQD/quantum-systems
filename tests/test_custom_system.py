import numpy as np
import warnings

from quantum_systems import (
    # construct_psi4_system,
    # construct_pyscf_system,
    construct_pyscf_system_ao,
    construct_pyscf_system_rhf,
)


def change_basis_h(h, c):
    _h = np.einsum("ap,bq,ab->pq", c.conj(), c, h, optimize=True)

    return _h


def change_basis_u(u, c):
    _u = np.einsum(
        "ap,bq,gr,ds,abgd->pqrs", c.conj(), c.conj(), c, c, u, optimize=True
    )

    return _u


def test_setters():
    n = 2
    l = 10

    h = np.random.random((l, l))
    u = np.random.random((l, l, l, l))
    s = np.random.random((l, l))
    dipole_moment = np.random.random((3, l, l))

    cs = QuantumSystem(n, 2 * l)

    cs.set_h(h, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.set_s(s, add_spin=True)
    cs.set_dipole_moment(dipole_moment, add_spin=True)

    assert True


def test_change_of_basis():
    n = 2
    l = 10

    h = np.random.random((l, l))
    u = np.random.random((l, l, l, l))
    s = np.random.random((l, l))
    c = np.random.random((l * 2, l * 2))
    dipole_moment = np.random.random((3, l, l))

    cs = QuantumSystem(n, 2 * l)

    cs.set_h(h, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.set_s(s, add_spin=True)
    cs.set_dipole_moment(dipole_moment, add_spin=True)

    h_cs = change_basis_h(cs.h.copy(), c)
    u_cs = change_basis_u(cs.u.copy(), c)

    cs.change_basis(c)

    np.testing.assert_allclose(h_cs, cs.h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(u_cs, cs.u, atol=1e-12, rtol=1e-12)

    cs = QuantumSystem(n, 2 * l)

    cs.set_h(h, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.set_s(s, add_spin=True)
    cs.set_dipole_moment(dipole_moment, add_spin=True)

    cs.change_basis(c, c_tilde=c.conj().T)

    np.testing.assert_allclose(h_cs, cs.h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(u_cs, cs.u, atol=1e-12, rtol=1e-12)


def test_psi4_construction():
    He = """
        He 0.0 0.0 0.0
        symmetry c1
    """

    options = {"basis": "cc-pVDZ", "scf_type": "pk", "e_convergence": 1e-8}

    _spas = SpatialOrbitalSystem(n, bs)
    spas = _spas.copy_system()
    gos = _spas.change_to_general_orbital_basis()

    C_spas = get_random_elements((spas.l, new_l), np)
    C_gos = get_random_elements((gos.l, new_l), np)

    h_spas = change_basis_h(spas.h.copy(), C_spas)
    u_spas = change_basis_u(spas.u.copy(), C_spas)
    h_gos = change_basis_h(gos.h.copy(), C_gos)
    u_gos = change_basis_u(gos.u.copy(), C_gos)

    spas.change_basis(C_spas)
    gos.change_basis(C_gos)

    assert spas.l == new_l
    assert all([new_l == s for s in spas.h.shape])
    assert all([new_l == s for s in spas.u.shape])
    np.testing.assert_allclose(h_spas, spas.h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(u_spas, spas.u, atol=1e-12, rtol=1e-12)

    assert gos.l == new_l
    assert all([new_l == s for s in gos.h.shape])
    assert all([new_l == s for s in gos.u.shape])
    np.testing.assert_allclose(h_gos, gos.h, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(u_gos, gos.u, atol=1e-12, rtol=1e-12)


# def test_psi4_construction():
#     He = """
#         He 0.0 0.0 0.0
#         symmetry c1
#     """
#
#     options = {"basis": "cc-pVDZ", "scf_type": "pk", "e_convergence": 1e-8}
#
#     try:
#         system = construct_psi4_system(He, options)
#         assert True
#     except ImportError:
#         warnings.warn("Unable to import psi4.")


# def test_pyscf_construction():
#     system = construct_pyscf_system("be 0 0 0")


def test_pyscf_ao_construction():
    system = construct_pyscf_system_ao("be 0 0 0")


def test_reference_energy():
    system = construct_pyscf_system_rhf("he", basis="cc-pvdz")

    # This energy is found from PySCF's RHF solver
    he_energy = -2.85516047724274

    assert abs(system.compute_reference_energy() - he_energy) < 1e-8
