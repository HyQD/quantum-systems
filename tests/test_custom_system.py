import numpy as np
import warnings

from quantum_systems import (
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
    # construct_psi4_system,
    # construct_pyscf_system,
    construct_pyscf_system_rhf,
)
from quantum_systems.random_basis import RandomBasisSet


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
    dim = 3

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = spas.construct_general_orbital_system()

    assert gos.l == 2 * spas.l


def test_change_of_basis():
    n = 2
    l = 10
    dim = 2
    new_l = 2 * l - n

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = spas.construct_general_orbital_system()

    C_spas = RandomBasisSet.get_random_elements((spas.l, new_l), np)
    C_gos = RandomBasisSet.get_random_elements((gos.l, new_l), np)

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


def test_reference_energy():
    gos_system = construct_pyscf_system_rhf(
        "li 0.0 0.0 0.0; h 0.0 0.0 3.08", basis="cc-pvdz"
    )
    spas_system = construct_pyscf_system_rhf(
        "li 0.0 0.0 0.0; h 0.0 0.0 3.08", basis="cc-pvdz", add_spin=False
    )

    # This energy is found from PySCF's RHF solver
    lih_energy = -7.98367215457454

    assert abs(gos_system.compute_reference_energy() - lih_energy) < 1e-8
    assert abs(spas_system.compute_reference_energy() - lih_energy) < 1e-8
