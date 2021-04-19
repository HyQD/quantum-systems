import numpy as np

from quantum_systems import (
    RandomBasisSet,
    SpatialOrbitalSystem,
    GeneralOrbitalSystem,
)
from quantum_systems.time_evolution_operators import (
    DipoleFieldInteraction,
    CustomOneBodyOperator,
    AdiabaticSwitching,
)


def test_no_operators():
    n = 4
    l = 10
    dim = 3

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = GeneralOrbitalSystem(n, RandomBasisSet(l, dim))

    assert not spas.has_one_body_time_evolution_operator
    assert not gos.has_one_body_time_evolution_operator
    assert not spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    np.testing.assert_allclose(spas.h_t(10), spas.h)
    np.testing.assert_allclose(spas.u_t(10), spas.u)
    np.testing.assert_allclose(gos.h_t(10), gos.h)
    np.testing.assert_allclose(gos.u_t(10), gos.u)

    spas.set_time_evolution_operator(
        [],
        add_h_0=False,
        add_u_0=False,
    )
    gos.set_time_evolution_operator([], add_h_0=False, add_u_0=False)

    assert not spas.has_one_body_time_evolution_operator
    assert not gos.has_one_body_time_evolution_operator
    assert not spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    np.testing.assert_allclose(spas.h_t(0), np.zeros_like(spas.h))
    np.testing.assert_allclose(spas.u_t(0), np.zeros_like(spas.u))
    np.testing.assert_allclose(gos.h_t(0), np.zeros_like(gos.h))
    np.testing.assert_allclose(gos.u_t(0), np.zeros_like(gos.u))


def test_single_time_evolution_operator():
    n = 4
    l = 10
    dim = 3

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = GeneralOrbitalSystem(n, RandomBasisSet(l, dim))

    assert not spas.has_one_body_time_evolution_operator
    assert not gos.has_one_body_time_evolution_operator
    assert not spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    np.testing.assert_allclose(spas.h_t(10), spas.h)
    np.testing.assert_allclose(spas.u_t(10), spas.u)
    np.testing.assert_allclose(gos.h_t(10), gos.h)
    np.testing.assert_allclose(gos.u_t(10), gos.u)

    spas.set_time_evolution_operator(
        CustomOneBodyOperator(2, spas.h), add_h_0=False
    )
    gos.set_time_evolution_operator(
        CustomOneBodyOperator(3, gos.h), add_u_0=False
    )

    assert spas.has_one_body_time_evolution_operator
    assert gos.has_one_body_time_evolution_operator
    assert not spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    np.testing.assert_allclose(
        spas.h_t(0),
        spas.h * 2,
    )
    np.testing.assert_allclose(spas.u_t(0), spas.u)

    np.testing.assert_allclose(
        gos.h_t(0),
        gos.h + gos.h * 3,
    )
    np.testing.assert_allclose(gos.u_t(0), np.zeros_like(gos.u))


def test_single_dipole_time_evolution_operator():
    n = 4
    l = 10
    dim = 3

    omega = 0.25

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = GeneralOrbitalSystem(n, RandomBasisSet(l, dim))

    field = lambda t: np.sin(omega * 2)
    polarization = np.zeros(dim)
    polarization[0] = 1

    spas.set_time_evolution_operator(
        DipoleFieldInteraction(
            field,
            polarization,
        )
    )
    gos.set_time_evolution_operator(
        DipoleFieldInteraction(
            field,
            polarization,
        )
    )

    assert spas.has_one_body_time_evolution_operator
    assert gos.has_one_body_time_evolution_operator
    assert not spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    for t in [0, 0.1, 0.5, 1.3]:
        np.testing.assert_allclose(
            spas.h_t(t),
            spas.h - field(t) * spas.dipole_moment[0],
        )
        np.testing.assert_allclose(
            gos.h_t(t),
            gos.h - field(t) * gos.dipole_moment[0],
        )

        np.testing.assert_allclose(spas.u_t(t), spas.u)
        np.testing.assert_allclose(gos.u_t(t), gos.u)


def test_multiple_time_evolution_operators():
    n = 4
    l = 10
    dim = 3

    spas = SpatialOrbitalSystem(n, RandomBasisSet(l, dim))
    gos = GeneralOrbitalSystem(n, RandomBasisSet(l, dim))

    assert not spas.has_one_body_time_evolution_operator
    assert not gos.has_one_body_time_evolution_operator
    assert not spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    spas.set_time_evolution_operator(
        [
            CustomOneBodyOperator(2, spas.h),
            CustomOneBodyOperator(3, spas.s),
            AdiabaticSwitching(2),
        ],
        add_u_0=False,
    )

    gos.set_time_evolution_operator(
        (
            CustomOneBodyOperator(1, gos.h),
            CustomOneBodyOperator(3, gos.s),
            CustomOneBodyOperator(-2, gos.position[0]),
        ),
        add_h_0=False,
    )

    assert spas.has_one_body_time_evolution_operator
    assert gos.has_one_body_time_evolution_operator
    assert spas.has_two_body_time_evolution_operator
    assert not gos.has_two_body_time_evolution_operator

    np.testing.assert_allclose(
        spas.h_t(0),
        spas.h + spas.h * 2 + spas.s * 3,
    )
    np.testing.assert_allclose(spas.u_t(0), 2 * spas.u)

    np.testing.assert_allclose(
        gos.h_t(0),
        gos.h + gos.s * 3 - gos.position[0] * 2,
    )
    np.testing.assert_allclose(gos.u_t(0), gos.u)
