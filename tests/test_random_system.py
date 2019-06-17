from quantum_systems import RandomSystem


def test_construction():
    n = 12
    l = 40

    rs = RandomSystem(n, l)
    rs.setup_system()

    assert True


def test_kwargs_conflict():
    n = 12
    l = 40

    rs = RandomSystem(n, l)
    rs.setup_system(anti_symmetrize=True)

    assert True


def test_spin():
    n = 12
    l = 40

    rs = RandomSystem(n, l)
    rs.setup_system(add_spin=True)

    assert True


def test_spin_as():
    n = 12
    l = 40

    rs = RandomSystem(n, l)
    rs.setup_system(add_spin=True, anti_symmetrize=True)

    assert True
