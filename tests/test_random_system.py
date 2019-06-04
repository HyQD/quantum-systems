from quantum_systems import RandomSystem


def test_construction():
    n = 12
    l = 40

    rs = RandomSystem(n, l)
    rs.setup_system()

    assert True
