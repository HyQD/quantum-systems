from quantum_systems import HomogeneousElectronGas


def test_table_length():
    assert HomogeneousElectronGas(1, 0, num_dimensions=1).l == 2
    assert HomogeneousElectronGas(1, 1, num_dimensions=1).l == 6
    assert HomogeneousElectronGas(1, 4, num_dimensions=1).l == 10
    assert HomogeneousElectronGas(1, 9, num_dimensions=1).l == 14
    assert HomogeneousElectronGas(1, 16, num_dimensions=1).l == 18

    assert HomogeneousElectronGas(1, 0, num_dimensions=2).l == 2
    assert HomogeneousElectronGas(1, 1, num_dimensions=2).l == 10
    assert HomogeneousElectronGas(1, 2, num_dimensions=2).l == 18
    assert HomogeneousElectronGas(1, 4, num_dimensions=2).l == 26
    assert HomogeneousElectronGas(1, 5, num_dimensions=2).l == 42

    assert HomogeneousElectronGas(1, 0, num_dimensions=3).l == 2
    assert HomogeneousElectronGas(1, 1, num_dimensions=3).l == 14
    assert HomogeneousElectronGas(1, 2, num_dimensions=3).l == 38
    assert HomogeneousElectronGas(1, 3, num_dimensions=3).l == 54
    assert HomogeneousElectronGas(1, 4, num_dimensions=3).l == 66
    assert HomogeneousElectronGas(1, 5, num_dimensions=3).l == 114
    assert HomogeneousElectronGas(1, 6, num_dimensions=3).l == 162
    assert HomogeneousElectronGas(1, 8, num_dimensions=3).l == 186
