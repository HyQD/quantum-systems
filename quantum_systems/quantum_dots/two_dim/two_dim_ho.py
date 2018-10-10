import numpy as np

from quantum_systems import QuantumSystem
from quantum_systems.quantum_dots.two_dim.two_dim_interface import (
    get_coulomb_elements,
    get_one_body_elements,
)
from quantum_systems.system_helper import (
    get_antisymmetrized_one_body_elements,
    get_antisymmetrized_two_body_elements,
)


class TwoDimensionalHarmonicOscillator(QuantumSystem):
    def __init__(self, n, l, omega=1, mass=1):
        super().__init__(n, l)

        self.omega = omega
        self.mass = mass

    def setup_system(self):
        self.__h = self.omega * get_one_body_elements(self.l).astype(
            np.complex128
        )
        self.__u = np.sqrt(self.omega) * get_coulomb_elements(self.l).astype(
            np.complex128
        )

        self._h = get_antisymmetrized_one_body_elements(self.__h)
        self._u = get_antisymmetrized_two_body_elements(self.__u)
        self.construct_fock_matrix()
        self.cast_to_complex()
