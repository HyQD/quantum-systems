import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import DWPotential


mass = 1
omega = 1
length_of_dw = 5

n = 2
l = 20
grid_length = 5
num_grid_points = 1001

odho = OneDimensionalHarmonicOscillator(
    n, l, grid_length, num_grid_points, mass=mass, omega=omega
)
odho.setup_system(potential=DWPotential(mass, omega, length_of_dw))

for i in range(l // 2):
    plt.plot(odho.grid, odho.spf[i].real ** 2 + odho.eigen_energies[i])

plt.plot(odho.grid, odho.potential(odho.grid), label=r"$v_{DW}$")
plt.legend(loc="best")
plt.show()
