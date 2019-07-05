import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotenial,
    DWPotential,
    GaussianPotential,
    DWPotentialSmooth,
    AtomicPotential,
)


def plot_one_body_system(system):
    for i in range(system.l // 2):
        plt.plot(
            system.grid,
            system.spf[2 * i].real ** 2 + system.eigen_energies[i],
            label=fr"$\psi_{i}(x)$",
        )

    plt.plot(system.grid, system.potential(system.grid), label=r"$v(x)$")
    plt.legend(loc="best")
    plt.show()


n = 2
l = 20
grid_length = 100
num_grid_points = 2001
mass = 1
omega = 1

"""
odho = OneDimensionalHarmonicOscillator(
    n, l, grid_length, num_grid_points, mass=mass, omega=omega
)
odho.setup_system(potential=HOPotenial(mass, omega))
plot_one_body_system(odho)


length_of_dw = 5

dw = OneDimensionalHarmonicOscillator(
    n, l, 6, num_grid_points, mass=mass, omega=omega
)
dw.setup_system(potential=DWPotential(mass, omega, length_of_dw))
plot_one_body_system(dw)

weight = 1
center = 0
deviation = 2.5

gauss = OneDimensionalHarmonicOscillator(
    n, l, 20, num_grid_points, mass=mass, omega=omega
)
gauss.setup_system(
    potential=GaussianPotential(weight, center, deviation, np=np)
)
plot_one_body_system(gauss)

tw_gauss_potential = lambda x: (
    GaussianPotential(weight, center, deviation, np)(x)
    + GaussianPotential(weight, center - 10, deviation, np)(x)
    + GaussianPotential(weight, center + 10, deviation, np)(x)
)

tw_gauss = OneDimensionalHarmonicOscillator(
    n, l, 30, num_grid_points, mass=mass, omega=omega
)
tw_gauss.setup_system(potential=tw_gauss_potential)
plot_one_body_system(tw_gauss)


dw_smooth = OneDimensionalHarmonicOscillator(
    n, l, grid_length, num_grid_points, mass=mass, omega=omega
)
dw_smooth.setup_system(potential=DWPotentialSmooth(a=5))
plot_one_body_system(dw_smooth)
"""

helium_ion = OneDimensionalHarmonicOscillator(
    n, l, grid_length, num_grid_points, mass=mass, omega=omega, a=0.7408
)
helium_ion.setup_system(potential=AtomicPotential(Za=2, c=0.7408 ** 2))
plot_one_body_system(helium_ion)
