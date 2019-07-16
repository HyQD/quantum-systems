import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.quantum_dots.one_dim.one_dim_potentials import (
    HOPotential,
    DWPotential,
    GaussianPotential,
    DWPotentialSmooth,
)


def save_data(system, system_name):
    np.save(f"{system_name}_dipole_moment", system.dipole_moment)
    np.save(f"{system_name}_h", system.h)
    np.save(f"{system_name}_u", system.u)
    np.save(f"{system_name}_spf", system.spf)


n = 2
l = 20

grid_length = 5
num_grid_points = 1001
omega = 1

odho = OneDimensionalHarmonicOscillator(n, l, grid_length, num_grid_points)
odho.setup_system(potential=HOPotential(omega))
save_data(odho, "odho")

length_of_dw = 5

oddw = OneDimensionalHarmonicOscillator(n, l, 6, num_grid_points)
oddw.setup_system(potential=DWPotential(omega, length_of_dw))
save_data(oddw, "oddw")

weight = 1
center = 0
deviation = 2.5

odgauss = OneDimensionalHarmonicOscillator(n, l, 20, num_grid_points)
odgauss.setup_system(
    potential=GaussianPotential(weight, center, deviation, np=np)
)
save_data(odgauss, "odgauss")

oddw_smooth = OneDimensionalHarmonicOscillator(
    n, l, grid_length, num_grid_points
)
oddw_smooth.setup_system(potential=DWPotentialSmooth(a=5))
save_data(oddw_smooth, "oddw_smooth")
