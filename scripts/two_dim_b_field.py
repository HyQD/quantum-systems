import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimHarmonicOscB, TwoDimensionalDoubleWell


def save_data(system, system_name):
    np.save(f"{system_name}_dipole_moment", system.dipole_moment)
    np.save(f"{system_name}_h", system.h)
    np.save(f"{system_name}_u", system.u)
    np.save(f"{system_name}_spf", system.spf)


num_grid_points = 201
tdhob = TwoDimHarmonicOscB(2, 20, 5, num_grid_points, omega_c=0.5)
tdhob.setup_system()
save_data(tdhob, "tdhob")

tddw = TwoDimensionalDoubleWell(
    2, 20, 8, num_grid_points, l_ho_factor=1, barrier_strength=3, omega=0.8
)
tddw.setup_system(axis=0)
save_data(tddw, "tddw")
