import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import ODQD


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

odho = ODQD(n, l, grid_length, num_grid_points)
odho.setup_system(potential=ODQD.HOPotential(omega))
save_data(odho, "odho")

length_of_dw = 5

oddw = ODQD(n, l, 6, num_grid_points)
oddw.setup_system(potential=ODQD.DWPotential(omega, length_of_dw))
save_data(oddw, "oddw")

weight = 1
center = 0
deviation = 2.5

odgauss = ODQD(n, l, 20, num_grid_points)
odgauss.setup_system(
    potential=ODQD.GaussianPotential(weight, center, deviation, np=np)
)
save_data(odgauss, "odgauss")

oddw_smooth = ODQD(n, l, grid_length, num_grid_points)
oddw_smooth.setup_system(potential=ODQD.DWPotentialSmooth(a=5))
save_data(oddw_smooth, "oddw_smooth")
