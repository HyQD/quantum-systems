import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import OneDimensionalHarmonicOscillator


n = 2
l = 20

grid_length = 5
num_grid_points = 1001
mass = 1
omega = 1

odho = OneDimensionalHarmonicOscillator(
    n, l, grid_length, num_grid_points, mass=mass, omega=omega
)
odho.setup_system()

# plt.figure()
# plt.imshow(odho.dipole_moment[0].real)
# plt.figure()
# plt.imshow(odho.dipole_moment[0].imag)
# plt.show()

np.save("odho_dipole_moment", odho.dipole_moment)
np.save("odho_h", odho.h)
np.save("odho_u", odho.u)
np.save("odho_spf", odho.spf)
