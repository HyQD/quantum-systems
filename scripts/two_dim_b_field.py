import numpy as np
import matplotlib.pyplot as plt

from quantum_systems import TwoDimHarmonicOscB


def save_data(system, system_name):
    np.save(f"{system_name}_dipole_moment", system.dipole_moment)
    np.save(f"{system_name}_h", system.h)
    np.save(f"{system_name}_u", system.u)


tdhob = TwoDimHarmonicOscB(2, 20, 5, 1001, omega_c=0.5)
tdhob.setup_system()

save_data(tdhob, "tdhob")
