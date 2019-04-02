import numpy as np
import pandas as pd

# TODO: Numba

from quantum_systems.quantum_dots.two_dim.coulomb_elements import coulomb_ho


def get_shell_energy(n, m, omega_c=0, omega=1):
    return omega * (2 * n + abs(m) + 1) - (omega_c * m) / 2


def construct_dataframe(n_array, m_array, omega_c=0, omega=1):
    df = pd.DataFrame()
    i = 0
    for n in n_array:
        for m in m_array:
            df.loc[i, "n"] = n
            df.loc[i, "m"] = m
            energy = get_shell_energy(n, m, omega_c=omega_c, omega=omega)
            df.loc[i, "E"] = get_shell_energy(n, m, omega_c=omega_c, omega=omega)
            i += 1

    df = df.sort_values("E").reset_index().drop("index", axis=1)

    df["level"] = 0
    energies = df["E"].round(decimals=8).unique()

    for i, energy in enumerate(energies):
        energy_position = np.where(np.abs(df["E"] - energy) < 1e-6)[0]
        df.loc[energy_position, "level"] = i

    return df


def get_one_body_elements(num_orbitals, dtype=np.float64, df=None, omega_c=0):

    # This does note work if DataFrame is not None
    # if df == None:
    #     n_array = np.arange(num_orbitals)
    #     m_array = np.arange(-num_orbitals, num_orbitals + 1)
    #     df = construct_dataframe(n_array, m_array, omega_c=omega_c)

    h = np.zeros((num_orbitals, num_orbitals), dtype=dtype)

    for p in range(num_orbitals):
        h[p, p] = df.loc[p, "E"]

    return h


def get_coulomb_elements(num_orbitals, dtype=np.float64, df=None, omega_c=0):

    # This does not work if DataFrame is not None
    # if df == None:
    #     n_array = np.arange(num_orbitals)
    #     m_array = np.arange(-num_orbitals, num_orbitals + 1)
    #     df = construct_dataframe(n_array, m_array, omega_c=omega_c)

    shape = tuple([num_orbitals] * 4)
    u = np.zeros(shape, dtype=dtype)

    for p in range(num_orbitals):
        n_p, m_p = df.loc[p, ["n", "m"]].values
        for q in range(num_orbitals):
            n_q, m_q = df.loc[q, ["n", "m"]].values
            for r in range(num_orbitals):
                n_r, m_r = df.loc[r, ["n", "m"]].values
                for s in range(num_orbitals):
                    n_s, m_s = df.loc[s, ["n", "m"]].values

                    u[p, q, r, s] = coulomb_ho(
                        n_p, m_p, n_q, m_q, n_r, m_r, n_s, m_s
                    )

    return u
