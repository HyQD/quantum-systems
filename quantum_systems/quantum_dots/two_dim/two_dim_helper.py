import numpy as np
import scipy
import sympy
import numba
import math
import pandas as pd

from quantum_systems.quantum_dots.two_dim.coulomb_elements import coulomb_ho


def spf_state(r, theta, p, mass, omega):
    n, m = get_indices_nm(p)

    norm = spf_norm(n, m, mass, omega)
    theta_dep = spf_theta(theta, m)
    radial_dep = spf_radial(r, n, m, mass, omega)

    return norm * theta_dep * radial_dep


def spf_norm(n, m, mass, omega):
    a = bohr_radius(mass, omega)

    norm = a * np.sqrt(
        scipy.special.factorial(n)
        / (np.pi * scipy.special.factorial(n + abs(m)))
    )

    return norm


def bohr_radius(mass, omega):
    return np.sqrt(mass * omega)


def spf_theta(theta, m):
    return np.exp(1j * m * theta)


def spf_radial(r, n, m, mass, omega):
    a = bohr_radius(mass, omega)

    laguerre = scipy.special.assoc_laguerre(a ** 2 * r ** 2, n, abs(m))
    radial_dep = np.exp(-a ** 2 * r ** 2 / 2.0)

    return (a * r) ** abs(m) * laguerre * radial_dep


def spf_radial_function(n, m, mass, omega):
    a = sympy.Float(bohr_radius(mass, omega))

    radial_function = (
        lambda r: (a * r) ** abs(m)
        * sympy.assoc_laguerre(n, abs(m), a ** 2 * r ** 2)
        * sympy.exp(-a ** 2 * r ** 2 / 2.0)
    )

    return radial_function


def radial_integral(r_p, r_q):
    r = sympy.Symbol("r")

    return sympy.integrate(
        r ** 2 * r_p(r).conjugate() * r_q(r), (r, 0, sympy.oo)
    )


def smooth_radial_integral_1(r_p, r_q):
    r = sympy.Symbol("r")

    return sympy.integrate(
        r ** 5 * r_p(r).conjugate() * r_q(r), (r, 0, sympy.oo)
    )


def smooth_radial_integral_2(r_p, r_q):
    r = sympy.Symbol("r")

    return sympy.integrate(
        r ** 3 * r_p(r).conjugate() * r_q(r), (r, 0, sympy.oo)
    )


def smooth_theta_integral_1(m_p, m_q):

    return (-1) ** (abs(m_p - m_q) % 2) * (3 * np.pi / 4)


def smooth_theta_integral_2(m_p, m_q):

    return (-1) ** (abs(m_p - m_q) % 2) * np.pi


def theta_1_integral(m_p, m_q):
    if abs(m_p - m_q) == 1:
        return np.pi

    return 0


def theta_2_integral(m_p, m_q):
    if abs(m_p - m_q) == 1:
        return -(m_p - m_q) * 1j * np.pi

    return 0


def theta_1_tilde_integral(m_p, m_q):
    if abs(m_p - m_q) == 1:
        return 0

    integral = (
        -1j
        * (
            -m_q
            + m_p
            + (m_q - m_p) * np.exp(1j * np.pi * (m_q - m_p))
            - 2 * 1j * np.exp(1j * np.pi * (m_q - m_p) / 2)
        )
        * (1 + np.exp(1j * np.pi * (m_q - m_p)))
        / ((m_q - m_p) ** 2 - 1)
    )

    return integral


def theta_2_tilde_integral(m_p, m_q):
    if abs(m_p - m_q) == 1:
        return 0

    integral = -(1 + np.exp(1j * np.pi * (m_q - m_p))) ** 2 / (
        (m_q - m_p) ** 2 - 1
    )

    return integral


@numba.njit(cache=True, nogil=True)
def get_index_p(n, m):
    num_shells = 2 * n + abs(m) + 1

    previous_shell = 0
    for i in range(1, num_shells):
        previous_shell += i

    current_shell = previous_shell + num_shells

    if m == 0:
        if n == 0:
            return 0

        p = previous_shell + (current_shell - previous_shell) // 2

        return p

    elif m < 0:
        return previous_shell + n

    else:
        return current_shell - (n + 1)


@numba.njit(cache=True, nogil=True)
def get_indices_nm(p):
    n, m = 0, 0
    previous_shell = 0
    current_shell = 1
    shell_counter = 1

    while current_shell <= p:
        shell_counter += 1
        previous_shell = current_shell
        current_shell = previous_shell + shell_counter

    middle = (current_shell - previous_shell) / 2 + previous_shell

    if (current_shell - previous_shell) & 0x1 == 1 and abs(
        p - math.floor(middle)
    ) < 1e-8:
        n = shell_counter // 2
        m = 0

        return n, m

    if p < middle:
        n = p - previous_shell
        m = -((shell_counter - 1) - 2 * n)

    else:
        n = (current_shell - 1) - p
        m = (shell_counter - 1) - 2 * n

    return n, m


@numba.njit(cache=True, nogil=True)
def get_shell_energy(n, m):
    return 2 * n + abs(m) + 1


@numba.njit(cache=True, nogil=True)
def get_one_body_elements(num_orbitals, dtype=np.float64):
    h = np.zeros((num_orbitals, num_orbitals), dtype=dtype)

    for p in range(num_orbitals):
        n, m = get_indices_nm(p)
        h[p, p] = get_shell_energy(n, m)

    return h


@numba.njit(fastmath=True, nogil=True, parallel=True)
def get_coulomb_elements(num_orbitals, dtype=np.float64):

    shape = (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
    u = np.zeros(shape, dtype=dtype)

    for p in numba.prange(num_orbitals):
        n_p, m_p = get_indices_nm(p)
        for q in range(num_orbitals):
            n_q, m_q = get_indices_nm(q)
            for r in range(num_orbitals):
                n_r, m_r = get_indices_nm(r)
                for s in range(num_orbitals):
                    n_s, m_s = get_indices_nm(s)

                    u[p, q, r, s] = coulomb_ho(
                        n_p, m_p, n_q, m_q, n_r, m_r, n_s, m_s
                    )

    return u


def get_shell_energy_B(n, m, omega_c=0, omega=1):
    return omega * (2 * n + abs(m) + 1) - (omega_c * m) / 2


def get_one_body_elements_B(num_orbitals, df, dtype=np.float64, omega_c=0):
    h = np.zeros((num_orbitals, num_orbitals), dtype=dtype)

    for p in range(num_orbitals):
        h[p, p] = df.loc[p, "E"]

    return h


def get_coulomb_elements_B(num_orbitals, df, dtype=np.float64, omega_c=0):
    shape = (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
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


def get_double_well_one_body_elements(
    num_orbitals, omega, mass, barrier_strength, dtype=np.float64, axis=0
):
    h = np.zeros((num_orbitals, num_orbitals), dtype=dtype)

    for p in range(num_orbitals):
        n_p, m_p = get_indices_nm(p)
        r_p = spf_radial_function(n_p, m_p, mass, omega)

        h[p, p] += (
            omega * get_shell_energy(n_p, m_p)
            + omega ** 2 * barrier_strength ** 2 / 8.0
        )

        for q in range(num_orbitals):
            n_q, m_q = get_indices_nm(q)
            r_q = spf_radial_function(n_q, m_q, mass, omega)

            if int(abs(m_p - m_q)) == 1:
                continue

            h[p, q] -= (
                0.5
                * omega ** 2
                * barrier_strength
                * spf_norm(n_p, m_p, mass, omega)
                * spf_norm(n_q, m_q, mass, omega)
                * radial_integral(r_p, r_q)
                * (
                    theta_1_tilde_integral(m_p, m_q)
                    if axis == 0
                    else theta_2_tilde_integral(m_p, m_q)
                )
            )

    return h


def get_smooth_double_well_one_body_elements(
    num_orbitals, omega, mass, a=2, b=2, dtype=np.float64
):
    h = np.zeros((num_orbitals, num_orbitals), dtype=dtype)

    prefactor = omega ** 2 / 4

    for p in range(num_orbitals):
        n_p, m_p = get_indices_nm(p)
        r_p = spf_radial_function(n_p, m_p, mass, omega)

        h[p, p] += omega * get_shell_energy(n_p, m_p) + omega ** 2 * a ** 2 / 64

        for q in range(num_orbitals):
            n_q, m_q = get_indices_nm(q)
            r_q = spf_radial_function(n_q, m_q, mass, omega)

            h[p, q] += (
                prefactor
                * (1 / a ** 2)
                * spf_norm(n_p, m_p, mass, omega)
                * spf_norm(n_q, m_q, mass, omega)
                * smooth_radial_integral_1(r_p, r_q)
                * smooth_theta_integral_1(m_p, m_q)
            )

            h[p, q] -= (
                prefactor
                * ((5 * b) / 2)
                * spf_norm(n_p, m_p, mass, omega)
                * spf_norm(n_q, m_q, mass, omega)
                * smooth_radial_integral_2(r_p, r_q)
                * smooth_theta_integral_2(m_p, m_q)
            )

    return h


def construct_dataframe(n_array, m_array, omega_c=0, omega=1):
    df = pd.DataFrame()
    i = 0
    for n in n_array:
        for m in m_array:
            df.loc[i, "n"] = n
            df.loc[i, "m"] = m
            # energy = get_shell_energy(n, m, omega_c=omega_c, omega=omega)
            df.loc[i, "E"] = get_shell_energy_B(
                n, m, omega_c=omega_c, omega=omega
            )
            i += 1

    df = df.sort_values("E").reset_index().drop("index", axis=1)

    df["level"] = 0
    energies = df["E"].round(decimals=8).unique()

    for i, energy in enumerate(energies):
        energy_position = np.where(np.abs(df["E"] - energy) < 1e-6)[0]
        df.loc[energy_position, "level"] = i

    # Computing degenracy.
    df["level"] = df["level"].astype(int)
    df["degeneracy"] = df["level"].map(df["level"].value_counts().to_dict())

    # Capping size of basis set.
    min_num_states = len(n_array) * (len(n_array) - 1) // 2
    level_of_orbital_cap = df.iloc[min_num_states]["level"]
    while df.iloc[min_num_states]["level"] == level_of_orbital_cap:
        min_num_states += 1

    df["n"] = df["n"].astype(int)
    df["m"] = df["m"].astype(int)

    return df.iloc[:min_num_states]
