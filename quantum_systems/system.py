import copy

from quantum_systems.system_helper import (
    transform_spf,
    transform_bra_spf,
    transform_one_body_elements,
    transform_two_body_elements,
    add_spin_spf,
    add_spin_bra_spf,
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
    check_axis_lengths,
    change_module,
)


class QuantumSystem:
    """Base class defining some of the common methods used by all the different
    quantum systems.
    """

    def __init__(self, n, l, n_up=None, np=None):
        assert n <= l

        if np is None:
            import numpy as np

        self.np = np

        self.set_system_size(n, l, n_up)

        self._h = None
        self._f = None
        self._u = None
        self._s = None
        self._dipole_moment = None

        self._time_evolution_operator = None

        self._spf = None
        self._bra_spf = None

        self._nuclear_repulsion_energy = 0

    def set_system_size(self, n, l, n_up=None):
        if n_up is None:
            n_up = n // 2

        assert n_up <= n

        n_down = n - n_up

        self.n = n
        self.n_up = n_up
        self.n_down = n_down
        self.l = l
        self.m = self.l - self.n

        self.o = slice(0, self.n)
        self.o_up = slice(0, self.n, 2)
        self.o_down = slice(1, self.n, 2)
        self.v = slice(self.n, self.l)
        self.v_up = slice(self.n, self.l, 2)
        self.v_down = slice(self.n + 1, self.l, 2)

    def setup_system(self):
        pass

    def change_module(self, np=None):
        if np is not None:
            self.np = np

        self._h = change_module(self._h, self.np)
        self._f = change_module(self._f, self.np)
        self._s = change_module(self._s, self.np)
        self._u = change_module(self._u, self.np)
        self._spf = change_module(self._spf, self.np)
        self._bra_spf = change_module(self._bra_spf, self.np)
        self._dipole_moment = change_module(self._dipole_moment, self.np)

    def change_to_spin_orbital_basis(self, anti_symmetrize=True):
        self._h = add_spin_one_body(self._h, np=self.np)
        assert all(check_axis_lengths(self._h, self.l))

        self._s = add_spin_one_body(self._s, np=self.np)
        assert all(check_axis_lengths(self._s, self.l))

        self._u = add_spin_two_body(self._u, np=self.np)

        if anti_symmetrize:
            self._u = anti_symmetrize_u(self._u)

        assert all(check_axis_lengths(self._u, self.l))

        self._f = self.construct_fock_matrix(self._h, self._u)

        if not self._dipole_moment is None:
            dipole_moment = [
                add_spin_one_body(self._dipole_moment[i], np=self.np)
                for i in range(len(self._dipole_moment))
            ]

            self._dipole_moment = self.np.array(dipole_moment)
            assert all(check_axis_lengths(self._dipole_moment[0], self.l))

        if not self._spf is None:
            self._spf = add_spin_spf(self._spf, self.np)
            assert self._spf.shape[0] == self.l

        if not self._bra_spf is None:
            self._bra_spf = add_spin_bra_spf(self._bra_spf, self.np)
            assert self._bra_spf.shape[0] == self.l

    def construct_fock_matrix(self, h, u, f=None):
        """Function setting up the Fock matrix, that is, the normal-ordered
        one-body elements. If the axis lengths of the two-body elements are
        half of the number of spin-orbitals, i.e., we are in the
        spin-restricted regime, we assume that the user wishes to compute the
        restricted Fock matrix and that the two-body elements are not
        antisymmetric.

        In a spin-orbital basis we compute:

        .. math:: f^{p}_{q} = h^{p}_{q} + u^{pi}_{qi},

        where :math:`p, q, r, s, ...` run over all indices and `i, j, k, l,
        ...` correspond to the occupied indices.
        For an orbital basis we return:

        .. math:: f^{p}_{q} = h^{p}_{q} + 2 u^{pi}_{qi} - u^{pi}_{iq},

        where the two-body elements are assumed to not be antisymmetric.

        Parameters
        ----------
        h : np.ndarray
            The one-body matrix elements.
        u : np.ndarray
            The two-body matrix elements.
        f : np.ndarray
            An empty array of the same shape as `h` to be filled with the Fock
            matrix elements. Default is `None` which means that we allocate a
            new matrix.

        Returns
        -------
        np.ndarray
            The filled Fock matrix.
        """
        np = self.np
        o, v = (self.o, self.v)

        if f is None:
            f = np.zeros_like(h)

        f.fill(0)

        if all(check_axis_lengths(u, self.l // 2)):
            o = slice(0, self.n // 2)
            v = slice(self.n // 2, self.n // 2 + self.m // 2)
            f += 2 * np.einsum("piqi -> pq", u[:, o, :, o])
            f -= np.einsum("piiq -> pq", u[:, o, o, :])
        else:
            f += np.einsum("piqi -> pq", u[:, o, :, o])

        f += h

        return f

    def cast_to_complex(self):
        np = self.np

        self._h = self._h.astype(np.complex128)
        self._u = self._u.astype(np.complex128)

        if self._f is not None:
            self._f = self._f.astype(np.complex128)

        if self._s is not None:
            self._s = self._s.astype(np.complex128)

    def transform_one_body_elements(self, h, c, c_tilde=None):
        return transform_one_body_elements(h, c, np=self.np, c_tilde=c_tilde)

    def transform_two_body_elements(self, u, c, c_tilde=None):
        return transform_two_body_elements(u, c, np=self.np, c_tilde=c_tilde)

    def change_basis_one_body_elements(self, c, c_tilde=None):
        self._h = transform_one_body_elements(
            self._h, c, np=self.np, c_tilde=c_tilde
        )

        if self._s is not None:
            self._s = transform_one_body_elements(
                self._s, c, c_tilde=c_tilde, np=self.np
            )

    def change_basis_two_body_elements(self, c, c_tilde=None):
        self._u = transform_two_body_elements(
            self._u, c, np=self.np, c_tilde=c_tilde
        )

    def change_basis_dipole_moment(self, c, c_tilde=None):
        dipole_moment = []
        for i in range(self._dipole_moment.shape[0]):
            dipole_moment.append(
                transform_one_body_elements(
                    self._dipole_moment[i], c, np=self.np, c_tilde=c_tilde
                )
            )

        self._dipole_moment = self.np.asarray(dipole_moment)

    def change_basis_spf(self, c, c_tilde=None):
        if c_tilde is not None:
            # In case of bi-orthogonal basis sets, we create an extra set
            # of single-particle functions for the bra-side
            self._bra_spf = transform_bra_spf(self.bra_spf, c_tilde, self.np)

        self._spf = transform_spf(self._spf, c, self.np)

    def change_basis(self, c, c_tilde=None):
        self.change_basis_one_body_elements(c, c_tilde)
        self.change_basis_two_body_elements(c, c_tilde)
        self._f = self.construct_fock_matrix(self._h, self._u)

        if self._dipole_moment is not None:
            self.change_basis_dipole_moment(c, c_tilde)

        if self._spf is not None:
            self.change_basis_spf(c, c_tilde)

    def change_to_hf_basis(self, *args, verbose=False, **kwargs):
        from tdhf import HartreeFock

        hf = HartreeFock(system=self, verbose=verbose, np=self.np)
        c = hf.scf(*args, **kwargs)
        self.change_basis(c)

    def compute_reference_energy(self):
        """Function computing the reference energy."""

        o, v = self.o, self.v

        return self.np.trace(self._h[o, o]) + 0.5 * self.np.trace(
            self.np.trace(self._u[o, o, o, o], axis1=1, axis2=3)
        )

    def compute_particle_density(self, rho_qp, c=None, c_tilde=None):
        """Function computing the particle density for a given one-body density
        matrix. This function (optionally) changes the basis of the
        single-particle states for given coefficient matrices.

        Parameters
        ----------
        rho_qp : np.ndarray
            One-body density matrix
        c : np.ndarray
            Coefficient matrix for basis change. Default is `None` and hence no
            transformation occurs.
        c_tilde : np.ndarray
            Bra-state coefficient matrix. Default is `None` which leads to one
            of two situations. If `c != None` `c_tilde` is assumed to be the
            conjugated transpose of `c`. Otherwise, no transformation occurs.

        Returns
        -------
        np.ndarray
            Particle density with the same dimensions as the grid.
        """
        assert (
            self._spf is not None
        ), "Set up single-particle functions prior to calling this function"

        ket_spf = self.spf
        bra_spf = self.bra_spf

        if c is not None:
            ket_spf = transform_spf(ket_spf, c, self.np)
            c_tilde = c_tilde if c_tilde is not None else c.conj().T
            bra_spf = transform_bra_spf(bra_spf, c_tilde, self.np)

        return compute_particle_density(rho_qp, ket_spf, bra_spf, self.np)

    @property
    def h(self):
        """Getter returning one-body matrix."""
        return self._h

    @property
    def f(self):
        """Getter returning one-body Fock matrix."""
        return self._f

    @f.setter
    def f(self, f):
        self._f = f

    @property
    def u(self):
        """Getter returning the antisymmetric two-body matrix."""
        return self._u

    @property
    def s(self):
        """Getter returning the overlap matrix of the atomic orbitals. If the
        overlap elements don't yet exist, we assume that the overlap is the
        identity.
        """
        np = self.np

        if self._s is None:
            self._s = np.eye(*self._h.shape)

        return self._s

    @property
    def dipole_moment(self):
        return self._dipole_moment

    @property
    def spf(self):
        """Getter returning the single particle functions, i.e, the eigenstates
        of the non-interacting Hamiltonian.
        """
        return self._spf

    @property
    def bra_spf(self):
        """Getter returning the conjugate single particle functions. This is
        `None`, unless we are working with a bi-variational basis.
        """
        if self._bra_spf is None:
            self._bra_spf = self._spf.conj()

        return self._bra_spf

    @property
    def nuclear_repulsion_energy(self):
        return self._nuclear_repulsion_energy

    def get_transformed_h(self, c):
        return transform_one_body_elements(self._h, c, np=self.np)

    def get_transformed_u(self, c):
        return transform_two_body_elements(self._u, c, np=self.np)

    def set_time_evolution_operator(self, time_evolution_operator):
        self._time_evolution_operator = time_evolution_operator
        self._time_evolution_operator.set_system(self)

    @property
    def has_one_body_time_evolution_operator(self):
        if self._time_evolution_operator is None:
            return False

        return self._time_evolution_operator.is_one_body_operator

    @property
    def has_two_body_time_evolution_operator(self):
        if self._time_evolution_operator is None:
            return False

        return self._time_evolution_operator.is_two_body_operator

    def h_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._h

        return self._time_evolution_operator.h_t(current_time)

    def u_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._u

        return self._time_evolution_operator.u_t(current_time)

    def set_h(self, h, add_spin=False):
        if add_spin:
            h = add_spin_one_body(h, np=self.np)

        self._h = h

    def set_u(self, u, add_spin=False, anti_symmetrize=False):
        if add_spin:
            u = add_spin_two_body(u, np=self.np)

        if anti_symmetrize:
            u = anti_symmetrize_u(u)

        self._u = u

    def set_s(self, s, add_spin=False):
        if add_spin:
            s = add_spin_one_body(s, np=self.np)

        self._s = s

    def set_dipole_moment(self, dipole_moment, add_spin=False):
        np = self.np

        if len(dipole_moment.shape) < 3:
            dipole_moment = np.array([dipole_moment])

        if not add_spin:
            self._dipole_moment = dipole_moment
            return

        new_shape = [dipole_moment.shape[0]]
        new_shape.extend(list(map(lambda x: x * 2, dipole_moment.shape[1:])))

        self._dipole_moment = np.zeros(
            tuple(new_shape), dtype=dipole_moment.dtype
        )

        for i in range(len(dipole_moment)):
            self._dipole_moment[i] = add_spin_one_body(dipole_moment[i], np=np)

    def set_spf(self, spf, add_spin=False):
        if not add_spin:
            self._spf = spf
            return

        self._spf = add_spin_spf(spf, self.np)

    def set_bra_spf(self, bra_spf, add_spin=False):
        if not add_spin:
            self._bra_spf = bra_spf
            return

        self._bra_spf = add_spin_bra_spf(bra_spf, self.np)

    def set_nuclear_repulsion_energy(self, nuclear_repulsion_energy):
        self._nuclear_repulsion_energy = nuclear_repulsion_energy

    def copy_system(self):
        """Function creating a deep copy of the current system. This function
        is a hack as we have to temporarily remove the stored module before
        using Python's `copy.deepcopy`-function.

        Returns
        -------
        QuantumSystem
            A deep copy of the current system.
        """

        np = self.np
        self.np = None

        new_system = copy.deepcopy(self)
        new_system.np = np
        self.np = np

        return new_system
