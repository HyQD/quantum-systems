from quantum_systems.system_helper import (
    transform_one_body_elements,
    transform_two_body_elements,
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

        self._nuclear_repulsion_energy = None

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
        self.o_up = slice(0, self.n_up)
        self.o_down = slice(0, self.n_down)
        self.v = slice(self.n, self.l)

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
            new_shape = [self._spf.shape[0] * 2, *self._spf.shape[1:]]

            spf = self.np.zeros(tuple(new_shape), dtype=self._spf.dtype)
            spf[::2] += self._spf
            spf[1::2] += self._spf

            self._spf = spf
            assert self._spf.shape[0] == self.l

        if not self._bra_spf is None:
            new_shape = [self._bra_spf.shape[0] * 2, *self._bra_spf.shape[1:]]

            bra_spf = self.np.zeros(tuple(new_shape), dtype=self._bra_spf.dtype)
            bra_spf[::2, :] = self._bra_spf
            bra_spf[1::2, :] = self._bra_spf

            self._bra_spf = bra_spf
            assert self._bra_spf.shape[0] == self.l

    def construct_fock_matrix(self, h, u, f=None):
        """Function setting up the Fock matrix"""
        np = self.np
        o, v = (self.o, self.v)

        if f is None:
            f = np.zeros_like(h)

        f.fill(0)
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
            self._bra_spf = self.np.tensordot(
                c_tilde,
                self._spf.conj() if self._bra_spf is None else self._bra_spf,
                axes=((1), (0)),
            )

        self._spf = self.np.tensordot(c, self._spf, axes=((0), (0)))

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

    @property
    def h(self):
        """Getter returning one-body matrix"""
        return self._h

    @property
    def f(self):
        """Getter returning one-body Fock matrix"""
        return self._f

    @f.setter
    def f(self, f):
        self._f = f

    @property
    def u(self):
        """Getter returning the antisymmetric two-body matrix"""
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
        of the non-interacting Hamiltonian"""
        return self._spf

    @property
    def bra_spf(self):
        """Getter returning the conjugate  single particle functions. This is
        None, unless we are working with a bi-variational basis."""
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
