import abc
import copy


class QuantumSystem(metaclass=abc.ABCMeta):
    """Abstract base class defining general methods and data on a quantum
    system. The class is used to further specify a system beyond a general
    ``BasisSet`` by specifying the number of particles in the system.

    Parameters
    ----------
    n : int
        Number of occupied basis functions.
    basis_set : BasisSet
        A ``BasisSet``-class or subclass containing matrix elements and basis
        states.
    """

    def __init__(self, n, basis_set):
        self._basis_set = basis_set

        assert n <= self._basis_set.l

        self.np = self._basis_set.np
        self.set_system_size(n, self._basis_set.l)

        self._time_evolution_operator = None

    def set_system_size(self, n, l):
        """Function setting the system size. Note that ``l`` should
        correspond to the length of each axis of the matrix elements.

        Parameters
        ----------
        n : int
            Number of basis functions.
        l : int
            Number of basis functions.
        """

        assert n <= l

        self.n = n
        self.l = l
        self.m = self.l - self.n

        self.o = slice(0, self.n)
        self.v = slice(self.n, self.l)

    @abc.abstractmethod
    def construct_fock_matrix(self, h, u, f=None):
        pass

    def change_module(self, np):
        """Function converting between modules.

        Parameters
        ----------
        np : module
            Array- and linalg-module.
        """

        self.np = np
        self._basis_set.change_module(self.np)

    def change_basis(self, C, C_tilde=None):
        self._basis_set.change_basis(C, C_tilde)
        self.set_system_size(self.n, self._basis_set.l)

    @abc.abstractmethod
    def change_to_hf_basis(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def compute_reference_energy(self):
        pass

    def compute_particle_density(self, rho_qp, C=None, C_tilde=None):
        return self._basis_set.compute_particle_density(
            rho_qp, C=C, C_tilde=C_tilde
        )

    @property
    def dim(self):
        return self._basis_set.dim

    @property
    def grid(self):
        return self._basis_set.grid

    @property
    def h(self):
        """Getter returning the one-body Hamiltonian."""
        return self._basis_set.h

    @property
    def u(self):
        """Getter returning the two-body Hamiltonian."""
        return self._basis_set.u

    @property
    def s(self):
        """Getter returning the overlap matrix."""
        return self._basis_set.s

    @property
    def position(self):
        """Getter retuning the position matrix."""
        return self._basis_set.position

    @property
    def momentum(self):
        """Getter returning the momentum matrix."""
        return self._basis_set.momentum

    @property
    def dipole_moment(self):
        """Getter returning the dipole moment elements."""
        return self._basis_set.dipole_moment

    @property
    def spf(self):
        """Getter returning the single particle functions."""
        return self._basis_set.spf

    @property
    def bra_spf(self):
        """Getter returning the conjugate single particle functions."""
        return self._basis_set.bra_spf

    @property
    def nuclear_repulsion_energy(self):
        """Getter returning the nuclear repulsion energy."""
        return self._basis_set.nuclear_repulsion_energy

    @property
    def particle_charge(self):
        """Getter returning the electronic charge of the particles."""
        return self._basis_set.particle_charge

    def set_time_evolution_operator(self, time_evolution_operator):
        # TODO: Support list of time-evolution operators
        self._time_evolution_operator = time_evolution_operator
        if not self._time_evolution_operator is None:
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
            return self._basis_set.h

        return self._time_evolution_operator.h_t(current_time)

    def u_t(self, current_time):
        if self._time_evolution_operator is None:
            return self._basis_set.u

        return self._time_evolution_operator.u_t(current_time)

    def transform_one_body_elements(self, h, C, C_tilde=None):
        return self._basis_set.transform_one_body_elements(
            h, C, np=self.np, C_tilde=C_tilde
        )

    def transform_two_body_elements(self, u, C, C_tilde=None):
        return self._basis_set.transform_two_body_elements(
            u, C, np=self.np, C_tilde=C_tilde
        )

    def copy_system(self):
        """Function creating a deep copy of the current system. This function
        is a hack as we have to temporarily remove the stored module before
        using Python's ``copy.deepcopy``-function.

        Returns
        -------
        QuantumSystem
            A deep copy of the current system.
        """

        np = self.np
        self.np = None
        self._basis_set.np = None

        new_system = copy.deepcopy(self)

        new_system.change_module(np)
        self.change_module(np)

        assert self.np is np
        assert self._basis_set.np is np

        return new_system
