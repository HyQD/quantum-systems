from quantum_systems.system import QuantumSystem
from quantum_systems.system_helper import (
    add_spin_one_body,
    add_spin_two_body,
    anti_symmetrize_u,
)
from quantum_systems.pyscf_system import PyscfSystem


class CustomSystem(QuantumSystem):
    """Custom quantum system where a user can pass in matrix elements from
    other sources. The purpose of this class is to allow usage of quantum
    solvers made by the author and collaborators using other sources of matrix
    elements.
    """

    def set_h(self, h, add_spin=False):
        if add_spin:
            h = add_spin_one_body(h, np=self.np)

        assert all(
            self.l == axis for axis in h.shape
        ), "Shape of one-body tensor must match the number of orbitals"

        self._h = h

    def set_u(self, u, add_spin=False, anti_symmetrize=False):
        if add_spin:
            u = add_spin_two_body(u, np=self.np)

        if anti_symmetrize:
            u = anti_symmetrize_u(u)

        assert all(
            self.l == axis for axis in u.shape
        ), "Shape of two-body tensor axis must match the number of orbitals"

        self._u = u

    def set_s(self, s, add_spin=False):
        if add_spin:
            s = add_spin_one_body(s, np=self.np)

        assert all(
            self.l == axis for axis in s.shape
        ), "Shape of overlap tensor must match the number of orbitals"

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

        assert all(
            self.l == axis for axis in new_shape[1:]
        ), "Shape of dipole moment matrices must match the number of orbitals"

        self._dipole_moment = np.zeros(tuple(new_shape))

        for i in range(len(dipole_moment)):
            self._dipole_moment[i] = add_spin_one_body(dipole_moment[i], np=np)

    def set_nuclear_repulsion_energy(self, nuclear_repulsion_energy):
        self._nuclear_repulsion_energy = nuclear_repulsion_energy


def construct_pyscf_system_new(molecule, basis="cc-pvdz", np=None):
    import pyscf

    if np is None:
        import numpy as np

    mol = pyscf.gto.Mole()
    mol.unit = "bohr"
    mol.build(atom=molecule, basis=basis, symmetry=False)
    mol.set_common_origin(np.array([0.0, 0.0, 0.0]))

    n = mol.nelectron
    l = mol.nao * 2

    h = pyscf.scf.hf.get_hcore(mol)
    s = mol.intor_symmetric("int1e_ovlp")
    u = (
        mol.intor("int2e")
        .reshape(l // 2, l // 2, l // 2, l // 2)
        .transpose(0, 2, 1, 3)
    )
    dipole_integrals = mol.intor("int1e_r").reshape(3, l // 2, l // 2)

    cs = CustomSystem(n, l, np=np)
    cs.set_h(h, add_spin=True)
    cs.set_s(s, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.set_dipole_moment(dipole_integrals, add_spin=True)
    cs.cast_to_complex()

    cs.change_to_hf_basis(verbose=True)

    return cs


def construct_pyscf_system(molecule, basis="cc-pvdz", np=None):
    if np is None:
        import numpy as np

    my_pyscf_system = PyscfSystem(molecule, basis)
    h = my_pyscf_system.get_H()
    u = my_pyscf_system.get_W()

    # my_pyscf_system performs the Hartree Fock calculation
    n, l = my_pyscf_system.nocc, h.shape[0]

    dipole_integrals = np.zeros((3, l, l), dtype=np.complex128)
    dipole_integrals[0] = my_pyscf_system.get_dipole(0)
    dipole_integrals[1] = my_pyscf_system.get_dipole(1)
    dipole_integrals[2] = my_pyscf_system.get_dipole(2)

    system = CustomSystem(n, l, np=np)
    system.set_h(h, add_spin=False)
    system.set_u(u, add_spin=False, anti_symmetrize=False)
    system.set_s(np.complex128(np.eye(l)), add_spin=False)
    system.set_dipole_moment(dipole_integrals, add_spin=False)

    return system


def construct_psi4_system(molecule, options, np=None):
    import psi4

    if np is None:
        import numpy as np

    psi4.core.be_quiet()
    psi4.set_options(options)

    mol = psi4.geometry(molecule)
    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    wavefunction = psi4.core.Wavefunction.build(
        mol, psi4.core.get_global_option("BASIS")
    )

    molecular_integrals = psi4.core.MintsHelper(wavefunction.basisset())

    kinetic = np.asarray(molecular_integrals.ao_kinetic())
    potential = np.asarray(molecular_integrals.ao_potential())
    h = kinetic + potential

    u = np.asarray(molecular_integrals.ao_eri()).transpose(0, 2, 1, 3)
    overlap = np.asarray(molecular_integrals.ao_overlap())

    n = wavefunction.nalpha() + wavefunction.nbeta()
    l = 2 * wavefunction.nmo()

    dipole_integrals = [
        np.asarray(mu) for mu in molecular_integrals.ao_dipole()
    ]
    dipole_integrals = np.stack(dipole_integrals)

    system = CustomSystem(n, l, np=np)
    system.set_h(h, add_spin=True)
    system.set_u(u, add_spin=True, anti_symmetrize=True)
    system.set_s(overlap, add_spin=True)
    system.set_dipole_moment(dipole_integrals, add_spin=True)
    system.set_nuclear_repulsion_energy(nuclear_repulsion_energy)

    return system
