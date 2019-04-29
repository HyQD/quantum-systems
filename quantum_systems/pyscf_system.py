import numpy as np
from pyscf import gto, scf, ao2mo

class PyscfSystem:
    """
    Class to store and retrieve integrals (in spin-orbital basis) along with other information regarding the Hamiltonian operator.
    Uses pyscf to generate orbitals and integrals.
    """

    def __init__(
        self, molecule, basis, spin_pol=0, eps=1.0e-10, maxit=100, memory=2000
    ):
        """
        Input:
           molecule - any molecular geometry string accepted by pyscf
           basis - string, basis set name
           spin_pol - integer, spin polarization; difference between the number of spin-up and spin-down electrons
           eps - float, UHF convergence threshold
           maxit - integer, max number of SCF iterations
           memory - integer, specifies max allowed memory usage (in pyscf part; in Mb)
        """
        self._DEBUG = False
        self.molecule = molecule
        self.basis = basis
        # setup based on pyscf UHF calculation
        self.ERef, self.H, self.F, self.W, self.dipole, self.nocc, self.nvirt, self.o, self.v = self._UHFsetup(
            spin_pol, eps, maxit, memory
        )
        if self._DEBUG:
            print("NOTE: RUNNING IN DEBUG MODE!")
            assert self._W_symmetries(), "W symmetry problem"

    def _W_symmetries(self, tol=1.0e-12):
        """Private method to check W symmetries"""
        nn = range(self.nocc + self.nvirt)
        symm = True
        asym = True
        for p in nn:
            for q in nn:
                for r in nn:
                    for s in nn:
                        Wpqrs = self.W[p, q, r, s]
                        Wrspq = self.W[r, s, p, q]
                        symm = symm and abs(Wpqrs - Wrspq) < tol
                        Wpqsr = self.W[p, q, s, r]
                        asym = asym and abs(Wpqrs + Wpqsr) < tol
                        Wqprs = self.W[q, p, r, s]
                        asym = asym and abs(Wpqrs + Wqprs) < tol
        return symm and asym

    def _UHFsetup(self, spin_pol, eps, maxit, memory):
        """
        Private method, generates UHF reference determinant in spin-orbital basis and computes F, W, and dipole integrals
        """
        mol = gto.Mole()
        mol.unit = "bohr"
        mol.build(atom=self.molecule, basis=self.basis, symmetry=False)
        mol.spin = spin_pol
        mol.max_memory = memory
        # charges = mol.atom_charges()
        # coords = mol.atom_coords()
        # nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
        # mol.set_common_orig_(nuc_charge_center)
        mol.set_common_orig_(np.array([0.0, 0.0, 0.0]))
        hf = scf.UHF(mol)
        hf.conv_tol = eps
        hf.max_cycle = maxit
        ehf = hf.kernel()
        if not hf.converged:
            raise Exception("UHF wave function not converged!")
        e_tot = hf.e_tot  # total energy, incl. nuclear repulsion
        nao = mol.nao
        mo_occ = hf.mo_occ
        mo_coeff = hf.mo_coeff
        Co = np.hstack(
            (mo_coeff[0][:, mo_occ[0] > 0], mo_coeff[1][:, mo_occ[1] > 0])
        )
        Cv = np.hstack(
            (mo_coeff[0][:, mo_occ[0] == 0], mo_coeff[1][:, mo_occ[1] == 0])
        )
        C = np.hstack((Co, Cv)).astype(complex)
        nso = C.shape[1]
        if self._DEBUG:
            assert C.shape[0] == nao, "Incorrect number of AO functions"
            assert nso == 2 * nao, "Incorrect number of spin-orbitals"
        noa = sum(mo_occ[0] > 0)
        nva = sum(mo_occ[0] == 0)
        nob = sum(mo_occ[1] > 0)
        nvb = sum(mo_occ[1] == 0)
        no = noa + nob
        nv = nva + nvb
        oa = slice(0, noa)
        ob = slice(noa, no)
        va = slice(no, no + nva)
        vb = slice(no + nva, no + nv)
        r_int = mol.intor("int1e_r").reshape(3, nao, nao).astype(complex)
        dipole = np.zeros((3, nso, nso), dtype=complex)
        for i in range(3):
            dipole[i] -= C.T @ (r_int[i, :, :]) @ C
        h1 = C.T.dot(hf.get_hcore()).dot(C)
        eri = np.einsum(
            "ap,bq,abcd,cr,ds->pqrs",
            C,
            C,
            ao2mo.restore(1, mol.intor("int2e"), mol.nao_nr()).astype(complex),
            C,
            C,
            optimize=True,
        )
        dipole[:, oa, ob] = complex(0)
        dipole[:, oa, vb] = complex(0)
        dipole[:, ob, oa] = complex(0)
        dipole[:, ob, va] = complex(0)
        dipole[:, va, ob] = complex(0)
        dipole[:, va, vb] = complex(0)
        dipole[:, vb, oa] = complex(0)
        dipole[:, vb, va] = complex(0)
        h1[oa, ob] = complex(0)
        h1[oa, vb] = complex(0)
        h1[ob, oa] = complex(0)
        h1[ob, va] = complex(0)
        h1[va, ob] = complex(0)
        h1[va, vb] = complex(0)
        h1[vb, oa] = complex(0)
        h1[vb, va] = complex(0)
        eri[oa, ob, :, :] = complex(0)
        eri[oa, vb, :, :] = complex(0)
        eri[ob, oa, :, :] = complex(0)
        eri[ob, va, :, :] = complex(0)
        eri[va, ob, :, :] = complex(0)
        eri[va, vb, :, :] = complex(0)
        eri[vb, oa, :, :] = complex(0)
        eri[vb, va, :, :] = complex(0)
        eri[:, :, oa, ob] = complex(0)
        eri[:, :, oa, vb] = complex(0)
        eri[:, :, ob, oa] = complex(0)
        eri[:, :, ob, va] = complex(0)
        eri[:, :, va, ob] = complex(0)
        eri[:, :, va, vb] = complex(0)
        eri[:, :, vb, oa] = complex(0)
        eri[:, :, vb, va] = complex(0)
        eri = np.swapaxes(eri, 1, 2)
        W = eri - np.swapaxes(eri, 2, 3)
        o = slice(0, no)
        F = h1 + np.einsum("pmqm->pq", W[:, o, :, o])
        if self._DEBUG:
            E = np.trace(F[o, o]) - 0.5 * np.einsum("ijij->", W[o, o, o, o])
            E += mol.energy_nuc()
            assert (
                np.abs(np.real(E) - ehf) < eps
            ), "Incorrect UHF energy from F,W"
        v = slice(no, nso)
        return e_tot, h1,F, W, dipole, no, nv, o, v

    def get_ERef(self):
        """
        Returns the energy of the reference determinant
        """
        return self.ERef

    def get_ov(self):
        """
        Returns slice objects representing occupied and virtual spin-orbitals
        """
        return self.o, self.v

    def get_Dia(self):
        """
        Returns singles spin-orbital energy differences D[i][a] = epsilon[a] - epsilon[i]
        Note: epsilon is taken from the Fock matrix diagonal, epsilon[p] = F[p][p]
        """
        Fdiag = np.diag(self.F)
        Focc = Fdiag[self.o]
        Fvir = Fdiag[self.v]
        return -Focc.reshape(-1, 1) + Fvir

    def get_Dijab(self):
        """
        Returns doubles spin-orbital energy differences 
           D[i][j][a][b] = epsilon[a] - epsilon[i] + epsilon[b] - epsilon[j]
        Note: epsilon is taken from the Fock matrix diagonal, epsilon[p] = F[p][p]
        """
        Fdiag = np.diag(self.F)
        Focc = Fdiag[self.o]
        Fvir = Fdiag[self.v]
        return (
            -Focc.reshape(-1, 1, 1, 1)
            - Focc.reshape(-1, 1, 1)
            + Fvir.reshape(-1, 1)
            + Fvir
        )

    def get_F(self):
        """
        Returns the Fock matrix in spin-orbital basis
        """
        return self.F

    def get_W(self):
        """
        Returns the anti-symmetrized two-electron integrals in spin-orbital basis
        """
        return self.W

    def get_H(self):
        """
        Returns the one-electron integrals in spin-orbital basis
        """
        return self.H  

    def get_dipole(self, component):
        """
        Returns specified component (0 -- x, 1 -- y, 2 -- z) of the electric-dipole moment integrals in spin-orbital basis
        """
        return self.dipole[component]
