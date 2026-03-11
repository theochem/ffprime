from ffprime.utils.connectivity import parse_gaussian_connectivities
from iodata import load_one
import os       
import numpy as np
__all__ = ["Bonded", "main_b_deriv"]
class Bonded:
    def __init__(self, log_path="bonding/lig.log", fchk_path="bonding/lig.fchk"):
        self.bonds, self.angles = parse_gaussian_connectivities(log_path)
        self.mol = load_one(fchk_path)
        self.hess = self.mol.athessian
        if np.isrealobj(self.hess) and np.allclose(self.hess, self.hess.T, atol=1e-8):
            evals, evecs = np.linalg.eigh(self.hess)
        else:
            raise ValueError("Hessian matrix is not real symmetric.")
        N = len(self.mol.atnums)
        tolerance = -1e-4  # skip small negative eigenvalues due to numerical noise
        self.hess_new = self.hess.copy()
        if np.any(evals < tolerance):
            print(f"Negative eigenvalues detected, transforming Hessian")
            evals = [1.0 if val < tolerance else val for val in evals]
            evals_m = np.zeros([3 * N, 3 * N])
            for i in range(len(evals)):
                evals_m[i][i] = evals[i]
            #self.hess_new = evecs @ evals_m @ np.linalg.inv(evecs)
            self.hess_new = evecs @ evals_m @ evecs.T  # Reconstruct Hessian using eigenvalues and eigenvectors
        # Check if the reconstructed Hessian matches the original
       
            if np.allclose(evecs.T, np.linalg.inv(evecs)):
                print("Eigenvectors are orthonormal.")
            else:
                raise ValueError("Eigenvectors are not orthonormal.")

        # Conversion to kcal/mol/A^2 (Hartree/Bohr^2 to kcal/mol/A^2)    
        # Units: (627.509391 kcal/mol / Hartree) / (0.529177 A / Bohr)^2
        self.hess_new = (self.hess_new * 627.509391) / (0.529177 ** 2)
        self.hess = (self.hess * 627.509391) / (0.529177 ** 2)
        if np.allclose(self.hess, self.hess_new, atol=1e-8):
            print("Hessian matrix is real symmetric and orthonormal.")

    def get_force_constant(self, atom_i, atom_j, hess=None):
        """
        Calculate the force constant (k_ij) for a pair of atoms (atom_i, atom_j).
        hess: Hessian matrix obtained from a frequency calculation.
        """
        if hess is None:
            hess = self.hess_new
        idx_i = atom_i
        idx_j = atom_j
        # Extract the submatrix for the atom pair
        sub_hessian = hess[(3 * idx_i): (3 * (idx_i + 1)), (3 * idx_j): (3 * (idx_j + 1))]
        # Calculate the force constant
        eg_val, eg_vec = np.linalg.eig(sub_hessian)
        vec = self.mol.atcoords[idx_j] - self.mol.atcoords[idx_i]
        unit_vec = vec / np.linalg.norm(vec)
        # Get the force constant along the vector connecting the atoms
        k_ij = 0
        for i in range(3):
            proj = abs(np.dot(unit_vec, eg_vec[:, i]))
            k_ij += (eg_val[i] * proj)
        k_ij *= -0.5   # Add vibrational scaling factor
        return k_ij
    
    def get_angle_force_constant(self, atom_i, atom_j, atom_k, hess=None):
        """
        Calculate the force constant ($k_{\\theta}$) for the angle $i-j-k$.

        This method implements the Seminario formula:
        $$\\frac{1}{k_{\\theta}} = \\frac{r_{ji}^2}{k_{PA}} + \\frac{r_{jk}^2}{k_{PC}}$$
        where $k_{PA}$ and $k_{PC}$ are projected force constants perpendicular 
        to the bonds in the plane of the angle.

        Parameters
        ----------
        atom_i
            Index of the first terminal atom.
        atom_j
            Index of the central atom.
        atom_k
            Index of the second terminal atom.
        hess
            Hessian matrix in kcal/mol/A^2. Defaults to self.hess_new.

        Returns
        -------
        k_theta
            The derived angle force constant in kcal/mol/rad^2.
        """
        if hess is None:
            hess = self.hess_new
        
        # Geometry vectors
        v_ji = self.mol.atcoords[atom_i] - self.mol.atcoords[atom_j]
        u_ji = v_ji / np.linalg.norm(v_ji)

        v_jk = self.mol.atcoords[atom_k] - self.mol.atcoords[atom_j]
        u_jk = v_jk / np.linalg.norm(v_jk)

        # Define the coordinate system for the angle plane
        u_N = np.cross(u_jk, u_ji)
        u_N /= np.linalg.norm(u_N)

        u_PA = np.cross(u_N, u_ji)
        u_PA /= np.linalg.norm(u_PA)

        u_PC =  np.cross(u_jk, u_N)
        u_PC /= np.linalg.norm(u_PC)

        # Off-diagonal Hessian blocks
        H_ij = hess[(3 * atom_i):(3 * (atom_i + 1)), (3 * atom_j):(3 * (atom_j + 1))]
        H_kj = hess[(3 * atom_k):(3 * (atom_k + 1)), (3 * atom_j):(3 * (atom_j + 1))]

        eval_ij, evec_ij = np.linalg.eig(H_ij)
        eval_kj, evec_kj = np.linalg.eig(H_kj)

        # Projections onto the bending directions
        k_PA = -np.real(sum(eval_ij[m] * abs(np.dot(u_PA, evec_ij[:, m])) for m in range(3)))
        k_PC = -np.real(sum(eval_kj[m] * abs(np.dot(u_PC, evec_kj[:, m])) for m in range(3)))

        # Harmonic combination
        k_theta = 1.0 / ((np.linalg.norm(v_ji)**2 / k_PA) + (np.linalg.norm(v_jk)**2 / k_PC))

        return k_theta

    def compute_all_k_ij(self, bonds_list, hessian):
        """
        Compute, store, and print the average force constant k_ij for every bond in bonds_list.
        """
        for idx, bond in enumerate(bonds_list):
            a, b = bond[0], bond[1]
            dist = bond[2] if len(bond) > 2 else None
            k_ij_1 = self.get_force_constant(a, b, hess=hessian)
            k_ij_2 = self.get_force_constant(b, a, hess=hessian)
            k_ij_avg = (0.5 * np.real(k_ij_1 + k_ij_2)) * (0.957 ** 2)
            # Store k_ij as the fourth element in the bond tuple
            if dist is not None:
                bonds_list[idx] = (*bond[:3], k_ij_avg)
            else:
                bonds_list[idx] = (*bond[:2], k_ij_avg)
        #return bonds_list
if __name__ == "__main__":
    job = Bonded(log_path="bonding/lig.log", fchk_path="bonding/lig.fchk")
    #job.compute_all_k_ij(job.bonds, job.hess_new)
    #print([b[3] for b in job.bonds])  # Print the force constants for each bond