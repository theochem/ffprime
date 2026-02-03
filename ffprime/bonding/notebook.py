from iodata import load_one
import os
import numpy as np
import sys
from utils.connectivity import parse_gaussian_connectivities
print(os.getcwd())
bonds, angles = parse_gaussian_connectivities("bonding/lig.log")
mol = load_one("bonding/lig.fchk")
hess=mol.athessian
#mol_log = load_one("ffprime/bonding/lig.log",fmt="gaussianlog") connectivities from IODATA parsing can be used in the future
evals,evecs = np.linalg.eigh(hess)
N=len(mol.atnums)
if any(val < 0 for val in evals[:-6]):
    print(f"Negative eigenvalues detected, transforming Hessian")
    hess_inv = np.zeros((3 * N, 3 * N))
    evals = [1.0 if val < 0 and i < len(evals) - 6 else val for i, val in enumerate(evals)] # change negative eigenvalues for 1.0 a.u. 
    evals_m=np.zeros([3 * N, 3 * N]) # build a eigenvalues matrix
    for i in range(len(evals)):
    	evals_m[i][i] = evals[i] + evals_m[i][i] # asign eigenvalues to the matrix
    t_evecs=np.transpose(evecs) # build transpose of eigenvector 
    hess_inv= np.einsum("ij,jk,kl", evecs,evals_m,t_evecs) # get a new hessian with the transformed eigenvalues  
    hess_inv=(hess_inv* (627.509391)) / (0.529**2) # get hessian matrix in Kcal/mol
else:
    hess_inv = (hess * 627.509391) / (0.529**2)  # Convert to Kcal/mol/Å²
def is_real_symmetric(matrix, tol=1e-8):
    is_real = np.isrealobj(matrix)
    is_symmetric = np.allclose(matrix, matrix.T, atol=tol)
    return is_real and is_symmetric
is_real_symmetric(hess)



def get_force_constant_bond(atom_i, atom_j, bonds, hessian):
    """Calculate the force constant for a bond."""
    idx_i = atom_i  
    idx_j = atom_j  
    bond = [idx_i, idx_j]
    # Check if the bond exists in the connectivities
    if any((bond == [a, b] or bond == [b, a]) for *pair, in bonds for a, b in [pair[:2]]):
        # Extract the submatrix for the bond
        sub_hessian = hessian[(3 * idx_i): (3 * (idx_i + 1)),(
                       3 * idx_j): (3 * (idx_j + 1))]
        # Calculate the force constant
        eg_val, eg_vec = np.linalg.eig(sub_hessian)
        vec_bond = mol.atcoords[idx_j] - mol.atcoords[idx_i]
        unit_vec = vec_bond / np.linalg.norm(vec_bond) 
        # Get the force constant along the bond vector
        k_ij = 0
        for i in range(0, 3):
            lol = abs(np.dot(unit_vec, eg_vec[:, i]))
            k_ij += (eg_val[i] * lol)
        k_ij *= -0.5   # Add vibrational scaling factor
        #print(f"Force constant for bond {atom_i}-{atom_j}: {k_ij} kJ/mol/Å²")

        return k_ij
    else:
        raise ValueError(f"Bond {atom_i}-{atom_j} not found in connectivities.")



def compute_all_k_ij(bonds_list,hessian):
        """
        Compute, store, and print the average force constant k_ij for every bond in self.bonds.
        """
        for idx, bond in enumerate(bonds_list):
            a, b = bond[0], bond[1]
            dist = bond[2] if len(bond) > 2 else None
            print(a,b)
            k_ij_1 = get_force_constant_bond(a, b, bonds_list, hessian)
            k_ij_2 = get_force_constant_bond(b, a, bonds_list, hessian)
            k_ij_avg = (0.5 * np.real(k_ij_1 + k_ij_2))*(0.957**2)
            print(f"Force constant k_ij for atoms {a}-{b} : {k_ij_avg}")
            # Store k_ij as the fourth element in the bond tuple
            if dist is not None:
                bonds_list[idx] = (*bond[:3], k_ij_avg)
            else:
                bonds_list[idx] = (*bond[:2], k_ij_avg)
        return bonds_list
compute_all_k_ij(bonds, hess_inv)









# Seminario from article
import pathlib
import re

import numpy as np
fid = open('bonding/lig.fchk', "r")
tline = fid.readline()
# constantes utilizadas 
bohr2m = 0.529177249e-10
hartree2joule = 4.35974434e-18
speed_of_light = 299792458
avogadro = 6.0221413e+23
vib_constant = np.sqrt((avogadro*hartree2joule*1000)/(bohr2m*bohr2m))/(2*np.pi*speed_of_light*100)
loop = 'y'
numbers = [] #Atomic numbers for use in xyz file
list_coords = [] #List of xyz coordinates
coords=[]
hessian_old = []

    #Obtiene los números atómicos, las coordenadas y la hessiana respectivamente
while tline:
        #Si el largo de la linea es mayor a 16 y existe la frase "Atomic numbers", lee las líneas
        if len(tline) > 16 and (tline[0:15].strip() == 'Atomic numbers'):
            tline = fid.readline()
            #mientras el largo de la línea sea menor a 17 o aparezca la frase "Nuclear charges":
            while len(tline) < 17 or (tline[0:16].strip() != 'Nuclear charges'):
                tmp = (tline.strip()).split()
                numbers.extend(tmp)
                tline = fid.readline()
            
        #Get coordinates
        if len(tline) > 31 and tline[0:31].strip() == 'Current cartesian coordinates':
            tline = fid.readline()
            while len(tline) < 15 or ( tline[0:17].strip() != 'Number of symbols' and tline[0:17].strip() != 'Int Atom Types'and tline[0:13].strip() != 'Atom Types'):
                tmp = (tline.strip()).split()
                list_coords.extend(tmp)
                tline = fid.readline()
            N = int( float(len(list_coords)) / 3.0 ) #Number of atoms

        #Gets Hessian 
        if len(tline) > 25 and (tline[0:26].strip() == 'Cartesian Force Constants'):
            tline = fid.readline()
            
            while len(tline) < 13 or (tline[0:21].strip() != 'Nonadiabatic coupling'):
                tmp = (tline.strip()).split()
                hessian_old.extend(tmp)
                tline = fid.readline()

            loop = 'n'

        tline = fid.readline()

fid.close()

unprocessed_Hessian= hessian_old
length_hessian = 3 * N
hessian_art = np.zeros((length_hessian, length_hessian))
m = 0
list_coords = [float(x)*float(0.529) for x in list_coords]
mol.atcoords*0.529
list_coords
    #Write the hessian in a 2D array format 
for i in range (0,(length_hessian)):
    for j in range (0,(i + 1)):
            hessian_art[i][j] = unprocessed_Hessian[m]
            hessian_art[j][i] = unprocessed_Hessian[m]
            m = m + 1
hessian_art = (hessian_art* 627.509391 ) / (0.529**2) 
hess
hessian_art
hessian_art == hess

fid = open('bonding/lig.log', 'r')

tline = fid.readline()
bond_list = []
angle_list = []

n = 1
n_bond = 1
n_angle = 1
tmp = 'R' #States if bond or angle
B = []

#Finds the bond and angles from the .log file
while tline:
    tline = fid.readline()
    #Line starts at point when bond and angle list occurs
    if len(tline) > 80 and tline[0:81].strip() == '! Name  Definition              Value          Derivative Info.                !':
        tline = fid.readline()
        tline = fid.readline()
        #Stops when all bond and angles recorded 
        while ( ( tmp[0] == 'R' ) or (tmp[0] == 'A') ):
            line = tline.split()
            tmp = line[1]
                
            #Bond or angles listed as string
            list_terms = line[2][2:-1]
            print(line[2][2:-1])
            #Bond List 
            if ( tmp[0] == 'R' ): 
                x = list_terms.split(',')
                #Subtraction due to python array indexing at 0
                x = [(int(i) - 1 ) for i in x]
                bond_list.append(x)

            #Angle List 
            if ( tmp[0] == 'A' ): 
                x = list_terms.split(',')
                #Subtraction due to python array indexing at 0
                x = [(int(i) - 1 ) for i in x]
                angle_list.append(x)

            tline = fid.readline()

        #Leave loop
        tline = -1
bond_list
from atomdb import load
c_slater = load(elem="C", charge=0, mult=3, dataset="slater")
from atomdb import Element
c = Element(6)
c.pold['chu']
c.c6['chu']