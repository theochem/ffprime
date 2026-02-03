def get_force_constant_bond(atom_i, atom_j, bonds, coords, hessian):
    """Calculate the force constant for a bond.""" 
    bond = [atom_i, atom_j]
    # Check if the bond exists in the connectivities
    if any((bond == [a, b] or bond == [b, a]) for a, b, _ in bonds):
        # Extract the submatrix for the bond
        sub_hessian = hessian[(3 * atom_i): (3 * (atom_i + 1)),(
                       3 * atom_j): (3 * (atom_j + 1))]
        # Calculate the force constant
        eg_val, eg_vec = np.linalg.eig(sub_hessian)
        vec_bond = coords[atom_j] - coords[atom_i]
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