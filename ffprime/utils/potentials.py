import numpy as np
import scipy.spatial
import scipy.constants as spc
from scipy.optimize import newton

# The unit conversion factors below can be used as follows:
angstrom: float = spc.angstrom / spc.value("atomic unit of length")
electronvolt: float = 1 / spc.value("hartree-electron volt relationship")


def compute_energy_coulomb_interaction(qa, qb, c1, c2, unit="au"):
    r"""
    Compute intermolecular electrostatic interaction energy using point-charge approximation.
    ..math:
      E_{coul} = \sum_{j>i}^{}\frac{q_{i}q_{j}}{r_{ij}}
    Parameters
    ----------
    q1 : ndarray shape (M, )
        Atomic point charge monomer 1
    q2 : ndarray shape (N, )
        Atomic point charge monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    unit : str, optional
        Unit of computed energy, options are 'au' or 'eV'.
    Returns
    -------
    c_energy : np.float64
        Electrostatic interaction energy in chosen unit.
    """
    # check charge and coordinates lenghts
    if len(qa) != len(c1):
        raise ValueError(
            f"Expected q1 and c1 to have the same length; got {len(qa)} and {len(c1)}"
        )
    if len(qb) != len(c2):
        raise ValueError(
            f"Expected q2 and c2 to have the same length; got {len(qb)} and {len(c2)}"
        )
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  
    # list of the charges product
    q_mult = np.multiply(
        np.array(qa).reshape(-1, 1), np.array(qb).reshape(1, -1)
    ).flatten()
    # distance unit in bohr
    c_ener = q_mult / (r12)  
    if unit == "eV":
        # conversion to eV
        return sum(c_ener) / electronvolt  
    else:
        return sum(c_ener)
        
def compute_electrostatic_energy_with_cp(
    qa,
    qb,
    atnum_a,
    atnum_b,
    c1,
    c2,
    unit="au",
):
    r"""
    Compute intermolecular electrostatic energy with charge penetration correction.
    ..math:
       E_{elec}^{CP}=\sum_{A, B>A}\frac{Z_{A}Z_{B}}{r_{AB}}+\sum_{A, B > A}\frac{Z_{A}\mathrm{q}_{B}}{r_{AB}}f_{damp}\left( r_{AB} \right)+\sum_{A, B > A}\frac{\mathrm{q}_{A}Z_{B}}{\mathbf{r}_{AB}}f_{damp}\left(r_{AB} \right)+\sum_{A, B>A}\frac{\mathrm{q}_{A}\mathrm{q}_{B}}{r_{AB}}f_{damp}^{overlap}\left(r_{AB} \right)
    Parameters
    ----------
    qa : ndarray shape (M, )
        Atomic c6 coefficient of monomer 1
    qb : ndarray shape (N, )
        Atomic c6 coefficient of monomer 2
    atnum_a : ndarray shape (M, )
        Atomic number for each atom in monomer 1
    atnum_b : ndarray shape (N, )
        Atomic number for each atom in monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy (units depends on units of C6 coefficients).
    """
    alpha = { # alpha value for every system 
     1: 10.000, 6: 2.9137, 7: 3.4066,
            8: 3.5677,}
    beta = 0.9150
    
    # check atomic charge and atomic number lenghts 
    if len(qa) != len(atnum_a):
        raise ValueError(
            f"Expected qa and atnum_a to have the same length; got {len(qa)} and {len(atnum_a)}"
        )
    if len(qb) != len(atnum_b):
        raise ValueError(
            f"Expected c6b and atnum_b to have the same length; got {len(qb)} and {len(atnum_b)}"
        )
        # check charge and coordinates lenghts
    if len(qa) != len(c1):
        raise ValueError(
            f"Expected q1 and c1 to have the same length; got {len(qa)} and {len(c1)}"
        )
    if len(qb) != len(c2):
        raise ValueError(
            f"Expected q2 and c2 to have the same length; got {len(qb)} and {len(c2)}"
        )
    # compute electron charges in fragment 1 and 2
    rho_a = -1*np.subtract(np.array(atnum_a), np.array(qa))
    rho_b = -1*np.subtract(np.array(atnum_b), np.array(qb))
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    dist_ab = scipy.spatial.distance.cdist(c1, c2, "euclidean")
    dist_ba = scipy.spatial.distance.cdist(c2, c1, "euclidean")
    
    #Nuclei Nuclei interaction
    q_nuclei = np.multiply(
            np.array(atnum_a).reshape(-1, 1), np.array(atnum_b).reshape(1, -1)
        ).flatten()
    c_nn = q_nuclei / ((dist_ab).flatten())
    
    #Nuclei A electron B interaction 
    q_na_eb = np.multiply(
            np.array(rho_b).reshape(-1, 1), np.array(atnum_a).reshape(1, -1)
        ).flatten()
    alpha_b = np.array([alpha[n] for n in atnum_b])
    # check charge penetration parameters
    for num in atnum_b:
        if num not in alpha.keys():
            raise ValueError(f"alpha parameter {num} not found, available options: {alpha.keys()}")
    
    damp_ba = (1 - np.exp(-1*alpha_b[:, np.newaxis]*dist_ba)).flatten()# damping function to electron B
    c_ebna = (q_na_eb/(dist_ba).flatten())*damp_ba
    
    #Nuclei B electron A interaction
    q_nb_ea = np.multiply(
            np.array(rho_a).reshape(-1, 1), np.array(atnum_b).reshape(1, -1)
        ).flatten()
    alpha_a = np.array([alpha[n] for n in atnum_a])
    # check charge penetration parameters
    for num in atnum_a:
        if num not in alpha.keys():
            raise ValueError(f"alpha parameter {num} not found, available options: {alpha.keys()}")
    
    damp_ab = (1 - np.exp(-1*alpha_a[:, np.newaxis]*dist_ab)).flatten()# damping function to electron A
    c_eanb = (q_nb_ea/(dist_ab).flatten())*damp_ab

    # check atomic charges and CP parameters lenghts 
    if len(qa) != len(alpha_a):
        raise ValueError(
            f"Expected qa and alpha_a to have the same length; got {len(qa)} and {len(alpha_a)}"
        )
    if len(qb) != len(alpha_b):
        raise ValueError(
            f"Expected qb and alpha_b to have the same length; got {len(qb)} and {len(alpha_b)}"
        )
    #electron a electron b interaction
    q_ea_eb = np.multiply(
            np.array(rho_a).reshape(-1, 1), np.array(rho_b).reshape(1, -1)
        ).flatten()
    damp_ov_model2 = (1 - np.exp(-1*beta*alpha_a[:, np.newaxis]*dist_ab)).flatten()*(1 - np.exp(-1*beta*alpha_b[:, np.newaxis]*dist_ba)).flatten()
    c_ee = (q_ea_eb / (dist_ab*angstrom).flatten())*damp_ov_model2
    # distance unit in bohr
    c_ener = sum(c_nn)+sum(c_ebna)+sum(c_eanb)+sum(c_ee)
    # unit depend on C6 coefficients
    if unit == "eV":
        # conversion to eV
        return c_ener / electronvolt  
    else:
        return c_ener 


def compute_energy_dispersion_interaction(
    c6a,
    c6b,
    c1,
    c2,
):
    r"""
    Compute intermolecular dispersion interaction energy using C6 coefficients.
    ..math:
      E_{disp} = -\sum_{j>i}^{}\frac{\sqrt{C_{6,i}C_{6,j}}}{r^{6}_{ij}}
    Parameters
    ----------
    C6a : ndarray shape (M, )
        Atomic c6 coefficient of monomer 1
    c6b : ndarray shape (N, )
        Atomic c6 coefficient of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy.
    """
    # check c6 coefficient and coordinates lenghts
    if len(c6a) != len(c1):
        raise ValueError(
            f"Expected c6a and c1 to have the same length; got {len(c6a)} and {len(c1)}"
        )
    if len(c6b) != len(c2):
        raise ValueError(
            f"Expected c6b and c2 to have the same length; got {len(c6b)} and {len(c2)}"
        )
    # check c6 coefficient are positive
    if any(np.array(c6a) < 0):
        raise ValueError(f"Expected c6a coefficients to be positive; got {c6a}")
    if any(np.array(c6b) < 0):
        raise ValueError(f"Expected c6b coefficients to be positive; got {c6b}")
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  # Euclidean distance
    # list of the c6 geometric mean
    c6_mean = (
        np.multiply(
            np.array(c6a).reshape(-1, 1), np.array(c6b).reshape(1, -1)
        ).flatten()
    ) ** 0.5 
    # distance unit in bohr
    d_ener = -c6_mean / ((r12) ** 6)
    # unit depend on C6 coefficients
    return sum(d_ener)  
    
def compute_energy_dispersion_interaction_LB(
    sa,
    sb,
    ea,
    eb,
    c1,
    c2,
):
    r"""
    Compute intermolecular dispersion interaction energy using Lennard Jonnes coefficients and the Lorentz-Berthelot combination rule.
    ..math:
      E_{disp} = -\sum_{j>i} \frac{4\epsilon_{ij} \sigma_{ij}^{6}}{r^{6}_{ij}}; && \epsilon_{ij}=\sqrt{\epsilon_{i}\epsilon_{j}},\sigma_{ij}=\frac{\sigma_{i}+\sigma_{j}}{2}
    Parameters
    ----------
    sa : ndarray shape (M, )
        Sigma parameters of monomer 1
    sb : ndarray shape (N, )
        Sigma parameters of monomer 2
    ea : ndarray shape (M, )
        Epsilon parameters of monomer 1
    eb : ndarray shape (N, )
        Epsilon parameters of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy.
    """
    # check c6 coefficient and coordinates lenghts
    if len(sa) != len(c1):
        raise ValueError(
            f"Expected sa and c1 to have the same length; got {len(sa)} and {len(c1)}"
        )
    if len(ea) != len(c1):
        raise ValueError(
            f"Expected ea and c1 to have the same length; got {len(ea)} and {len(c1)}"
        )
    if len(sb) != len(c2):
        raise ValueError(
            f"Expected sb and c2 to have the same length; got {len(sb)} and {len(c2)}"
        )
    if len(eb) != len(c2):
        raise ValueError(
            f"Expected eb and c2 to have the same length; got {len(eb)} and {len(c2)}"
        )
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  # Euclidean distance
    eab = (np.multiply(
            np.array(ea).reshape(-1, 1), np.array(eb).reshape(1, -1)
        ).flatten())**0.5

    sab = np.add.outer(sa, sb).flatten() / 2
    # list of the c6 geometric mean
    c6ab = 4*eab*(sab**6)
    # distance unit in bohr
    d_ener = -c6ab/((r12)**6)
    # unit depend on C6 coefficients
    return sum(d_ener)  

def compute_energy_repulsion_interaction(
    c12a,
    c12b,
    c1,
    c2,
):
    r"""
    Compute intermolecular repulsion interaction energy using atomic C12 coefficients.
    ..math:
      E_{rep} = \sum_{j>i}^{}\frac{\sqrt{C_{12,i}C_{12,j}}}{r^{12}_{ij}}
    Parameters
    ----------
    C12a : ndarray shape (M, )
        Atomic c12 coefficient of monomer 1
    c12b : ndarray shape (N, )
        Atomic c12 coefficient of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy (units depends on units of C6 coefficients).
    """
    # check c6 coefficient and coordinates lenghts
    if len(c12a) != len(c1):
        raise ValueError(
            f"Expected c12a and c1 to have the same length; got {len(c12a)} and {len(c1)}"
        )
    if len(c12b) != len(c2):
        raise ValueError(
            f"Expected c12b and c2 to have the same length; got {len(c12b)} and {len(c2)}"
        )
    # check c6 coefficient are positive
    if any(np.array(c12a) < 0):
        raise ValueError(f"Expected c12a coefficients to be positive; got {c12a}")
    if any(np.array(c12b) < 0):
        raise ValueError(f"Expected c12b coefficients to be positive; got {c12b}")
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  # Euclidean distance
    # list of the c6 geometric mean
    c12_mean = (
        np.multiply(
            np.array(c12a).reshape(-1, 1), np.array(c12b).reshape(1, -1)
        ).flatten()
    ) ** 0.5 
    # distance unit in bohr
    d_ener = c12_mean / ((r12) ** 12)
    # unit depend on C6 coefficients
    return sum(d_ener)  

def compute_energy_repulsion_interaction_LB(
    sa,
    sb,
    ea,
    eb,
    c1,
    c2,
):
    r"""
    Compute intermolecular repulsion interaction energy using Lennard Jonnes coefficients and the Lorentz-Berthelot combination rule.
    ..math:
      E_{rep} = \sum_{j>i} \frac{4\epsilon_{ij} \sigma_{ij}^{12}}{r^{12}_{ij}}; && \epsilon_{ij}=\sqrt{\epsilon_{i}\epsilon_{j}},\sigma_{ij}=\frac{\sigma_{i}+\sigma_{j}}{2}
    Parameters
    ----------
    sa : ndarray shape (M, )
        Sigma parameters of monomer 1
    sb : ndarray shape (N, )
        Sigma parameters of monomer 2
    ea : ndarray shape (M, )
        Epsilon parameters of monomer 1
    eb : ndarray shape (N, )
        Epsilon parameters of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    Returns
    -------
    d_energy : np.float64
        Repulsion interaction energy.
    """
    # check c6 coefficient and coordinates lenghts
    if len(sa) != len(c1):
        raise ValueError(
            f"Expected sa and c1 to have the same length; got {len(sa)} and {len(c1)}"
        )
    if len(ea) != len(c1):
        raise ValueError(
            f"Expected ea and c1 to have the same length; got {len(ea)} and {len(c1)}"
        )
    if len(sb) != len(c2):
        raise ValueError(
            f"Expected sb and c2 to have the same length; got {len(sb)} and {len(c2)}"
        )
    if len(eb) != len(c2):
        raise ValueError(
            f"Expected eb and c2 to have the same length; got {len(eb)} and {len(c2)}"
        )
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  # Euclidean distance
    eab = (np.multiply(
            np.array(ea).reshape(-1, 1), np.array(eb).reshape(1, -1)
        ).flatten())**0.5

    sab = np.add.outer(sa, sb).flatten() / 2
    # list of the c6 geometric mean
    c12ab = 4*eab*(sab**12)
    # distance unit in bohr
    r_ener = c12ab/((r12)**12)
    # unit depend on C6 coefficients
    return sum(r_ener)  

def compute_energy_dispersion_interaction_denspart_tang(
    c6a,
    c6b,
    pol_a,
    pol_b,
    c1,
    c2,
):
    r"""
    Compute intermolecular dispersion interaction energy using C6 coefficients.
    ..math:
      E_{disp} = -\sum_{j>i}^{}\frac{C_{6,ij}^{Tang}}{r^{6}_{ij}}; && C_{6,ij}^{Tang}=\frac{2C_{6,i}C_{6,j}}{\frac{\alpha_{j}}{\alpha_{i}}C_{6,i}+\frac{\alpha_{i}}{\alpha_{j}}C_{6,j}}
    Parameters
    ----------
    C6a : ndarray shape (M, )
        Atomic c6 coefficients of monomer 1
    c6b : ndarray shape (N, )
        Atomic c6 coefficients of monomer 2
    pol_a : ndarray shape (M, )
        Atomic polarizabilities of monomer 1
    pol_b : ndarray shape (N, )
        Atomic polarizabilities of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy (units depends on units of C6 coefficients).
    """

    # check c6s, atomic polarizabilities and coordinates have the same lenghts
    if len(c6a) != len(c1):
        raise ValueError(
            f"Expected c6a and c1 to have the same length; got {len(c6a)} and {len(c1)}"
        )
    if len(c6b) != len(c2):
        raise ValueError(
            f"Expected c6b and c2 to have the same length; got {len(c6b)} and {len(c2)}"
        )
    if len(pol_a) != len(c1):
        raise ValueError(
            f"Expected pol_a and c1 to have the same length; got {len(pol_a)} and {len(c1)}"
        )
    if len(pol_b) != len(c2):
        raise ValueError(
            f"Expected pol_b and c2 to have the same length; got {len(pol_b)} and {len(c2)}"
        )
    # check c6s and polarizabilities coefficients are positive
    if np.any(np.array(c6a) < 0):
        raise ValueError(f"Expected c6a coefficient to be positive; got {c6a}")
    if np.any(np.array(c6b) < 0):
        raise ValueError(f"Expected c6b coefficient to be positive; got {c6b}")
    if np.any(np.array(pol_a) < 0):
        raise ValueError(f"Expected pol_a coefficient to be positive; got {pol_a}")
    if np.any(np.array(pol_b) < 0):
        raise ValueError(f"Expected pol_b coefficient to be positive; got {pol_b}")
    
    c6ab = (
        np.multiply(
            np.array(c6a).reshape(-1, 1), np.array(c6b).reshape(1, -1)
            ).flatten()
        ) * 2  
    da = (np.divide(np.array(pol_b).reshape(1, -1), np.array(pol_a).reshape(-1, 1)) * np.array(c6a).reshape(-1, 1)).flatten()
    db = (np.divide(np.array(pol_a).reshape(-1, 1), np.array(pol_b).reshape(1, -1)) * np.array(c6b).reshape(1, -1)).flatten()
    c6_mean = c6ab / np.add(da,db)
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  # Euclidean distance
    # distance unit in bohr
    d_ener = -c6_mean / ((r12) ** 6)
    # unit depend on C6 coefficients
    return np.sum(d_ener)  

def compute_energy_dispersion_interaction_TKAT(
    c6a,
    c6b,
    c1,
    c2,
    alpha_a,
    alpha_b,
    return_mu_omega=False
):
    r"""
    Compute intermolecular dispersion interaction energy using C6, c8 and c10 coefficients
    """
    # Reference: ¨Universal Pairise Interatomic van der Waals Potentials Based on Quantum Drude oscillator.¨
    #            Khabibrakhmanov, A. and Fedorov, V. D. and Tkatchenko, A. J. Chem Theory Comput. 2023, 19, 7895-7909
    #            https://doi.org/10.1021/acs.jctc.3c00797
    # ============================================================================================
    # 1. Input validation
    # ============================================================================================
    c6a = np.asarray(c6a, dtype=float)
    c6b = np.asarray(c6b, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    c2 = np.asarray(c2, dtype=float)
    alpha_a = np.asarray(alpha_a, dtype=float)
    alpha_b = np.asarray(alpha_b, dtype=float)
    
    if len(c6a) != len(c1):
        raise ValueError(f"c6a and c1 length mismatch: {len(c6a)} vs {len(c1)}")
    if len(c6b) != len(c2):
        raise ValueError(f"c6b and c2 length mismatch: {len(c6b)} vs {len(c2)}")
    if len(c6a) != len(alpha_a):
        raise ValueError(f"c6a and alpha_a length mismatch: {len(c6a)} vs {len(alpha_a)}")
    if len(c6b) != len(alpha_b):
        raise ValueError(f"c6b and alpha_b length mismatch: {len(c6b)} vs {len(alpha_b)}")
    
    if np.any(c6a < 0) or np.any(c6b < 0):
        raise ValueError("C6 coefficients must be positive")
    if np.any(alpha_a < 0) or np.any(alpha_b < 0):
        raise ValueError("Polarizabilities must be positive")
    
    # ============================================================================================
    # 2. Compute distances in Bohr
    # ============================================================================================
    c1_bohr = c1 * angstrom
    c2_bohr = c2 * angstrom
    
    r12 = scipy.spatial.distance.cdist(c1_bohr, c2_bohr, metric="euclidean").flatten()
    if len(r12) == 0:
        return 0.0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # ============================================================================================
    # 3. Effective polarizabilities (Eq. 21 from ref: α_AB = (α_A + α_B) / 2)
    # ============================================================================================
    alpha_matrix = (alpha_a[:, np.newaxis] + alpha_b[np.newaxis, :]) / 2
    alpha_pair = alpha_matrix.flatten()
    
    # ============================================================================================
    # 4. Effective C6 coefficients (Eq. 19 from ref)
    # ============================================================================================
    c6_a_matrix = c6a[:, np.newaxis]
    c6_b_matrix = c6b[np.newaxis, :]
    alpha_a_matrix = alpha_a[:, np.newaxis]
    alpha_b_matrix = alpha_b[np.newaxis, :]
    
    numerator = 2.0 * alpha_a_matrix * alpha_b_matrix * c6_a_matrix * c6_b_matrix
    denominator = c6_a_matrix * (alpha_b_matrix ** 2) + c6_b_matrix * (alpha_a_matrix ** 2)
    c6_matrix = numerator / denominator
    c6_pair = c6_matrix.flatten()
    
    # ============================================================================================
    # 5. Quantum Drude oscillator parametrization (Eq. S33 supporting info)
    # ============================================================================================

    alpha_fsc = spc.alpha
    a_const = 9 * alpha_fsc**(4/3) / 64

    muomega = np.zeros_like(alpha_pair)
    for i, alpha in enumerate(alpha_pair):
        b_const = 2 * alpha**(2/7) / alpha_fsc**(8/21)
        def f(x):
            """Equation: a*exp(b*x) - (2*x² + x/b) = 0"""
            exponent = b_const * x
            if exponent > 700:
                return a_const * np.exp(700) - 2*x**2 - x/b_const
            return a_const*np.exp(exponent) - 2*x**2 - x/b_const


        def df(x):
            """Derivative"""
            exponent = b_const*x
            if exponent > 700:
                return a_const*b_const*np.exp(700) - 4*x - 1/b_const
            return a_const*b_const*np.exp(exponent) - 4*x - 1/b_const
        x0 = 0.5 + 0.3/(1 + alpha/10) # Initial guess
        
        x = newton(
        f,
        x0,
        fprime=df,
        tol=1e-12,
        maxiter=100
    )

    if np.isfinite(x) and x > 1e-8:
        muomega[i] = x
    else:
        raise RuntimeError(f"Non physical root for alpha={alpha}")
    
    # ============================================================================================
    # 6. Compute C8 and C10 coefficients (Eq. 2 from ref)
    # ============================================================================================
    c8_pair = 5 * c6_pair / muomega
    c10_pair = (245 / 8) * c6_pair / muomega**2
    
    # ============================================================================================
    # 7. Compute dispersion energy with QDO damping function
    #    Damping function (Eq. 24 from reference)
    #    f_{2n}(z) = 1 - exp(-z) * sum_{k=0}^{n} z^k / k!
    #    z = (gamma * R)^2 / 2, gamma = sqrt(muomega) (in a.u., hbar = 1)
    # ============================================================================================
    r6 = r12 ** 6
    r8 = r12 ** 8
    r10 = r12 ** 10
    
    # Calculate damping functions
    gamma = np.sqrt(muomega)
    z = (gamma[:, np.newaxis] * r12)**2 / 2
    
    f6 = 1.0 - np.exp(-z) * (1 + z + z**2/2 + z**3/6)
    f8 = 1.0 - np.exp(-z) * (1 + z + z**2/2 + z**3/6 + z**4/24)
    f10 = 1.0 - np.exp(-z) * (1 + z + z**2/2 + z**3/6 + z**4/24 + z**5/120)
    
    # Damped dispersion energy
    HARTREE_TO_KCAL = 627.509  # kcal/mol
    d_ener_hartree = -f6 * c6_pair / r6 - f8 * c8_pair / r8 - f10 * c10_pair / r10
    total_energy_hartree = np.sum(d_ener_hartree)
    total_energy_kcal = total_energy_hartree * HARTREE_TO_KCAL

    # ============================================================================================
    # 8. Optional: Return mu and omega separately parameters for QDO model
    # ============================================================================================
    if return_mu_omega:
        # C₆ = (3/4) * omega * α²  (atomic units)
        omega = (4 * c6_pair) / (3 * alpha_pair**2)
        mu = muomega / omega
        return total_energy_kcal, c6_pair, c8_pair, c10_pair, muomega, r12, mu, omega
    
    return total_energy_kcal, c6_pair, c8_pair, c10_pair, muomega, r12

def compute_d1_grimme_dispersion_interaction(
    c6a,
    c6b,
    c1,
    c2,
    atnum_a,
    atnum_b,
    method="blyp",
):
    r"""
    Compute intermolecular dispersion interaction energy using Grimme D1 formula:
    ..math:
      E_{D1} = - \sum_{j>i}^{}\frac{s_{6}}{1+e^{-23\left( \frac{r_{ij}}{R_{0ij}}-1\right)}}
                 \frac{\frac{2C_{6,i}C_{6,j}}{C_{6,i}+C_{6,j}}}{r^{6}_{ij}}
    Parameters
    ----------
    c6a : ndarray shape (M, )
        Atomic c6 coefficient of monomer 1
    c6b : ndarray shape (N, )
        Atomic c6 coefficient of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    atnum_a : ndarray shape (M, )
        Atomic number for each atom in monomer 1
    atnum_b : ndarray shape (N, )
        Atomic number for each atom in monomer 2
    method : str, optional
        The s6 value according to a density functional, options are 'pbe', 'blyp', and 'bp86'.
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy (units depends on units of C6 coefficients).
    """
    # Reference: “Accurate Description of van Der Waals Complexes by Density Functional
    #            Theory Including Empirical Corrections.”
    #            Grimme, Stefan. J. Comp. Chem., vol. 25, no. 12, 2004, pp. 1463–73
    #            https://doi.org/10.1002/jcc.20078.

    # reference R0 (van der Waals radii) values in Å units from Table 1
    ref_r0 = {
        1: 1.11,
        6: 1.61,
        7: 1.55,
        8: 1.49,
        9: 1.43,
        10: 1.38,
    }
    # reference s6 (scale factor) values from "Theory" section
    s6 = {  
        "pbe": 0.7,
        "blyp": 1.4,
        "bp86": 1.3,
    }
    # check c6 coefficient and coordinates lenghts
    if len(c6a) != len(c1):
        raise ValueError(
            f"Expected c6a and c1 to have the same length; got {len(c6a)} and {len(c1)}"
        )
    if len(c6b) != len(c2):
        raise ValueError(
            f"Expected c6b and c2 to have the same length; got {len(c6b)} and {len(c2)}"
        )
    # check c6 coefficient are positive
    if any(np.array(c6a) < 0):
        raise ValueError(f"Expected c6a coefficiente to be positive; got {c6a}")
    if any(np.array(c6b) < 0):
        raise ValueError(f"Expected c6b coefficiente to be positive; got {c6b}")
    # check method
    if method not in s6.keys():
        raise ValueError(f"Unrecognized method={method}, available options: {s6.keys()}")

    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()
    c6_mult = (
        np.multiply(
            np.array(c6a).reshape(-1, 1), np.array(c6b).reshape(1, -1)
        ).flatten()
    ) * 2
    # list of the c6 average
    c6_mean = (
        c6_mult
        / (
            np.array(c6a).reshape(len(c6a), 1) + np.array(c6b).reshape(1, len(c6b))
        ).flatten()
    )
    # distance unit in bohr
    d_ener = -c6_mean / ((r12) ** 6)
    # make arrays of reference R0 for atoms in each fragment
    r0a = np.array([ref_r0[n] for n in atnum_a])
    r0b = np.array([ref_r0[n] for n in atnum_b])
    r_mean = (r0a.reshape(-1, 1) + r0b.reshape(1, -1)).flatten()
    # add Grimme correction with fermi damping function (D1)
    d_ener *= s6[method] / (1.0 + np.exp(-23 * ((r12 / r_mean) - 1.0)))
    # unit depend on C6 coefficients
    return np.sum(d_ener)


def compute_d2_grimme_dispersion_interaction(
    c6a,
    c6b,
    c1,
    c2,
    atnum_a,
    atnum_b,
    method="blyp",
):
    r"""
    Compute intermolecular dispersion interaction energy using Grimme D2 formula:
    ..math:
      E_{D2} = -\sum_{j>i}^{}\frac{s_{6}}{1+e^{-20\left(
                \frac{r_{ij}}{R_{0ij}}-1\right)}}\frac{\sqrt{C_{6,i}C_{6,j}}}{r^{6}_{ij}}
    Parameters
    ----------
    c6a : ndarray shape (M, )
        Atomic c6 coefficient of monomer 1
    c6b : ndarray shape (N, )
        Atomic c6 coefficient of monomer 2
    c1 : ndarray shape (M, 3)
        Atomic Cartesian coordinates monomer 1
    c2 : ndarray shape (N, 3)
        Atomic Cartesian coordinates monomer 2
    atnum_a : ndarray shape (M, )
        Atomic number for each atom in monomer 1
    atnum_b : ndarray shape (N, )
        Atomic number for each atom in monomer 2
    method : str, optional
        The s6 value according to a density functional, options are 'pbe', 'blyp', 'bp86', 'tpss',
        'b3lyp', and 'b97'.
    Returns
    -------
    d_energy : np.float64
        Dispersion interaction energy (units depends on units of C6 coefficients).
    """
    # Reference: “Semiempirical GGA-Type Density Functional Constructed with a 
    #            Long-Range Dispersion Correction.”
    #            Grimme, Stefan. J. Comp. Chem.,vol. 27, no. 15, 2006, pp. 1787–99
    #            https://doi.org/10.1002/jcc.20495.
    
    # reference R0 (van der Waals radii) values in Å units from Table 1
    ref_r0 = {  
        1: 1.001,
        2: 1.012,
        3: 0.825,
        4: 1.408,
        5: 1.485,
        6: 1.452,
        7: 1.397,
        8: 1.342,
        9: 1.287,
        10: 1.243,
        11: 1.144,
        12: 1.364,
        13: 1.639,
        14: 1.716,
        15: 1.705,
        16: 1.683,
        17: 1.639,
        18: 1.595,
        19: 1.485,
        20: 1.474,
    }
    # reference s6 (scale factor) values from "Paramater Fitting" section
    s6 = {   
        "pbe": 0.75,
        "blyp": 1.2,
        "bp86": 1.05,
        "tpss": 1.0,
        "b3lyp": 1.05,
        "b97": 1.25,
    }
    # check c6 coefficient and coordinates lenghts
    if len(c6a) != len(c1):
        raise ValueError(
            f"Expected c6a and c1 to have the same length; got {len(c6a)} and {len(c1)}"
        )
    if len(c6b) != len(c2):
        raise ValueError(
            f"Expected c6b and c2 to have the same length; got {len(c6b)} and {len(c2)}"
        )
    # check c6 coefficient are positive
    if any(np.array(c6a) < 0):
        raise ValueError(f"Expected c6a coefficiente to be positive; got {c6a}")
    if any(np.array(c6b) < 0):
        raise ValueError(f"Expected c6b coefficiente to be positive; got {c6b}")
    # check method
    if method not in s6.keys():
        raise ValueError(f"Unrecognized method={method}, available options: {s6.keys()}")
    # compute Euclidean distance between pairs of atoms in fragment 1 and 2
    r12 = scipy.spatial.distance.cdist(c1, c2, "euclidean").flatten()  # Euclidean distance
    # list of the c6 geometric mean
    c6_mean = (
        np.multiply(
            np.array(c6a).reshape(-1, 1), np.array(c6b).reshape(1, -1)
        ).flatten()
    ) ** 0.5  
    # distance unit in bohr
    d_ener = -c6_mean / ((r12) ** 6)
    # make arrays of reference R0 for atoms in each fragment
    r0a = np.array([ref_r0[n] for n in atnum_a])
    r0b = np.array([ref_r0[n] for n in atnum_b])
    r_mean = (r0a.reshape(-1, 1) + r0b.reshape(1, -1)).flatten()
    # add Grimme correction with fermi damping function (D2)
    d_ener *= s6[method] / (1.0 + np.exp(-20 * ((r12 / r_mean) - 1.0)))
    # unit depend on C6 coefficients
    return sum(d_ener)  
    
def compute_molecular_c6 (c6a,c6b,atnum_a,atnum_b):
    r"""
    Compute molecular C6 coefficient using Tkatchenko combination rule:
    ..math:
       C_{6AB} = \frac{2C_{6AA}C_{6BB}}{\frac{\alpha_{B}}{\alpha_{A}}
                 C_{6AA}+\frac{\alpha_{A}}{\alpha_{B}}C_{6BB}}
                 
    Parameters
    ----------
    c6a : ndarray shape (M, )
        Atomic c6 coefficient of monomer 1
    c6b : ndarray shape (N, )
        Atomic c6 coefficient of monomer 2
    atnum_a : ndarray shape (M, )
        Atomic number for each atom in monomer 1
    atnum_b : ndarray shape (N, )
        Atomic number for each atom in monomer 2
    Returns
    -------
    c6_mol : np.float64
        molecular C6 coefficient (units depends on units of C6 coefficients).
    """
    

    # Reference: “Accurate Molecular Van Der Waals Interactions from Ground-State
    #            Electron Density and Free-Atom Reference Data.”
    #            Tkatchenko, Alexandre, and Matthias Scheffler. Phys. Rev. Lett.
    #            vol. 102, no. 7, Feb. 2009, p. 073005
    #            https://doi.org/10.1103/PhysRevLett.102.073005.

    ref_a = { # reference static polarizabilities of free atoms 
     1: 4.5, 2: 1.38, 3: 164, 4: 38, 5: 21, 6: 12, 7: 7.4,
            8: 5.4, 9: 3.8, 10: 2.67, 11: 163, 12: 71, 13: 60, 14:
            37, 15: 25, 16: 19.6, 17: 15, 18: 11.1, 19: 294, 20:
            160, 21: 120, 22: 98, 23: 84, 24: 78, 25: 63, 26:
            56, 27: 50, 28: 48, 29: 42, 30: 40, 31: 60, 32:
            41, 33: 29, 34: 25, 35: 20, 36: 16.7, 37: 320, 38:
            199, 49: 75, 50: 60, 51: 44, 52: 40, 53: 35, 
        }
    # check c6 coefficient are positive
    if any(np.array(c6a) < 0):
        raise ValueError(f"Expected c6a coefficiente to be positive; got {c6a}")
    if any(np.array(c6b) < 0):
        raise ValueError(f"Expected c6b coefficiente to be positive; got {c6b}")
    
    # check c6 coefficient and atomic number lenghts 
    if len(c6a) != len(atnum_a):
        raise ValueError(
            f"Expected c6a and atnum_a to have the same length; got {len(c6a)} and {len(atnum_a)}"
        )
    if len(c6b) != len(atnum_b):
        raise ValueError(
            f"Expected c6b and atnum_b to have the same length; got {len(c6b)} and {len(atnum_b)}"
        )
    c6ab = (
        np.multiply(
            np.array(c6a).reshape(-1, 1), np.array(c6b).reshape(1, -1)
            ).flatten()
        ) * 2  
    # check static polarizabilities
    for num in atnum_a:
        if num not in ref_a.keys():
            raise ValueError(f"static polarizability {num} not found, available options: {ref_a.keys()}")
    for num in atnum_b:
        if num not in ref_a.keys():
            raise ValueError(f"static polarizability {num} not found, available options: {ref_a.keys()}")
    alpha_a = [ref_a[n] for n in atnum_a]
    alpha_b = [ref_a[n] for n in atnum_b]
    # check c6 coefficient and static polarizability lenghts 
    if len(c6a) != len(alpha_a):
        raise ValueError(
            f"Expected c6a and alpha_a to have the same length; got {len(c6a)} and {len(alpha_a)}"
        )
    if len(c6b) != len(alpha_b):
        raise ValueError(
            f"Expected c6b and alpha_b to have the same length; got {len(c6b)} and {len(alpha_b)}"
        )
    da = (np.divide(np.array(alpha_b).reshape(1, -1), np.array(alpha_a).reshape(-1, 1)) * np.array(c6a).reshape(-1, 1)).flatten()
    db = (np.divide(np.array(alpha_a).reshape(-1, 1), np.array(alpha_b).reshape(1, -1)) * np.array(c6b).reshape(1, -1)).flatten()
    c6_mol = c6ab / np.add(da,db)
    return sum(c6_mol)