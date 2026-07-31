import numpy as np
from atomdb import Element, load
from denspart.cache import ComputeCache
from denspart.mbis import MBISProModel
from denspart.properties import compute_radial_moments, compute_multipole_moments
from denspart.vh import optimize_reduce_pro_model
from types import SimpleNamespace
from iodata import load_one

import scipy.constants as spc
from scipy.integrate import quad

from ffprime.electrostatics.spherical import (
    dipole_spherical_to_cartesian,
    quadrupole_spherical_to_cartesian,
)

meter: float = 1 / spc.value('Bohr radius')
nanometer: float = 1e-9 * meter
kjmol: float = 1e3 / spc.value('Avogadro constant') / spc.value('Hartree energy')

class Partitioning:
    def __init__(self, scheme, mol, moldens, grid, proatomdb, unique_atnums, pro_level, molecule_iodata=None):
        self.scheme = scheme
        self.mol = mol
        self.moldens = moldens
        self.grid = grid
        self.proatomdb = proatomdb
        self.unique_atnums = unique_atnums
        self.pro_level = pro_level
        self.molecule_iodata = molecule_iodata
        self.part = None

    def compute(self):
        scheme = self.scheme
        mol = self.mol
        moldens = self.moldens
        grid = self.grid
        proatomdb = self.proatomdb
        part = SimpleNamespace()
        part.name = scheme.upper()

        #if proatomdb is not None:
        ref_vol = {}
        elem2mult = {1:2, 6:3, 7:4, 8:3, 16:6, 15:4}
        for i in self.unique_atnums:
            proat_db = load(elem=i, charge=0, mult=elem2mult[i], dataset="gaussian")
            dens_spline = proat_db.dens_func(spin="t", log=True)
            r_max = 1000.0  # or just 6.0
            integrand = lambda r: 4 * np.pi * (r**5) * dens_spline(np.array([r]))[0]
            rcubic, err = quad(integrand, 0, r_max)
            ref_vol[i]=rcubic
            # initialize parameters arrays
        ref_volumes = np.zeros(len(mol.atnums))
        volumes = np.zeros(len(mol.atnums))
        volume_ratios = np.zeros(len(mol.atnums))
        c6s_eff = np.zeros(len(mol.atnums))
        a_eff = np.zeros(len(mol.atnums))
        sigma = np.zeros(len(mol.atnums))
        epsilon = np.zeros(len(mol.atnums))

        # Partitioning scheme selection

        if scheme == "mbis":
            pro_model_init = MBISProModel.from_geometry(mol.atnums, mol.atcoords)
            pro_model, localgrids = optimize_reduce_pro_model(
                pro_model_init,
                grid,
                moldens,
                1e-8,
                1000,
                1e-20,
                ComputeCache(),
            )
            print("Compute Teochem_MBIS partitioning model:")
            radial_moments = compute_radial_moments(pro_model, grid, moldens, localgrids)
            cartesian_moments = compute_cartesian_atomic_moments(pro_model, grid, moldens, localgrids)
            atdipoles, atquads = cartesian_moments[:, 1:4], cartesian_moments[:, 4:]
            print(mol.atnums)
            atcharges = pro_model.charges
            for i, atnum in enumerate(mol.atnums):
                # store atomic volume as the 3rd atomic radial moment
                volumes[i] =  radial_moments[i, 3]
                # store reference atomic volume as the 3rd radial moment of neutral atom
                ref_volumes[i] = ref_vol[atnum]
                volume_ratios[i] = volumes[i] / ref_volumes[i]
                c6s_eff[i] = (volume_ratios[i]) ** 2 * Element(atnum).c6['chu']
                a_eff[i] = (volume_ratios[i]) * Element(atnum).pold['chu']
                sigma[i]=((5.08 * a_eff[i] ** (1.0 / 7.0))/(2 ** (1.0 / 6.0)))/nanometer
                epsilon[i]=(c6s_eff[i]/(2*(5.08 * a_eff[i] ** (1.0 / 7.0))**6))/kjmol

        else:
            raise ValueError(f"Given scheme={scheme} not supported!")

        #save the attributes in part object

        result = SimpleNamespace()

        result.name = scheme.upper()
        result.ref_volumes = ref_volumes
        result.volume_ratios = volume_ratios
        result.charges = atcharges
        result.atdipoles = atdipoles
        result.atquadrupoles = atquads
        result.c6s = c6s_eff
        result.alpha = a_eff
        result.sigma = sigma
        result.epsilon = epsilon
        result.part = part

        self.part = result
        return result


def compute_cartesian_atomic_moments(pro_model, grid, moldens, localgrids):
    """Compute per-atom Cartesian charge/dipole/quadrupole moments.

    ``compute_multipole_moments`` returns pure (real) spherical
    moments per atom, ``pm``, ordered as:

        pm[0:3] -> l=1 moments (Stone order: Q_10, Q_11c, Q_11s)
        pm[3:8] -> l=2 moments (Q_20, Q_21c, Q_21s, Q_22c, Q_22s)

    ``pm`` is expressed in terms of the *electron* density, whereas
    Stone's convention (and the physical, nuclear-frame multipole we
    want to expose) is defined in terms of the *charge* density
    rho_c = -rho_e. That is the only discrepancy between ``pm`` and a
    genuine Stone spherical multipole: there is no extra sqrt(3)/2
    rescaling needed on the m=+-1/+-2 components, because
    ``compute_multipole_moments`` already returns them with the correct
    Racah normalization expected by
    ``ffprime.electrostatics.spherical.quadrupole_spherical_to_cartesian``.

    We therefore only flip the overall sign (electron density -> charge
    density) before delegating the actual spherical -> Cartesian
    conversion to the shared, tested utilities
    ``dipole_spherical_to_cartesian`` and
    ``quadrupole_spherical_to_cartesian``.
    """
    cartesian_moments = []
    pure_moments = compute_multipole_moments(pro_model, grid, moldens, localgrids)

    for i, pm in enumerate(pure_moments):
        # atomic charge
        atmom_cart = [pro_model.charges[i]]

        # ---- Dipole ----
        # pm[0:3] are already in Stone order [Q_10, Q_11c, Q_11s].
        # Only the electron-density -> charge-density sign flip is
        # needed before converting to Cartesian form.
        dipole_cart = -dipole_spherical_to_cartesian(pm[0:3])
        atmom_cart.append(dipole_cart[0])  # x
        atmom_cart.append(dipole_cart[1])  # y
        atmom_cart.append(dipole_cart[2])  # z

        # ---- Quadrupole ----
        # pm[3:8] are already genuine Stone spherical quadrupole
        # components (Racah-normalized); only the electron-density ->
        # charge-density sign flip is applied.
        Q20 = -pm[3]
        Q21c = -pm[4]
        Q21s = -pm[5]
        Q22c = -pm[6]
        Q22s = -pm[7]
        quad_sph = np.array([Q20, Q21c, Q21s, Q22c, Q22s])

        theta = quadrupole_spherical_to_cartesian(quad_sph)

        # Preserve original flattened ordering: xx, xy, xz, yy, yz, zz
        atmom_cart.append(theta[0, 0])  # xx
        atmom_cart.append(theta[0, 1])  # xy
        atmom_cart.append(theta[0, 2])  # xz
        atmom_cart.append(theta[1, 1])  # yy
        atmom_cart.append(theta[1, 2])  # yz
        atmom_cart.append(theta[2, 2])  # zz

        cartesian_moments.append(atmom_cart)

    return np.array(cartesian_moments)