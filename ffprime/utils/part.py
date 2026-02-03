import numpy as np
from atomdb import Element, load
from denspart.cache import ComputeCache
from denspart.mbis import MBISProModel
from denspart.properties import compute_radial_moments
from denspart.vh import optimize_reduce_pro_model
from types import SimpleNamespace
from iodata import load_one

import scipy.constants as spc
from scipy.integrate import quad

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
                
   
        
        #save the attributes in part object

        result = SimpleNamespace()

        result.name = scheme.upper()
        result.ref_volumes = ref_volumes
        result.volume_ratios = volume_ratios
        result.charges = atcharges
        result.c6s = c6s_eff
        result.alpha = a_eff
        result.sigma = sigma
        result.epsilon = epsilon
      
        # guarda el objeto "part" real por si lo necesitas
        result.part = part

        self.part = result
        return result
    
        #else:
        #    raise ValueError(f"Given scheme={scheme} not supported!")
    


