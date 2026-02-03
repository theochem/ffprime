from ffprime.utils.part import Partitioning
from atomdb import Element
import os
import numpy as np
import scipy.constants as spc
from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density
from grid import GaussChebyshev, BeckeRTransform, MolGrid, BeckeWeights
from iodata import load_one


meter: float = 1 / spc.value('Bohr radius')
nanometer: float = 1e-9 * meter
kjmol: float = 1e3 / spc.value('Avogadro constant') / spc.value('Hartree energy')
__all__ = ["Nonbonded"]

class Nonbonded:
    def __init__(self, atnums=None, atcharges=None, atvolumes=None, ref_atvolumes=None):
        """
        If atomic data are provided, compute the derived nonbonded parameters.
        Otherwise create an "empty" Nonbonded container that can be filled later.
        """

        # Persistent storage
        self.molecules = []
        self.unique_atnums = []
        self.molecules_iodata = []
        self.data = {}
        self.proatomdb = None

        # If no direct parameters provided return blank shell
        if any(v is None for v in (atnums, atcharges, atvolumes, ref_atvolumes)):
            print("Warning: empty Nonbonded object created (no input atomic properties).")
            return

        # Pre-allocate arrays
        n = len(atnums)
        volume_ratios = np.zeros(n)
        c6s_eff = np.zeros(n)
        a_eff = np.zeros(n)
        sigma = np.zeros(n)
        epsilon = np.zeros(n)

        # Compute FF parameters
        for i, atnum in enumerate(atnums):
            volume_ratios[i] = atvolumes[i] / ref_atvolumes[i]

            c6s_eff[i] = (volume_ratios[i])**2 * Element(atnum).c6["chu"]
            a_eff[i]  = (volume_ratios[i])    * Element(atnum).pold["chu"]

            sigma[i] = (5.08 * a_eff[i]**(1/7)) / (2**(1/6)) / nanometer
            epsilon[i] = c6s_eff[i] / (2 * (5.08 * a_eff[i]**(1/7))**6) / kjmol

        # Store computed parameters
        self.atnums = atnums
        self.atcharges = atcharges
        self.atvolumes = atvolumes
        self.ref_atvolumes = ref_atvolumes
        self.volume_ratios = volume_ratios
        self.c6s_eff = c6s_eff
        self.a_eff = a_eff
        self.sigma = sigma
        self.epsilon = epsilon


   
    #  ---- Option 1: Load from file ----
  
    @classmethod
    def from_file(cls, fname, schemes, args=None, pro_level=None,
                  pro_charge=[-2,2]):

        self = cls()

        print(f"LOAD {fname}, LOT={pro_level}")

        # Store user parameters
        self.schemes = schemes or ["mbis"]
        self.agspec = args
        self.pro_charge = pro_charge
        self.pro_level = pro_level
        self.pro_agspec = "exp:5e-4:2e1:175:230"

        # Load molecule
        mol = load_one(fname)
        self._register_molecule(mol)

        # Register file info
        folder, fn = os.path.split(fname)
        data_mol = self.data.setdefault(fn, {})
        data_mol["atnums"] = mol.atnums.tolist()
        data_mol["atcoords"] = mol.atcoords.tolist()
        data_mol["denspart"] = {}

        # Partition and assign parameters
        for scheme in self.schemes:
            results = self._process_molecule(mol, method=scheme)

        # Copy results *to the Nonbonded object*
        for key, val in results.items():
            setattr(self, key, val)

        print(f"\nEnd PART for {len(self.molecules)} molecules and {len(self.schemes)} schemes.\n")
        return self


  
    # Option 2: Load an IOData molecule
   
    @classmethod
    def from_IODATA(cls, mol, schemes, args=None, pro_level=None,
                    pro_charge=[-2,2]):

        self = cls()

        # Store configuration
        self.schemes = schemes or ["mbis"]
        self.agspec = args
        self.pro_charge = pro_charge
        self.pro_level = pro_level
        self.pro_agspec = "exp:5e-4:2e1:175:230"

        self._register_molecule(mol)

        # ---- Partition and compute parameters ----
        for scheme in self.schemes:
            results = self._process_molecule(mol, method=scheme)

        # ensure the containers exist
        if not hasattr(mol, "atffparams"):
            mol.atffparams = {}

        if not hasattr(mol, "extra"):
            mol.extra = {}

        # fields that go into atffparams
        ff_fields = ["atcharges", "sigma", "epsilon"]
        ff_extra = ['ref_atvolumes', 'volume_ratios', 'c6s_eff', 'a_eff']

        for key, val in results.items():
        # Convert lists → numpy arrays for consistency
            if isinstance(val, list):
                val = np.array(val)

            if key in ff_fields:
                mol.atffparams[key] = val
            elif key in ff_extra:
                mol.extra[key] = val
            else:
                continue

        print(f"\nEnd PART for 1 molecule using schemes={self.schemes}\n")

        return mol


   
    # Internal utility method to store more than one molecule 
   
    def _register_molecule(self, mol):
        """Register a molecule in the class lists."""
        self.molecules.append(mol)
        self.unique_atnums.extend(mol.atnums.tolist())
        self.unique_atnums = sorted(set(self.unique_atnums))


    def _process_molecule(self, mol, method):
        """Run partitioning and compute nonbonded parameters for the molecule."""
        rdm = mol.one_rdms.get("scf")

        # Build density matrix 
        if rdm is None:
            if mol.mo is None:
                print("Missing mol.mo ! Density matrix...")
                raise NotImplementedError

            print("Couldn't read Density matrix. Calculating from its components.")
            coeffs, occs = mol.mo.coeffs, mol.mo.occs
            rdm = np.dot(coeffs * occs, coeffs.T)
        basis = from_iodata(mol)
        

        print(f"PREPARE molecule ({len(mol.atnums)} atoms)")

        if method == "mbis":
            oned = GaussChebyshev(npoints=100)
            rgrid = BeckeRTransform(0.0, R=1.5).transform_1d_grid(oned)
            grid = MolGrid.from_preset(
                atnums=mol.atnums,
                atcoords=mol.atcoords,
                rgrid=rgrid,
                preset=self.agspec,
                aim_weights=BeckeWeights(),
                store=True,
            )
            moldens = evaluate_density(rdm, basis, grid.points)

        # Run partitioning
        print(f"PART {method.upper()} ...")

        part_job = Partitioning(
            method,
            mol,
            moldens,
            grid,
            self.proatomdb,
            self.unique_atnums,
            self.pro_level,
            molecule_iodata=self.molecules_iodata,
            )
        part = part_job.compute()

        # Output dict to apply to self or IOData
        return {
            "atnums": mol.atnums,
            "atcoords": mol.atcoords,
            "atcharges": part.charges,
            "ref_atvolumes": part.ref_volumes,
            "volume_ratios": part.volume_ratios,
            "c6s_eff": part.c6s,
            "a_eff": part.alpha,
            "sigma": part.sigma,
            "epsilon": part.epsilon,
        }
    
    
    # Save all nonbonded parameters as a NPZ file
    
    def write_npz(self, filename, verbose=True):
        """
        Save only safe numeric nonbonded data to a .npz file.
        Skips IOData, grids, Horton objects, and all non-serializable fields.
        """

        # strict whitelist of attributes we know are numeric and serializable
        numeric_attrs = [
            "atnums",
            "atcoords",
            "atcharges",
            "ref_atvolumes",
            "volume_ratios",
            "c6s_eff",
            "a_eff",
            "sigma",
            "epsilon",
        ]

        save_dict = {}

        for name in numeric_attrs:
            if hasattr(self, name):
                val = getattr(self, name)

                # convert lists to numpy arrays
                if isinstance(val, list):
                    val = np.array(val)

                # accept only numpy arrays, floats, ints
                if isinstance(val, (np.ndarray, float, int)):
                    save_dict[name] = val
                    if verbose:
                        print(f"[OK] saved: {name}")
                else:
                    if verbose:
                        print(f"[SKIP] non-numeric attribute: {name} ({type(val)})")
            else:
                if verbose:
                    print(f"[MISSING] attribute not found: {name}")

        # write NPZ
        np.savez_compressed(filename, **save_dict)

        if verbose:
            print(f"\n[OK] Nonbonded parameters saved to {filename}\n")



