## FFprime

FFprime is a Python-based toolkit designed to derive bonded and nonbonded parameters for molecular systems based on Atoms-in-Molecules (AIM) partitioning schemes.
It provides an automated framework to extract interaction parameters directly from electronic structure data, analyze atomic and interatomic properties, and validate potential energy models through comparison with reference quantum-mechanical energies.

The code also includes tools to:

Evaluate molecular interaction energies and force-field potentials.

Generate parameter sets consistent with AIM density partitioning (e.g., from HORTON and Denspart tools).

Benchmark derived models against ab initio reference data.

### Install Libraries (only once):

You need to make conda environment and install these libraries on your computer only once.
Afterwards, you just `conda activate env_qcdevs_py310` to restore the environment.

```bash
# create a Python 3 conda environment
conda create -n env_qcdevs python=3.10
conda activate env_qcdevs_py310

# install HORTON2.3
conda install  -c conda-forge -c theochem horton --yes
conda install nose
# run tests (you should get "OK (SKIP=1)")
pip install pytest # otherwise nosetests crashes
nosetests -v horton

 # install IOData, Grid, GBasis, DensPart, AtomDB
 pip install git+https://github.com/theochem/iodata.git
 pip install git+https://github.com/theochem/grid.git
 pip install git+https://github.com/theochem/gbasis.git
 pip install git+https://github.com/theochem/denspart.git
 pip install git+https://github.com/theochem/AtomDB.git

 # install FFprime
 git clone git@github.com:ccastilloo/FFprime.git
 cd FFprime
 pip install -e .
```

### EXAMPLE USAGE

Deriving Nonbonded Parameters from an Electronic Structure File

Below is a minimal example showing how to derive nonbonded parameters (atomic charges, Lennard-Jones σ and ε) directly from an electronic structure calculation (e.g., Gaussian .fchk file) using FFprime.

```python
from ffprime.nb import Nonbonded

# Derive nonbonded parameters from a Gaussian-formatted checkpoint file
mol = Nonbonded.from_file(
    fname="ffprime/examples/lig.fchk",
    schemes=["mbis-horton"],      # AIM partitioning scheme
    args="insane",                # calculation control argument
    pro_level="ub3lyp_def2tzvpd"  # level of theory and basis set
)

# Display derived atomic parameters
print("Atomic charges (MBIS-HORTON):")
print(mol.atcharges)

print("\nLennard-Jones σ parameters (Å):")
print(mol.sigma)

print("\nLennard-Jones ε parameters (kcal/mol):")
print(mol.epsilon)
```