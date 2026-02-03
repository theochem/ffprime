import pytest
from pathlib import Path
from iodata import load_one

TEST_DATA = Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def lig_fchk():
    return TEST_DATA / "lig.fchk"

@pytest.fixture(scope="session")
def lig_molden():
    path = TEST_DATA / "orca.molden.input"
    if not path.exists():
    	print("ORCA TEST SKIPPED: THERE IS NO ORCA FILES IN DATA FOLDER")
    	pytest.skip("orca.molden.input not found")
    return path


#@pytest.fixture(scope="session")
#def h2o_fchk():
#    return TEST_DATA / "h2o.fchk"

@pytest.fixture(scope="session")
def lig_mol(lig_fchk):
    return load_one(lig_fchk)

@pytest.fixture(scope="session")
def schemes():
    return ["mbis"]


@pytest.fixture(scope="session")
def pro_level():
    return "ub3lyp_def2tzvpd"

