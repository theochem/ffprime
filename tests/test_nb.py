import numpy as np
import pytest
from ffprime.nb import Nonbonded


# ---------------------------------------------------------------------
# Derived fixtures (avoid recomputation)
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def nb_from_iodata(lig_mol, schemes, pro_level):
    return Nonbonded.from_IODATA(
        lig_mol,
        schemes=schemes,
        args="insane",
        pro_level=pro_level,
    )


@pytest.fixture(scope="session")
def nb_from_file(lig_fchk, schemes, pro_level):
    return Nonbonded.from_file(
        fname=str(lig_fchk),
        schemes=schemes,
        args="insane",
        pro_level=pro_level,
    )

@pytest.fixture(scope="session")
def nb_from_file(lig_molden, schemes, pro_level):
    return Nonbonded.from_file(
        fname=str(lig_molden),
        schemes=schemes,
        args="insane",
        pro_level=pro_level,
    )

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_from_file_creates_nonbonded(nb_from_file):
    nb = nb_from_file

    assert isinstance(nb, Nonbonded)
    assert hasattr(nb, "sigma")
    assert hasattr(nb, "epsilon")
    assert hasattr(nb, "atcharges")

    n = len(nb.atnums)

    assert nb.sigma.shape == (n,)
    assert nb.epsilon.shape == (n,)
    assert nb.atcharges.shape == (n,)


def test_from_iodata_populates_atffparams_and_extra(nb_from_iodata):
    mol = nb_from_iodata

    assert hasattr(mol, "atffparams")
    assert hasattr(mol, "extra")

    # FF parameters
    for key in ["atcharges", "sigma", "epsilon"]:
        assert key in mol.atffparams
        assert isinstance(mol.atffparams[key], np.ndarray)

    # Extra parameters
    for key in ["ref_atvolumes", "volume_ratios", "c6s_eff", "a_eff"]:
        assert key in mol.extra
        assert isinstance(mol.extra[key], np.ndarray)

    # Ensure no duplication
    assert "atnums" not in mol.extra
    assert "atcoords" not in mol.extra


def test_array_lengths_consistent(nb_from_iodata):
    mol = nb_from_iodata
    n = len(mol.atnums)

    for arr in mol.atffparams.values():
        assert arr.shape == (n,)

    for arr in mol.extra.values():
        if arr.ndim == 1:
            assert arr.shape == (n,)


def test_physical_sanity(nb_from_iodata):
    mol = nb_from_iodata

    sigma = mol.atffparams["sigma"]
    epsilon = mol.atffparams["epsilon"]
    charges = mol.atffparams["atcharges"]

    # sigma > 0
    assert (sigma > 0).all()

    # epsilon > 0
    assert (epsilon > 0).all()

    # charges should be finite
    assert np.isfinite(charges).all()

    # total charge should be reasonable (not exact)
    total_charge = charges.sum()
    assert abs(total_charge) < 2.0


def test_volume_ratios_consistent(nb_from_iodata):
    vols = nb_from_iodata.extra["volume_ratios"]

    assert (vols > 0).all()
    assert (vols < 10).all()   # very loose upper bound


def test_write_npz(tmp_path, nb_from_file):
    out = tmp_path / "nb.npz"
    nb_from_file.write_npz(out, verbose=False)

    data = np.load(out)

    for key in ["sigma", "epsilon", "atcharges"]:
        assert key in data


