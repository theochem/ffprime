import numpy as np
import pytest
import os
import sys

from ffprime.bond import Bonded

@pytest.fixture(scope="module")
def bonded_job():
    """Fixture to initialize a Bonded object with example data."""
    # Paths are relative to ffprime directory
    log_path = os.path.join("examples", "lig.log")
    fchk_path = os.path.join("examples", "lig.fchk")
    return Bonded(log_path=log_path, fchk_path=fchk_path)

def test_bonded_init(bonded_job):
    """Test that the Bonded class initializes correctly and parses data."""
    assert len(bonded_job.bonds) == 16
    assert len(bonded_job.angles) == 26
    assert bonded_job.mol is not None
    assert hasattr(bonded_job, "hess_new")
    assert bonded_job.hess_new.shape == (51, 51)

def test_get_force_constant(bonded_job):
    """Test the bond force constant calculation."""
    k_ij = bonded_job.get_force_constant(0, 1)
    assert isinstance(k_ij, (float, np.float64))
    assert 100.0 < k_ij < 1000.0

def test_get_angle_force_constant_accuracy(bonded_job):
    """Test the angle force constant calculation against reference."""
    i, j, k = 1, 0, 12
    k_theta = bonded_job.get_angle_force_constant(i, j, k)
    expected = 14.696188
    assert np.isclose(k_theta, expected, atol=1e-4)

def test_get_angle_force_constant_range(bonded_job):
    """Ensure all calculated angles are within range."""
    for angle in bonded_job.angles:
        i, j, k = int(angle[0]), int(angle[1]), int(angle[2])
        k_theta = bonded_job.get_angle_force_constant(i, j, k)
        assert 5.0 < k_theta < 100.0
