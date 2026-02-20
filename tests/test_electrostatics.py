import numpy as np
from ffprime.electrostatics.multipole import (
    monopole_potential,
    monopole_field,
)


def test_monopole_potential_unit_distance():
    q = 1.0
    r_vec = np.array([1.0, 0.0, 0.0])
    V = monopole_potential(q, r_vec)
    assert np.isclose(V, 1.0)


def test_monopole_field_direction():
    q = 1.0
    r_vec = np.array([1.0, 0.0, 0.0])
    E = monopole_field(q, r_vec)
    assert np.allclose(E, np.array([1.0, 0.0, 0.0]))


def test_zero_distance_safe():
    q = 1.0
    r_vec = np.array([0.0, 0.0, 0.0])
    V = monopole_potential(q, r_vec)
    E = monopole_field(q, r_vec)
    assert V == 0.0
    assert np.allclose(E, np.zeros(3))