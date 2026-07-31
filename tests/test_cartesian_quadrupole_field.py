"""
Regression tests for the ``E = -grad V`` bug fix in
:func:`ffprime.electrostatics.cartesian.quadrupole_field`.

Background
----------
The quadrupole potential is

    V(r) = Theta_ab r_a r_b / r^5

Differentiating (Theta symmetric, traceless) gives

    E = -grad V = 5 (Theta:rr) r / r^7 - 2 (Theta . r) / r^5

The previous implementation omitted the factor of 2 on the second term
(effectively computing ``Theta . r / r^5`` instead of
``2 * Theta . r / r^5``), which is a pure Cartesian differentiation
error -- unrelated to the Stone-convention (Racah-normalized spherical
quadrupole) migration. This file pins the corrected physics down with:

  * a symbolic derivation of grad V (skipped if sympy is unavailable),
  * finite-difference checks of E = -grad V for axial and fully
    off-diagonal tensors,
  * hand-evaluated analytical field values for axial and off-diagonal
    tensors.

Every test below is written so that it FAILS against the old,
factor-of-2-short implementation and PASSES against the corrected one.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ffprime.electrostatics.cartesian import quadrupole_potential, quadrupole_field


# ---------------------------------------------------------------------------
# 1. Symbolic derivation (skipped if sympy is not installed -- this repo
#    does not depend on sympy at runtime, only for this verification test).
# ---------------------------------------------------------------------------

def test_symbolic_derivation_of_quadrupole_field():
    """
    Derive E = -grad V symbolically from V = Theta_ab r_a r_b / r^5 for a
    fully general symmetric tensor, and confirm it agrees with
    ``5*Qrr*r/r^7 - 2*Qr/r^5`` (the corrected formula) but NOT with
    ``5*Qrr*r/r^7 - Qr/r^5`` (the old, buggy formula).
    """
    sympy = pytest.importorskip("sympy")
    import sympy as sp

    x, y, z = sp.symbols("x y z", real=True)
    Txx, Tyy, Tzz, Txy, Txz, Tyz = sp.symbols(
        "Txx Tyy Tzz Txy Txz Tyz", real=True
    )

    r_vec = sp.Matrix([x, y, z])
    Theta = sp.Matrix(
        [
            [Txx, Txy, Txz],
            [Txy, Tyy, Tyz],
            [Txz, Tyz, Tzz],
        ]
    )
    r = sp.sqrt(x**2 + y**2 + z**2)

    Qrr = (r_vec.T * Theta * r_vec)[0, 0]
    V = Qrr / r**5

    E_derived = [sp.simplify(-sp.diff(V, v)) for v in (x, y, z)]

    Qr = Theta * r_vec
    E_corrected = [
        sp.simplify(5 * Qrr * r_vec[i] / r**7 - 2 * Qr[i] / r**5)
        for i in range(3)
    ]
    E_buggy = [
        sp.simplify(5 * Qrr * r_vec[i] / r**7 - Qr[i] / r**5)
        for i in range(3)
    ]

    for i in range(3):
        assert sp.simplify(E_derived[i] - E_corrected[i]) == 0
        assert sp.simplify(E_derived[i] - E_buggy[i]) != 0


# ---------------------------------------------------------------------------
# 2. Finite-difference checks: E = -grad V, for both axial and fully
#    off-diagonal tensors.
# ---------------------------------------------------------------------------

def _random_traceless_symmetric(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2, 2, size=(3, 3))
    a = (a + a.T) / 2.0
    trace = np.trace(a)
    a[0, 0] -= trace / 3.0
    a[1, 1] -= trace / 3.0
    a[2, 2] -= trace / 3.0
    return a


def _numerical_gradient_of_potential(theta, atcoords, point, h=1e-6):
    grad = np.zeros(3)
    for i in range(3):
        p_plus = point.copy()
        p_minus = point.copy()
        p_plus[i] += h
        p_minus[i] -= h
        v_plus = quadrupole_potential(theta, atcoords, p_plus.reshape(1, 3))[0]
        v_minus = quadrupole_potential(theta, atcoords, p_minus.reshape(1, 3))[0]
        grad[i] = (v_plus - v_minus) / (2.0 * h)
    return grad


def test_field_matches_gradient_axial_tensor():
    """Axial (diagonal) tensor: E must equal -grad V by finite differences."""
    theta = np.array([[[-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 2.0]]])
    atcoords = np.array([[0.0, 0.0, 0.0]])
    point = np.array([1.3, -0.8, 0.6])

    analytic = quadrupole_field(theta, atcoords, point.reshape(1, 3))[0]
    numerical = -_numerical_gradient_of_potential(theta, atcoords, point)

    assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)


def test_field_matches_gradient_off_diagonal_tensor():
    """Tensor with nonzero xy, xz, yz: E must equal -grad V by finite
    differences. This is the case the old (factor-of-2-short)
    implementation gets wrong along every axis, not just on-axis."""
    theta = np.array([[[0.6, 0.4, -0.3],
                        [0.4, -0.9, 0.2],
                        [-0.3, 0.2, 0.3]]])
    assert np.isclose(np.trace(theta[0]), 0.0)

    atcoords = np.array([[0.0, 0.0, 0.0]])
    point = np.array([0.9, 1.4, -1.1])

    analytic = quadrupole_field(theta, atcoords, point.reshape(1, 3))[0]
    numerical = -_numerical_gradient_of_potential(theta, atcoords, point)

    assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_field_matches_gradient_randomized_off_diagonal(seed):
    """Randomized traceless symmetric tensors (guaranteed nonzero
    off-diagonal components with high probability) and random field
    points: E = -grad V must hold for every trial."""
    theta = _random_traceless_symmetric(seed)
    atcoords = np.array([[0.0, 0.0, 0.0]])

    rng = np.random.default_rng(100 + seed)
    while True:
        point = rng.uniform(-3.0, 3.0, size=3)
        if np.linalg.norm(point) > 0.5:
            break

    analytic = quadrupole_field(theta[np.newaxis], atcoords, point.reshape(1, 3))[0]
    numerical = -_numerical_gradient_of_potential(
        theta[np.newaxis], atcoords, point
    )

    assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. Hand-evaluated analytical values (axial and off-diagonal), computed
#    directly from E = 5(Theta:rr)r/r^7 - 2(Theta.r)/r^5.
# ---------------------------------------------------------------------------

def test_analytical_field_axial_on_axis():
    """
    Theta = diag(-1, -1, 2), point = (0, 0, 2).

    Theta:rr = 2*4 = 8, r = 2, r^5 = 32, r^7 = 128.
    term1 = 5*8*(0,0,2)/128 = (0, 0, 0.625)
    Theta.r = (0, 0, 2*2) = (0, 0, 4)
    term2 = 2*(0,0,4)/32 = (0, 0, 0.25)
    E = term1 - term2 = (0, 0, 0.375)
    """
    theta = np.array([[[-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 2.0]]])
    atcoords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    result = quadrupole_field(theta, atcoords, points)
    assert_allclose(result, [[0.0, 0.0, 0.375]], rtol=1e-10)


def test_analytical_field_axial_equatorial():
    """
    Theta = diag(-1, -1, 2), point = (2, 0, 0).

    Theta:rr = -1*4 = -4, r = 2, r^5 = 32, r^7 = 128.
    term1 = 5*(-4)*(2,0,0)/128 = (-0.3125, 0, 0)
    Theta.r = (-1*2, 0, 0) = (-2, 0, 0)
    term2 = 2*(-2,0,0)/32 = (-0.125, 0, 0)
    E = term1 - term2 = (-0.1875, 0, 0)
    """
    theta = np.array([[[-1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 2.0]]])
    atcoords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[2.0, 0.0, 0.0]])

    result = quadrupole_field(theta, atcoords, points)
    assert_allclose(result, [[-0.1875, 0.0, 0.0]], rtol=1e-10)


def test_analytical_field_off_diagonal():
    """
    Theta with only Txy = 1 nonzero (traceless, symmetric):

        Theta = [[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, 0]]

    point = (1, 1, 0), r = sqrt(2), r^2 = 2.

    Theta:rr = 2*(Txy*x*y) = 2*1*1*1 = 2
    r^5 = 2^(5/2) = 4*sqrt(2), r^7 = 2^(7/2) = 8*sqrt(2)
    term1 = 5*2*(1,1,0)/(8*sqrt(2)) = (10,10,0)/(8*sqrt(2))
    Theta.r = (Txy*y, Txy*x, 0) = (1, 1, 0)
    term2 = 2*(1,1,0)/(4*sqrt(2)) = (2,2,0)/(4*sqrt(2))

    E = term1 - term2
      = (10/(8*sqrt(2)) - 2/(4*sqrt(2)), same, 0)
      = (10/(8*sqrt(2)) - 4/(8*sqrt(2)), same, 0)
      = (6/(8*sqrt(2)), 6/(8*sqrt(2)), 0)
      = (3/(4*sqrt(2)), 3/(4*sqrt(2)), 0)
    """
    theta = np.array([[[0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0]]])
    atcoords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 1.0, 0.0]])

    result = quadrupole_field(theta, atcoords, points)
    expected_component = 3.0 / (4.0 * np.sqrt(2.0))
    assert_allclose(
        result, [[expected_component, expected_component, 0.0]], rtol=1e-10
    )


def test_old_buggy_formula_would_fail_this_test():
    """
    Sanity check that this test suite is actually sensitive to the bug:
    reimplement the OLD (factor-of-2-short) formula inline and confirm
    it disagrees with the corrected ``quadrupole_field`` on the same
    axial case used in ``test_analytical_field_axial_on_axis``. This
    guards against a future accidental regression back to the old
    formula going unnoticed.
    """
    theta = np.array([[-1.0, 0.0, 0.0],
                       [0.0, -1.0, 0.0],
                       [0.0, 0.0, 2.0]])
    r_vec = np.array([0.0, 0.0, 2.0])
    r = np.linalg.norm(r_vec)

    Qrr = r_vec @ theta @ r_vec
    Qr = theta @ r_vec

    old_buggy = 5 * Qrr * r_vec / r**7 - Qr / r**5
    corrected = quadrupole_field(
        theta[np.newaxis], np.array([[0.0, 0.0, 0.0]]), r_vec.reshape(1, 3)
    )[0]

    assert not np.allclose(old_buggy, corrected)
    assert_allclose(corrected, [0.0, 0.0, 0.375], rtol=1e-10)
