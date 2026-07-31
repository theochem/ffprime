"""
NOTE: imports of Cartesian helpers below moved from
``ffprime.electrostatics.multipole`` to ``ffprime.electrostatics.cartesian``
as part of the Cartesian/spherical architecture split. Test bodies and
assertions are unchanged.
"""

import numpy as np
import pytest

from ffprime.electrostatics.spherical import (
    dipole_cartesian_to_spherical,
    dipole_spherical_to_cartesian,
    quadrupole_cartesian_to_spherical,
    quadrupole_spherical_to_cartesian,
    spherical_monopole_field,
    spherical_dipole_field,
    spherical_total_potential,
    spherical_total_field,
)
from ffprime.electrostatics.cartesian import monopole_field
from ffprime.electrostatics.cartesian import monopole_potential
from ffprime.electrostatics.cartesian import dipole_potential, dipole_field
from ffprime.electrostatics.spherical import spherical_dipole_potential
from ffprime.electrostatics.cartesian import quadrupole_potential, quadrupole_field
from ffprime.electrostatics.spherical import (
    spherical_quadrupole_potential,
    spherical_quadrupole_field,
)

SQRT3 = np.sqrt(3.0)


# ---------------------------------------------------------------------------
# Monopole field
# ---------------------------------------------------------------------------

def test_spherical_monopole_field_matches_cartesian():
    charges = np.array([1.0])
    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    e_cart = monopole_field(
        charges,
        coords,
        points,
    )
    e_sph = spherical_monopole_field(
        charges,
        coords,
        points,
    )

    assert np.allclose(e_cart, e_sph)


def test_monopole_field_along_x():
    """
    Unit charge at origin, point at (2, 0, 0).

    E = q * r / r³ = 1 * (2, 0, 0) / 8 = (0.25, 0, 0)
    """
    charges = np.array([1.0])
    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[2.0, 0.0, 0.0]])

    result = spherical_monopole_field(
        charges,
        coords,
        points,
    )

    assert np.allclose(result, [[0.25, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# Dipole
# ---------------------------------------------------------------------------

def test_dipole_z_aligned():
    """Pure z-dipole maps entirely to Q_10, others zero."""
    p = np.array([0.0, 0.0, 1.0])
    sph = dipole_cartesian_to_spherical(p)

    assert np.isclose(sph[0], 1.0)
    assert np.isclose(sph[1], 0.0)
    assert np.isclose(sph[2], 0.0)


def test_dipole_roundtrip():
    """Cartesian -> spherical -> Cartesian recovers original."""
    p = np.array([1.5, -2.0, 3.0])
    assert np.allclose(
        dipole_spherical_to_cartesian(dipole_cartesian_to_spherical(p)), p
    )


def test_dipole_potential_consistency():
    q = np.array([1.0, 2.0, 3.0])
    cart = dipole_spherical_to_cartesian(q)

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    v_cart = dipole_potential(
        np.array([cart]),
        coords,
        points,
    )
    v_sph = spherical_dipole_potential(
        np.array([q]),
        coords,
        points,
    )

    assert np.allclose(v_cart, v_sph)


def test_dipole_potential_along_z():
    """
    Pure Q10 dipole evaluated on the z-axis.

    Q10 = 1, r = (0, 0, 2)
    V = z / r³ = 2 / 8 = 0.25
    """
    dipoles = np.array([[1.0, 0.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    result = spherical_dipole_potential(
        dipoles,
        coords,
        points,
    )

    assert np.allclose(result, [0.25])


def test_spherical_dipole_field_matches_cartesian():
    q = np.array([1.0, 2.0, 3.0])
    cart = dipole_spherical_to_cartesian(q)

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    e_cart = dipole_field(
        np.array([cart]),
        coords,
        points,
    )
    e_sph = spherical_dipole_field(
        np.array([q]),
        coords,
        points,
    )

    assert np.allclose(e_cart, e_sph)


def test_dipole_field_along_z():
    """
    Pure Q10 dipole evaluated on the z-axis.

    Q10 = 1, r = (0, 0, 2)
    E = [3(p.r̂)r̂ - p] / r³ = [3*(0,0,1) - (0,0,1)] / 8 = (0, 0, 0.25)
    """
    dipoles = np.array([[1.0, 0.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    result = spherical_dipole_field(
        dipoles,
        coords,
        points,
    )

    assert np.allclose(result, [[0.0, 0.0, 0.25]])


# ---------------------------------------------------------------------------
# Quadrupole
# ---------------------------------------------------------------------------

def test_quadrupole_traceless_check():
    """Non-traceless tensor raises ValueError."""
    bad_tensor = np.eye(3)
    with pytest.raises(ValueError, match="traceless"):
        quadrupole_cartesian_to_spherical(bad_tensor)


def test_quadrupole_z_axial():
    """Axially symmetric quadrupole along z has only Q_20 nonzero."""
    theta = np.array([
        [-0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    sph = quadrupole_cartesian_to_spherical(theta)

    assert np.isclose(sph[0], 1.0)
    assert np.allclose(sph[1:], 0.0)


def test_quadrupole_roundtrip():
    """Cartesian -> spherical -> Cartesian recovers original tensor."""
    theta = np.array([
        [-0.5, 0.3, 0.1],
        [0.3, -0.5, 0.2],
        [0.1, 0.2, 1.0]
    ])
    recovered = quadrupole_spherical_to_cartesian(
        quadrupole_cartesian_to_spherical(theta)
    )

    assert np.allclose(recovered, theta)


def test_quadrupole_potential_consistency():
    q = np.array([1.0, 0.3, -0.2, 0.4, 0.1])
    theta = quadrupole_spherical_to_cartesian(q)

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    v_cart = quadrupole_potential(
        np.array([theta]),
        coords,
        points,
    )
    v_sph = spherical_quadrupole_potential(
        np.array([q]),
        coords,
        points,
    )

    assert np.allclose(v_cart, v_sph)


def test_quadrupole_potential_along_z():
    """
    Pure Q20 quadrupole evaluated on the z-axis.

    Q20 = 1, r = (0, 0, 2)
    V = (z² - 0.5(x² + y²)) / r⁵ = 4 / 32 = 0.125
    """
    quadrupoles = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    result = spherical_quadrupole_potential(
        quadrupoles,
        coords,
        points,
    )

    assert np.allclose(result, [0.125])


def test_spherical_quadrupole_field_matches_cartesian():
    q = np.array([1.0, 0.3, -0.2, 0.4, 0.1])
    theta = quadrupole_spherical_to_cartesian(q)

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    e_cart = quadrupole_field(
        np.array([theta]),
        coords,
        points,
    )
    e_sph = spherical_quadrupole_field(
        np.array([q]),
        coords,
        points,
    )

    assert np.allclose(e_cart, e_sph)


def test_quadrupole_field_along_z():
    """
    Pure Q20 quadrupole evaluated on the z-axis.

    Q20 = 1, r = (0, 0, 2)

    The corresponding Cartesian tensor is Θzz = 1, Θxx = Θyy = -0.5
    (the same traceless axial tensor used in test_quadrupole_z_axial /
    test_quadrupole_potential_along_z). Evaluating the existing
    Cartesian quadrupole_field implementation for this tensor at
    (0, 0, 2) gives the expected field directly, which is then checked
    against spherical_quadrupole_field's result.

    Analytical check (E = 5(Θ:rr)r/r^7 - 2(Θ·r)/r^5, see
    ``cartesian.quadrupole_field``): Θ:rr = Θzz*z^2 = 4, r = 2, so
    term1 = 5*4*(0,0,2)/128 = (0,0,0.3125); Θ·r = (0,0,Θzz*2) = (0,0,2),
    so term2 = 2*(0,0,2)/32 = (0,0,0.125); E = term1 - term2 =
    (0,0,0.1875). NOTE: this literal was previously (0,0,0.25), which
    was only correct under the old, buggy ``quadrupole_field``
    implementation that was missing the factor of 2 on the Θ·r term
    (unrelated to the Stone-convention normalization); it has been
    corrected here to match the fixed physics.
    """
    quadrupoles = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    # Ground truth computed directly from the existing Cartesian
    # implementation, for the equivalent tensor.
    theta = np.array([
        [-0.5, 0.0, 0.0],
        [0.0, -0.5, 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected = quadrupole_field(
        np.array([theta]),
        coords,
        points,
    )

    result = spherical_quadrupole_field(
        quadrupoles,
        coords,
        points,
    )

    assert np.allclose(result, expected)
    assert np.allclose(result, [[0.0, 0.0, 0.1875]])


# ---------------------------------------------------------------------------
# Quadrupole -- Stone-convention regression tests (hand-derived, NOT
# round-trip-only). These pin down the exact numeric normalization
# (the Racah sqrt(3) factors on the m=+-1/+-2 components) so that a
# future regression that reintroduces the old, incorrect normalization
# is caught even though the round-trip and internal-consistency tests
# above cannot detect it (a normalization error applied consistently
# to both the forward and inverse conversion is invisible to a
# round-trip check).
# ---------------------------------------------------------------------------

def test_quadrupole_cartesian_to_spherical_hand_computed():
    """Explicit tensor with all off-diagonal components nonzero, checked
    against the literal Stone equations:

        Q20  = Theta_zz
        Q21c = (2/sqrt(3)) Theta_xz
        Q21s = (2/sqrt(3)) Theta_yz
        Q22c = (Theta_xx - Theta_yy) / sqrt(3)
        Q22s = (2/sqrt(3)) Theta_xy
    """
    theta = np.array([
        [1.0, 0.5, 0.3],
        [0.5, -2.0, 0.7],
        [0.3, 0.7, 1.0],
    ])
    assert np.isclose(np.trace(theta), 0.0)

    q = quadrupole_cartesian_to_spherical(theta)

    expected = np.array([
        theta[2, 2],
        (2.0 / SQRT3) * theta[0, 2],
        (2.0 / SQRT3) * theta[1, 2],
        (theta[0, 0] - theta[1, 1]) / SQRT3,
        (2.0 / SQRT3) * theta[0, 1],
    ])
    np.testing.assert_allclose(q, expected, atol=1e-12)

    # Literal numeric values for this tensor (independent sanity pin).
    np.testing.assert_allclose(
        q,
        [
            1.0,
            0.34641016151377546,
            0.8082903768654762,
            1.7320508075688772,
            0.5773502691896257,
        ],
        atol=1e-12,
    )


def test_quadrupole_spherical_to_cartesian_hand_computed():
    """Pick spherical components directly and verify the reconstructed
    tensor against the hand-derived inverse Stone relations:

        Theta_zz = Q20
        Theta_xz = (sqrt(3)/2) Q21c
        Theta_yz = (sqrt(3)/2) Q21s
        Theta_xy = (sqrt(3)/2) Q22s
        Theta_xx = -Q20/2 + (sqrt(3)/2) Q22c
        Theta_yy = -Q20/2 - (sqrt(3)/2) Q22c
    """
    Q20, Q21c, Q21s, Q22c, Q22s = 0.8, -1.2, 0.4, 0.6, -0.9
    theta = quadrupole_spherical_to_cartesian(
        np.array([Q20, Q21c, Q21s, Q22c, Q22s])
    )

    expected_xx = -Q20 / 2.0 + (SQRT3 / 2.0) * Q22c
    expected_yy = -Q20 / 2.0 - (SQRT3 / 2.0) * Q22c
    expected_zz = Q20
    expected_xy = (SQRT3 / 2.0) * Q22s
    expected_xz = (SQRT3 / 2.0) * Q21c
    expected_yz = (SQRT3 / 2.0) * Q21s

    np.testing.assert_allclose(theta[0, 0], expected_xx, atol=1e-12)
    np.testing.assert_allclose(theta[1, 1], expected_yy, atol=1e-12)
    np.testing.assert_allclose(theta[2, 2], expected_zz, atol=1e-12)
    np.testing.assert_allclose(theta[0, 1], expected_xy, atol=1e-12)
    np.testing.assert_allclose(theta[0, 2], expected_xz, atol=1e-12)
    np.testing.assert_allclose(theta[1, 2], expected_yz, atol=1e-12)
    np.testing.assert_allclose(theta, theta.T, atol=1e-12)
    np.testing.assert_allclose(np.trace(theta), 0.0, atol=1e-12)


def _random_traceless_symmetric(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2, 2, size=(3, 3))
    a = (a + a.T) / 2.0
    trace = np.trace(a)
    a[0, 0] -= trace / 3.0
    a[1, 1] -= trace / 3.0
    a[2, 2] -= trace / 3.0
    return a


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_quadrupole_roundtrip_random_tensors_with_all_off_diagonals(seed):
    """Round-trip check over random traceless tensors with non-zero xy,
    xz, and yz components (the earlier ``test_quadrupole_roundtrip``
    only exercises one fixed tensor)."""
    theta_in = _random_traceless_symmetric(seed)
    q = quadrupole_cartesian_to_spherical(theta_in)
    theta_out = quadrupole_spherical_to_cartesian(q)
    np.testing.assert_allclose(theta_out, theta_in, atol=1e-10)


def test_spherical_quadrupole_potential_matches_cartesian_off_diagonal():
    """Cross-check spherical_quadrupole_potential's own closed-form
    Q_lm formula (not just quadrupole_spherical_to_cartesian) against
    the independent Cartesian Theta_ab r_a r_b / r^5 implementation,
    for tensors with non-zero xy/xz/yz components. This is the test
    that catches a mismatched Racah factor in the potential formula
    itself (as opposed to just in the conversion routines).
    """
    rng = np.random.default_rng(42)
    thetas = np.array([_random_traceless_symmetric(s) for s in range(4)])
    coords = rng.uniform(-1, 1, size=(4, 3))
    points = rng.uniform(3, 5, size=(6, 3))

    quads_sph = np.array(
        [quadrupole_cartesian_to_spherical(t) for t in thetas]
    )

    v_spherical = spherical_quadrupole_potential(quads_sph, coords, points)
    v_cartesian = quadrupole_potential(thetas, coords, points)

    np.testing.assert_allclose(v_spherical, v_cartesian, atol=1e-10)


# ---------------------------------------------------------------------------
# Total potential / field (superposition of all multipole orders)
# ---------------------------------------------------------------------------

def test_spherical_total_potential_monopole_only():
    charges = np.array([1.0])

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 0.0, 0.0]])

    expected = monopole_potential(
        charges,
        coords,
        points,
    )

    result = spherical_total_potential(
        coords,
        points,
        charges=charges,
    )

    assert np.allclose(result, expected)


def test_spherical_total_potential_superposition():
    charges = np.array([1.0])
    dipoles = np.array([[1.0, 0.0, 0.0]])
    quadrupoles = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    expected = (
        monopole_potential(charges, coords, points)
        + spherical_dipole_potential(dipoles, coords, points)
        + spherical_quadrupole_potential(quadrupoles, coords, points)
    )

    result = spherical_total_potential(
        coords,
        points,
        charges=charges,
        dipoles=dipoles,
        quadrupoles=quadrupoles,
    )

    assert np.allclose(result, expected)


def test_spherical_total_field_monopole_only():
    charges = np.array([1.0])

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 0.0, 0.0]])

    expected = monopole_field(
        charges,
        coords,
        points,
    )

    result = spherical_total_field(
        coords,
        points,
        charges=charges,
    )

    assert np.allclose(result, expected)


def test_spherical_total_field_superposition():
    charges = np.array([1.0])
    dipoles = np.array([[1.0, 0.0, 0.0]])
    quadrupoles = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])

    coords = np.array([[0.0, 0.0, 0.0]])
    points = np.array([[1.0, 2.0, 3.0]])

    expected = (
        spherical_monopole_field(charges, coords, points)
        + spherical_dipole_field(dipoles, coords, points)
        + spherical_quadrupole_field(quadrupoles, coords, points)
    )

    result = spherical_total_field(
        coords,
        points,
        charges=charges,
        dipoles=dipoles,
        quadrupoles=quadrupoles,
    )

    assert np.allclose(result, expected)