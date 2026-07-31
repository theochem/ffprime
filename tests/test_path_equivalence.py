"""
Path-equivalence tests for the multipole electrostatics engine.

These tests verify that the Cartesian and spherical (Stone-convention)
implementations of the electrostatic potential and electric field are
numerically equivalent for the *same physical multipole moments*. A
Cartesian moment is converted to its spherical form using the existing
conversion routines in :mod:`ffprime.electrostatics.spherical`, and the
two code paths are then evaluated at identical field points and
compared directly with ``assert_allclose``.

No new physics or helper routines are introduced here -- every
potential/field evaluation and every Cartesian <-> spherical conversion
reuses functions that already exist in :mod:`ffprime.electrostatics.cartesian`
and :mod:`ffprime.electrostatics.spherical`.
"""

import numpy as np
from numpy.testing import assert_allclose

from ffprime.electrostatics.cartesian import (
    monopole_field,
    dipole_potential,
    dipole_field,
    quadrupole_potential,
    quadrupole_field,
    total_potential,
    total_field,
)
from ffprime.electrostatics.spherical import (
    dipole_cartesian_to_spherical,
    quadrupole_cartesian_to_spherical,
    spherical_monopole_field,
    spherical_dipole_potential,
    spherical_dipole_field,
    spherical_quadrupole_potential,
    spherical_quadrupole_field,
    spherical_total_potential,
    spherical_total_field,
)


# ─────────────────────────────────────────────
# MONOPOLE PATH EQUIVALENCE
# ─────────────────────────────────────────────

class TestMonopoleEquivalence:
    """
    Monopoles have no orientation, so the Cartesian and spherical code
    paths should agree exactly (up to floating point) for both the
    potential and the field.
    """

    def test_monopole_potential_matches(self):
        """
        Verify that the Cartesian and spherical total potentials agree
        for a monopole-only system (charges, no dipoles/quadrupoles).
        """
        charges = np.array([1.0, -0.5, 2.0])
        atcoords = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]])
        points = np.array([[2.0, 1.0, 0.5],
                            [-1.0, 3.0, 2.0]])

        v_cart = total_potential(atcoords, points, atcharges=charges)
        v_sph = spherical_total_potential(atcoords, points, charges=charges)

        assert_allclose(v_cart, v_sph, rtol=1e-10)

    def test_monopole_field_matches(self):
        """
        Verify that ``monopole_field`` (Cartesian) and
        ``spherical_monopole_field`` agree for the same charges.
        """
        charges = np.array([1.0, -0.5, 2.0])
        atcoords = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]])
        points = np.array([[2.0, 1.0, 0.5],
                            [-1.0, 3.0, 2.0]])

        e_cart = monopole_field(charges, atcoords, points)
        e_sph = spherical_monopole_field(charges, atcoords, points)

        assert_allclose(e_cart, e_sph, rtol=1e-10)

    def test_monopole_total_field_matches(self):
        """
        Verify that the Cartesian and spherical total fields agree for
        a monopole-only system.
        """
        charges = np.array([0.75, 1.25])
        atcoords = np.array([[0.2, -0.1, 0.0],
                              [-0.3, 0.4, 0.1]])
        points = np.array([[1.5, -2.0, 0.3]])

        e_cart = total_field(atcoords, points, atcharges=charges)
        e_sph = spherical_total_field(atcoords, points, charges=charges)

        assert_allclose(e_cart, e_sph, rtol=1e-10)


# ─────────────────────────────────────────────
# DIPOLE PATH EQUIVALENCE
# ─────────────────────────────────────────────

class TestDipoleEquivalence:
    """
    Verify that Cartesian dipoles, converted to spherical form via
    ``dipole_cartesian_to_spherical``, produce identical potentials and
    fields under both code paths.
    """

    def test_dipole_potential_matches_single_site(self):
        """Single dipole: Cartesian and spherical potentials agree."""
        cart_dipoles = np.array([[0.5, -0.3, 1.2]])
        atcoords = np.array([[0.1, 0.2, -0.3]])
        points = np.array([[1.0, 2.0, 3.0],
                            [-1.5, 0.5, 2.0]])

        sph_dipoles = np.array(
            [dipole_cartesian_to_spherical(p) for p in cart_dipoles]
        )

        v_cart = dipole_potential(cart_dipoles, atcoords, points)
        v_sph = spherical_dipole_potential(sph_dipoles, atcoords, points)

        assert_allclose(v_cart, v_sph, rtol=1e-10)

    def test_dipole_field_matches_single_site(self):
        """Single dipole: Cartesian and spherical fields agree."""
        cart_dipoles = np.array([[0.5, -0.3, 1.2]])
        atcoords = np.array([[0.1, 0.2, -0.3]])
        points = np.array([[1.0, 2.0, 3.0],
                            [-1.5, 0.5, 2.0]])

        sph_dipoles = np.array(
            [dipole_cartesian_to_spherical(p) for p in cart_dipoles]
        )

        e_cart = dipole_field(cart_dipoles, atcoords, points)
        e_sph = spherical_dipole_field(sph_dipoles, atcoords, points)

        assert_allclose(e_cart, e_sph, rtol=1e-10)

    def test_dipole_potential_matches_multiple_sites(self):
        """
        Several dipoles at different sites: verify path equivalence
        holds under superposition, not just for a single site.
        """
        cart_dipoles = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.3, -0.4, 0.7]])
        atcoords = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [-0.5, 0.5, 0.5]])
        points = np.array([[2.0, 2.0, 2.0],
                            [-1.0, -1.0, 3.0]])

        sph_dipoles = np.array(
            [dipole_cartesian_to_spherical(p) for p in cart_dipoles]
        )

        v_cart = dipole_potential(cart_dipoles, atcoords, points)
        v_sph = spherical_dipole_potential(sph_dipoles, atcoords, points)

        assert_allclose(v_cart, v_sph, rtol=1e-10)

    def test_dipole_field_matches_multiple_sites(self):
        """
        Several dipoles at different sites: verify field path
        equivalence holds under superposition.
        """
        cart_dipoles = np.array([[1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [0.3, -0.4, 0.7]])
        atcoords = np.array([[0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [-0.5, 0.5, 0.5]])
        points = np.array([[2.0, 2.0, 2.0],
                            [-1.0, -1.0, 3.0]])

        sph_dipoles = np.array(
            [dipole_cartesian_to_spherical(p) for p in cart_dipoles]
        )

        e_cart = dipole_field(cart_dipoles, atcoords, points)
        e_sph = spherical_dipole_field(sph_dipoles, atcoords, points)

        assert_allclose(e_cart, e_sph, rtol=1e-10)


# ─────────────────────────────────────────────
# QUADRUPOLE PATH EQUIVALENCE
# ─────────────────────────────────────────────

class TestQuadrupoleEquivalence:
    """
    Verify that symmetric traceless Cartesian quadrupoles, converted to
    spherical form via ``quadrupole_cartesian_to_spherical``, produce
    identical potentials and fields under both code paths.
    """

    def test_quadrupole_potential_matches_single_site(self):
        """Single quadrupole: Cartesian and spherical potentials agree."""
        cart_quadrupoles = np.array([[[-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0],
                                       [0.0, 0.0, 2.0]]])
        atcoords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 2.0, 3.0],
                            [-2.0, 0.5, 1.0]])

        sph_quadrupoles = np.array(
            [quadrupole_cartesian_to_spherical(q) for q in cart_quadrupoles]
        )

        v_cart = quadrupole_potential(cart_quadrupoles, atcoords, points)
        v_sph = spherical_quadrupole_potential(sph_quadrupoles, atcoords, points)

        assert_allclose(v_cart, v_sph, rtol=1e-10)

    def test_quadrupole_field_matches_single_site(self):
        """Single quadrupole: Cartesian and spherical fields agree."""
        cart_quadrupoles = np.array([[[-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0],
                                       [0.0, 0.0, 2.0]]])
        atcoords = np.array([[0.0, 0.0, 0.0]])
        points = np.array([[1.0, 2.0, 3.0],
                            [-2.0, 0.5, 1.0]])

        sph_quadrupoles = np.array(
            [quadrupole_cartesian_to_spherical(q) for q in cart_quadrupoles]
        )

        e_cart = quadrupole_field(cart_quadrupoles, atcoords, points)
        e_sph = spherical_quadrupole_field(sph_quadrupoles, atcoords, points)

        assert_allclose(e_cart, e_sph, rtol=1e-10)

    def test_quadrupole_potential_matches_multiple_sites(self):
        """
        Several quadrupoles at different sites: verify path
        equivalence holds under superposition.
        """
        cart_quadrupoles = np.array([
            [[-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 2.0]],
            [[0.5, 0.3, 0.1],
             [0.3, -0.8, 0.2],
             [0.1, 0.2, 0.3]],
        ])
        atcoords = np.array([[0.0, 0.0, 0.0],
                              [0.5, -0.5, 0.5]])
        points = np.array([[2.0, 1.0, 0.0],
                            [-1.0, 2.0, 3.0]])

        sph_quadrupoles = np.array(
            [quadrupole_cartesian_to_spherical(q) for q in cart_quadrupoles]
        )

        v_cart = quadrupole_potential(cart_quadrupoles, atcoords, points)
        v_sph = spherical_quadrupole_potential(sph_quadrupoles, atcoords, points)

        assert_allclose(v_cart, v_sph, rtol=1e-10)

    def test_quadrupole_field_matches_multiple_sites(self):
        """
        Several quadrupoles at different sites: verify field path
        equivalence holds under superposition.
        """
        cart_quadrupoles = np.array([
            [[-1.0, 0.0, 0.0],
             [0.0, -1.0, 0.0],
             [0.0, 0.0, 2.0]],
            [[0.5, 0.3, 0.1],
             [0.3, -0.8, 0.2],
             [0.1, 0.2, 0.3]],
        ])
        atcoords = np.array([[0.0, 0.0, 0.0],
                              [0.5, -0.5, 0.5]])
        points = np.array([[2.0, 1.0, 0.0],
                            [-1.0, 2.0, 3.0]])

        sph_quadrupoles = np.array(
            [quadrupole_cartesian_to_spherical(q) for q in cart_quadrupoles]
        )

        e_cart = quadrupole_field(cart_quadrupoles, atcoords, points)
        e_sph = spherical_quadrupole_field(sph_quadrupoles, atcoords, points)

        assert_allclose(e_cart, e_sph, rtol=1e-10)


# ─────────────────────────────────────────────
# RANDOMIZED PATH EQUIVALENCE
# ─────────────────────────────────────────────

def _random_traceless_symmetric_tensor(rng):
    """
    Build a random symmetric traceless 3x3 tensor, matching the pattern
    already used for randomized quadrupole generation in
    ``tests/test_multipole.py``.
    """
    A = rng.uniform(-1.0, 1.0, (3, 3))
    tensor = 0.5 * (A + A.T)
    tensor -= np.eye(3) * np.trace(tensor) / 3
    return tensor


class TestRandomEquivalence:
    """
    Randomized configurations of coordinates, dipoles, quadrupoles, and
    evaluation points, verifying that the Cartesian and spherical
    implementations agree to machine precision across many independent
    trials. No expected numerical values are hardcoded -- the two code
    paths are always compared directly against each other.
    """

    def test_random_dipole_configurations(self):
        """
        Randomized dipole-only systems: Cartesian and spherical
        potentials and fields must agree for every trial.
        """
        rng = np.random.default_rng(0)

        for _ in range(20):
            n_sites = rng.integers(1, 4)
            cart_dipoles = rng.uniform(-2.0, 2.0, size=(n_sites, 3))
            atcoords = rng.uniform(-1.0, 1.0, size=(n_sites, 3))
            n_points = rng.integers(1, 4)
            points = rng.uniform(-3.0, 3.0, size=(n_points, 3))

            sph_dipoles = np.array(
                [dipole_cartesian_to_spherical(p) for p in cart_dipoles]
            )

            v_cart = dipole_potential(cart_dipoles, atcoords, points)
            v_sph = spherical_dipole_potential(sph_dipoles, atcoords, points)
            assert_allclose(v_cart, v_sph, rtol=1e-10)

            e_cart = dipole_field(cart_dipoles, atcoords, points)
            e_sph = spherical_dipole_field(sph_dipoles, atcoords, points)
            assert_allclose(e_cart, e_sph, rtol=1e-10)

    def test_random_quadrupole_configurations(self):
        """
        Randomized quadrupole-only systems (symmetric, traceless):
        Cartesian and spherical potentials and fields must agree for
        every trial.
        """
        rng = np.random.default_rng(1)

        for _ in range(20):
            n_sites = rng.integers(1, 4)
            cart_quadrupoles = np.array(
                [_random_traceless_symmetric_tensor(rng) for _ in range(n_sites)]
            )
            atcoords = rng.uniform(-1.0, 1.0, size=(n_sites, 3))
            n_points = rng.integers(1, 4)
            points = rng.uniform(-3.0, 3.0, size=(n_points, 3))

            sph_quadrupoles = np.array(
                [quadrupole_cartesian_to_spherical(q) for q in cart_quadrupoles]
            )

            v_cart = quadrupole_potential(cart_quadrupoles, atcoords, points)
            v_sph = spherical_quadrupole_potential(sph_quadrupoles, atcoords, points)
            assert_allclose(v_cart, v_sph, rtol=1e-10)

            e_cart = quadrupole_field(cart_quadrupoles, atcoords, points)
            e_sph = spherical_quadrupole_field(sph_quadrupoles, atcoords, points)
            assert_allclose(e_cart, e_sph, rtol=1e-10)

    def test_random_full_system_total_potential_and_field(self):
        """
        Randomized full systems combining charges, dipoles, and
        quadrupoles: the Cartesian ``total_potential``/``total_field``
        and the spherical ``spherical_total_potential``/
        ``spherical_total_field`` must agree for every trial, verifying
        equivalence under superposition of all multipole orders at
        once.
        """
        rng = np.random.default_rng(2)

        for _ in range(15):
            n_sites = rng.integers(1, 4)
            atcoords = rng.uniform(-1.0, 1.0, size=(n_sites, 3))
            charges = rng.uniform(-2.0, 2.0, size=n_sites)
            cart_dipoles = rng.uniform(-2.0, 2.0, size=(n_sites, 3))
            cart_quadrupoles = np.array(
                [_random_traceless_symmetric_tensor(rng) for _ in range(n_sites)]
            )
            n_points = rng.integers(1, 4)
            points = rng.uniform(-3.0, 3.0, size=(n_points, 3))

            sph_dipoles = np.array(
                [dipole_cartesian_to_spherical(p) for p in cart_dipoles]
            )
            sph_quadrupoles = np.array(
                [quadrupole_cartesian_to_spherical(q) for q in cart_quadrupoles]
            )

            v_cart = total_potential(
                atcoords,
                points,
                atcharges=charges,
                dipoles=cart_dipoles,
                quadrupoles=cart_quadrupoles,
            )
            v_sph = spherical_total_potential(
                atcoords,
                points,
                charges=charges,
                dipoles=sph_dipoles,
                quadrupoles=sph_quadrupoles,
            )
            assert_allclose(v_cart, v_sph, rtol=1e-10)

            e_cart = total_field(
                atcoords,
                points,
                atcharges=charges,
                dipoles=cart_dipoles,
                quadrupoles=cart_quadrupoles,
            )
            e_sph = spherical_total_field(
                atcoords,
                points,
                charges=charges,
                dipoles=sph_dipoles,
                quadrupoles=sph_quadrupoles,
            )
            assert_allclose(e_cart, e_sph, rtol=1e-10)