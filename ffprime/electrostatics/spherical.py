"""Spherical multipole conversions and electrostatic potential functions.

Conversion between Cartesian and real spherical (Stone convention) forms
for dipole and quadrupole moments, plus routines to evaluate the
electrostatic potential and electric field from collections of spherical
multipoles.

This module contains only spherical-representation routines (the
conversions, and the potential/field formulas expressed directly in
terms of spherical components). Pure Cartesian routines live in
:mod:`ffprime.electrostatics.cartesian`; shared validation/geometry
helpers live in :mod:`ffprime.electrostatics.utils`.

For electric *fields*, where there's no extra cost or accuracy lost in
doing so, the spherical routines here convert to the equivalent
Cartesian moment and reuse the existing, already-tested Cartesian field
implementation rather than re-deriving the formula -- this applies to
``spherical_monopole_field`` (trivially -- a point charge has no
orientation), ``spherical_dipole_field``, and
``spherical_quadrupole_field``. Potentials keep their own
spherical-component formulas, since they're already expressed directly
in terms of Q_lm without needing a detour through Cartesian tensors.

Stone convention:

Dipole components:

    Q_10  = pz
    Q_11c = px
    Q_11s = py

Quadrupole components (real spherical, Racah-normalized off-diagonal
terms -- Stone, "The Theory of Intermolecular Forces", eq. 3.xx):

    Q_20  = Theta_zz
    Q_21c = (2/sqrt(3)) * Theta_xz
    Q_21s = (2/sqrt(3)) * Theta_yz
    Q_22c = (Theta_xx - Theta_yy) / sqrt(3)
    Q_22s = (2/sqrt(3)) * Theta_xy

with inverse (using tracelessness, Theta_xx + Theta_yy + Theta_zz = 0):

    Theta_zz = Q_20
    Theta_xz = (sqrt(3)/2) * Q_21c
    Theta_yz = (sqrt(3)/2) * Q_21s
    Theta_xy = (sqrt(3)/2) * Q_22s
    Theta_xx = -Q_20/2 + (sqrt(3)/2) * Q_22c
    Theta_yy = -Q_20/2 - (sqrt(3)/2) * Q_22c

References
----------
Stone, A.J. "The Theory of Intermolecular Forces", Oxford University Press.
"""

import numpy as np

from ffprime.electrostatics.cartesian import (
    dipole_field,
    monopole_field,
    monopole_potential,
    quadrupole_field,
)
from ffprime.electrostatics.utils import compute_displacement, validate_shapes

_SQRT3 = np.sqrt(3.0)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

def dipole_cartesian_to_spherical(p: np.ndarray) -> np.ndarray:
    """Convert a Cartesian dipole moment to real spherical form.

    Uses the Stone convention:

        Q_10  = pz
        Q_11c = px
        Q_11s = py
    """
    p = np.asarray(p)

    if p.shape != (3,):
        raise ValueError(
            f"dipole must have shape (3,), got {p.shape}"
        )

    px, py, pz = p
    return np.array([pz, px, py])


def dipole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """Convert real spherical dipole components to Cartesian form.

    Input ordering:

        [Q_10, Q_11c, Q_11s]
    """
    q = np.asarray(q)

    if q.shape != (3,):
        raise ValueError(
            f"dipole must have shape (3,), got {q.shape}"
        )

    Q_10, Q_11c, Q_11s = q
    return np.array([Q_11c, Q_11s, Q_10])


def quadrupole_cartesian_to_spherical(theta: np.ndarray) -> np.ndarray:
    """Convert a traceless Cartesian quadrupole tensor to real spherical form.

    Parameters
    ----------
    theta : np.ndarray, shape (3, 3)
        Symmetric traceless Cartesian quadrupole tensor.

    Returns
    -------
    np.ndarray, shape (5,)
        Real spherical components ordered as

        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s]

    Notes
    -----
    Stone convention (Racah-normalized off-diagonal terms):

        Q_20  = Theta_zz
        Q_21c = (2/sqrt(3)) * Theta_xz
        Q_21s = (2/sqrt(3)) * Theta_yz
        Q_22c = (Theta_xx - Theta_yy) / sqrt(3)
        Q_22s = (2/sqrt(3)) * Theta_xy

    The factor of 2/sqrt(3) (equivalently 1/sqrt(3) on the diagonal
    combination) is the Racah normalization that makes these components
    consistent with the real solid/spherical harmonics used throughout
    Stone's "The Theory of Intermolecular Forces". It is *not* optional
    bookkeeping -- omitting it changes the physical magnitude represented
    by Q_21c/Q_21s/Q_22c/Q_22s relative to a genuine Stone multipole.
    """
    theta = np.asarray(theta)

    if theta.shape != (3, 3):
        raise ValueError(
            f"quadrupole tensor must have shape (3, 3), got {theta.shape}"
        )

    if not np.allclose(theta, theta.T, atol=1e-10):
        raise ValueError("quadrupole tensor must be symmetric")

    if not np.isclose(np.trace(theta), 0.0, atol=1e-10):
        raise ValueError(
            f"quadrupole tensor must be traceless, got trace = {np.trace(theta):.6e}"
        )

    Q_20 = theta[2, 2]
    Q_21c = (2.0 / _SQRT3) * theta[0, 2]
    Q_21s = (2.0 / _SQRT3) * theta[1, 2]
    Q_22c = (theta[0, 0] - theta[1, 1]) / _SQRT3
    Q_22s = (2.0 / _SQRT3) * theta[0, 1]

    return np.array([Q_20, Q_21c, Q_21s, Q_22c, Q_22s])


def quadrupole_spherical_to_cartesian(q: np.ndarray) -> np.ndarray:
    """Convert real spherical quadrupole moments to a Cartesian tensor.

    Parameters
    ----------
    q : np.ndarray, shape (5,)
        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s]

    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric traceless Cartesian quadrupole tensor.

    Notes
    -----
    Inverse of the Stone convention used by
    ``quadrupole_cartesian_to_spherical``:

        Theta_zz = Q_20
        Theta_xz = (sqrt(3)/2) * Q_21c
        Theta_yz = (sqrt(3)/2) * Q_21s
        Theta_xy = (sqrt(3)/2) * Q_22s
        Theta_xx = -Q_20/2 + (sqrt(3)/2) * Q_22c
        Theta_yy = -Q_20/2 - (sqrt(3)/2) * Q_22c
    """
    q = np.asarray(q)

    if q.shape != (5,):
        raise ValueError(
            f"quadrupole must have shape (5,), got {q.shape}"
        )

    Q_20, Q_21c, Q_21s, Q_22c, Q_22s = q

    theta = np.zeros((3, 3))
    theta[2, 2] = Q_20
    theta[0, 0] = -Q_20 / 2.0 + (_SQRT3 / 2.0) * Q_22c
    theta[1, 1] = -Q_20 / 2.0 - (_SQRT3 / 2.0) * Q_22c
    theta[0, 2] = theta[2, 0] = (_SQRT3 / 2.0) * Q_21c
    theta[1, 2] = theta[2, 1] = (_SQRT3 / 2.0) * Q_21s
    theta[0, 1] = theta[1, 0] = (_SQRT3 / 2.0) * Q_22s

    return theta


# ---------------------------------------------------------------------------
# Direct spherical evaluation: potentials
# ---------------------------------------------------------------------------

def spherical_dipole_potential(dipoles, coords, points):
    """Compute electrostatic potential directly from spherical dipole moments.

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Spherical dipole components [Q_10, Q_11c, Q_11s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    validate_shapes(coords, points)

    if len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    r_vecs, _, safe_r = compute_displacement(coords, points)

    x = r_vecs[:, :, 0]
    y = r_vecs[:, :, 1]
    z = r_vecs[:, :, 2]

    Q10 = dipoles[:, 0]
    Q11c = dipoles[:, 1]
    Q11s = dipoles[:, 2]

    p_dot_r = (
        Q11c[np.newaxis, :] * x
        + Q11s[np.newaxis, :] * y
        + Q10[np.newaxis, :] * z
    )

    return np.sum(p_dot_r / safe_r**3, axis=1)


def spherical_quadrupole_potential(quadrupoles, coords, points):
    """
    Compute electrostatic potential directly from spherical quadrupole moments.

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 5)
        Spherical quadrupole components
        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    potential : np.ndarray, shape (M,)

    Notes
    -----
    The potential is

        V(r) = Theta_ab r_a r_b / r^5
             = [ Theta_xx x^2 + Theta_yy y^2 + Theta_zz z^2
                 + 2 Theta_xy x y + 2 Theta_xz x z + 2 Theta_yz y z ] / r^5

    Substituting the Stone relations

        Theta_zz = Q_20
        Theta_xx = -Q_20/2 + (sqrt(3)/2) Q_22c
        Theta_yy = -Q_20/2 - (sqrt(3)/2) Q_22c
        Theta_xy = (sqrt(3)/2) Q_22s
        Theta_xz = (sqrt(3)/2) Q_21c
        Theta_yz = (sqrt(3)/2) Q_21s

    gives

        V(r) = [ Q_20 (z^2 - (x^2+y^2)/2)
                 + (sqrt(3)/2) Q_22c (x^2 - y^2)
                 + sqrt(3) Q_22s x y
                 + sqrt(3) Q_21c x z
                 + sqrt(3) Q_21s y z ] / r^5
    """
    quadrupoles = np.asarray(quadrupoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    validate_shapes(coords, points)

    if len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    if quadrupoles.ndim != 2 or quadrupoles.shape[1] != 5:
        raise ValueError(
            f"quadrupoles must have shape (N, 5), got {quadrupoles.shape}"
        )

    r_vecs, _, safe_r = compute_displacement(coords, points)

    x = r_vecs[:, :, 0]
    y = r_vecs[:, :, 1]
    z = r_vecs[:, :, 2]

    Q20 = quadrupoles[:, 0]
    Q21c = quadrupoles[:, 1]
    Q21s = quadrupoles[:, 2]
    Q22c = quadrupoles[:, 3]
    Q22s = quadrupoles[:, 4]

    numerator = (
        Q20[np.newaxis, :] * (z**2 - 0.5 * (x**2 + y**2))
        + (_SQRT3 / 2.0) * Q22c[np.newaxis, :] * (x**2 - y**2)
        + _SQRT3 * Q22s[np.newaxis, :] * x * y
        + _SQRT3 * Q21c[np.newaxis, :] * x * z
        + _SQRT3 * Q21s[np.newaxis, :] * y * z
    )

    return np.sum(numerator / safe_r**5, axis=1)


# ---------------------------------------------------------------------------
# Direct spherical evaluation: fields
#
# Each of these converts spherical components to their Cartesian
# equivalent and reuses the corresponding cartesian.py field routine
# (E = -grad V), rather than duplicating the field formula.
# ---------------------------------------------------------------------------

def spherical_monopole_field(charges, coords, points):
    """Compute electric field directly from point charges.

    A point charge has no orientation, so its field is identical in the
    Cartesian and spherical pictures -- this delegates straight to
    ``cartesian.monopole_field``.

    Parameters
    ----------
    charges : np.ndarray, shape (N,)
        Point charges in atomic units.
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    return monopole_field(charges, coords, points)


def spherical_dipole_field(dipoles, coords, points):
    """Compute electric field directly from spherical dipole moments.

    Converts the spherical dipole components to their Cartesian
    equivalent and evaluates the field using the existing Cartesian
    ``dipole_field`` routine (E = -grad V).

    Parameters
    ----------
    dipoles : np.ndarray, shape (N, 3)
        Spherical dipole components [Q_10, Q_11c, Q_11s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    dipoles = np.asarray(dipoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    validate_shapes(coords, points)

    if len(dipoles) != len(coords):
        raise ValueError("dipoles and coords must have same length")

    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"dipoles must have shape (N, 3), got {dipoles.shape}"
        )

    cartesian = np.array(
        [dipole_spherical_to_cartesian(p) for p in dipoles]
    )

    return dipole_field(cartesian, coords, points)


def spherical_quadrupole_field(quadrupoles, coords, points):
    """Compute electric field directly from spherical quadrupole moments.

    Reuses the same infrastructure as ``spherical_quadrupole_potential``:
    the spherical components are converted back to a Cartesian traceless
    quadrupole tensor via ``quadrupole_spherical_to_cartesian``, and the
    field is then evaluated using the existing Cartesian
    ``quadrupole_field`` routine (E = -grad V).

    Parameters
    ----------
    quadrupoles : np.ndarray, shape (N, 5)
        Spherical quadrupole components
        [Q_20, Q_21c, Q_21s, Q_22c, Q_22s].
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    quadrupoles = np.asarray(quadrupoles)
    coords = np.asarray(coords)
    points = np.asarray(points)

    validate_shapes(coords, points)

    if len(quadrupoles) != len(coords):
        raise ValueError("quadrupoles and coords must have same length")

    if quadrupoles.ndim != 2 or quadrupoles.shape[1] != 5:
        raise ValueError(
            f"quadrupoles must have shape (N, 5), got {quadrupoles.shape}"
        )

    cartesian = np.array(
        [
            quadrupole_spherical_to_cartesian(q)
            for q in quadrupoles
        ]
    )

    return quadrupole_field(
        cartesian,
        coords,
        points,
    )


# ---------------------------------------------------------------------------
# High-level interfaces: superposition over all spherical multipole orders
# ---------------------------------------------------------------------------

def spherical_total_potential(
    coords,
    points,
    charges=None,
    dipoles=None,
    quadrupoles=None,
):
    """
    Compute total electrostatic potential from all spherical multipole
    contributions.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)
    charges : np.ndarray, shape (N,), optional
    dipoles : np.ndarray, shape (N, 3), optional
        Spherical dipole moments.
    quadrupoles : np.ndarray, shape (N, 5), optional
        Spherical quadrupole moments.

    Returns
    -------
    potential : np.ndarray, shape (M,)
    """
    coords = np.asarray(coords)
    points = np.asarray(points)

    validate_shapes(coords, points)

    potential = np.zeros(points.shape[0])

    if charges is not None:
        potential += monopole_potential(
            charges,
            coords,
            points,
        )

    if dipoles is not None:
        potential += spherical_dipole_potential(
            dipoles,
            coords,
            points,
        )

    if quadrupoles is not None:
        potential += spherical_quadrupole_potential(
            quadrupoles,
            coords,
            points,
        )

    return potential


def spherical_total_field(
    coords,
    points,
    charges=None,
    dipoles=None,
    quadrupoles=None,
):
    """
    Compute total electric field from all spherical multipole
    contributions.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
    points : np.ndarray, shape (M, 3)
    charges : np.ndarray, shape (N,), optional
    dipoles : np.ndarray, shape (N, 3), optional
        Spherical dipole moments.
    quadrupoles : np.ndarray, shape (N, 5), optional
        Spherical quadrupole moments.

    Returns
    -------
    field : np.ndarray, shape (M, 3)
    """
    coords = np.asarray(coords)
    points = np.asarray(points)

    validate_shapes(coords, points)

    field = np.zeros((points.shape[0], 3))

    if charges is not None:
        field += spherical_monopole_field(
            charges,
            coords,
            points,
        )

    if dipoles is not None:
        field += spherical_dipole_field(
            dipoles,
            coords,
            points,
        )

    if quadrupoles is not None:
        field += spherical_quadrupole_field(
            quadrupoles,
            coords,
            points,
        )

    return field