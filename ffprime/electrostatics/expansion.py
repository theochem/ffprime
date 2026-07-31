"""
Container class for atom-centered multipole expansions.

This module defines :class:`MultipoleExpansion`, a small container that
holds the per-atom multipole data (charges, dipoles, quadrupoles) for a
single molecule or system, along with the representation in which those
moments are stored. The container does not yet evaluate any potential
or field -- evaluation is delegated to the existing routines in
:mod:`ffprime.electrostatics.cartesian` and
:mod:`ffprime.electrostatics.spherical`.
"""

import numpy as np

__all__ = ["MultipoleExpansion"]


_VALID_REPRESENTATIONS = ("cartesian", "spherical")


class MultipoleExpansion:
    """Container for an atom-centered multipole expansion.

    Parameters
    ----------
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates of the multipole sites, in atomic units.
    atcharges : np.ndarray, shape (N,), optional
        Atomic monopole (charge) moments, in atomic units.
    atdipoles : np.ndarray, optional
        Atomic dipole moments. Shape depends on ``representation``: ``(N, 3)``
        for both ``"cartesian"`` and ``"spherical"``.
    atquadrupoles : np.ndarray, optional
        Atomic quadrupole moments. Shape depends on ``representation``:
        ``(N, 3, 3)`` for ``"cartesian"`` (symmetric traceless tensor)
        and ``(N, 5)`` for ``"spherical"`` (real spherical components).
    representation : str, optional
        Either ``"cartesian"`` or ``"spherical"``. Defaults to
        ``"cartesian"``.

    Attributes
    ----------
    representation : str
        Representation in which ``atdipoles`` and ``atquadrupoles`` are
        stored. One of ``"cartesian"`` or ``"spherical"``.
    atcoords : np.ndarray, shape (N, 3)
        Atomic coordinates of the multipole sites.
    atcharges : np.ndarray, shape (N,) or None
        Atomic monopole (charge) moments.
    atdipoles : np.ndarray or None
        Atomic dipole moments, in the stored representation.
    atquadrupoles : np.ndarray or None
        Atomic quadrupole moments, in the stored representation.

    Raises
    ------
    TypeError
        If ``atcoords`` is ``None``.
    ValueError
        If ``representation`` is not one of ``"cartesian"`` or
        ``"spherical"``.
    """

    def __init__(
        self,
        atcoords: np.ndarray,
        atcharges: np.ndarray | None = None,
        atdipoles: np.ndarray | None = None,
        atquadrupoles: np.ndarray | None = None,
        representation: str = "cartesian",
    ) -> None:
        """Store an atom-centered multipole expansion in the chosen form."""
        if atcoords is None:
            raise TypeError("atcoords cannot be None")

        if representation not in _VALID_REPRESENTATIONS:
            raise ValueError(
                "representation must be one of "
                f"{_VALID_REPRESENTATIONS}, got {representation!r}"
            )

        self.representation: str = representation
        self.atcoords: np.ndarray = np.asarray(atcoords)
        self.atcharges: np.ndarray | None = (
            None if atcharges is None else np.asarray(atcharges)
        )
        self.atdipoles: np.ndarray | None = (
            None if atdipoles is None else np.asarray(atdipoles)
        )
        self.atquadrupoles: np.ndarray | None = (
            None if atquadrupoles is None else np.asarray(atquadrupoles)
        )

    @classmethod
    def from_cartesian(
        cls,
        atcoords: np.ndarray,
        atcharges: np.ndarray | None = None,
        atdipoles: np.ndarray | None = None,
        atquadrupoles: np.ndarray | None = None,
    ) -> "MultipoleExpansion":
        """Build a :class:`MultipoleExpansion` from Cartesian multipoles.

        Parameters
        ----------
        atcoords : np.ndarray, shape (N, 3)
            Atomic coordinates of the multipole sites, in atomic units.
        atcharges : np.ndarray, shape (N,), optional
            Atomic monopole (charge) moments, in atomic units.
        atdipoles : np.ndarray, shape (N, 3), optional
            Cartesian atomic dipole moment vectors, in atomic units.
        atquadrupoles : np.ndarray, shape (N, 3, 3), optional
            Symmetric traceless Cartesian atomic quadrupole moment
            tensors, in atomic units.

        Returns
        -------
        MultipoleExpansion
            Container storing the moments in ``"cartesian"``
            representation.
        """
        return cls(
            atcoords=atcoords,
            atcharges=atcharges,
            atdipoles=atdipoles,
            atquadrupoles=atquadrupoles,
            representation="cartesian",
        )

    @classmethod
    def from_spherical(
        cls,
        atcoords: np.ndarray,
        atcharges: np.ndarray | None = None,
        atdipoles: np.ndarray | None = None,
        atquadrupoles: np.ndarray | None = None,
    ) -> "MultipoleExpansion":
        """Build a :class:`MultipoleExpansion` from real spherical multipoles.

        Parameters
        ----------
        atcoords : np.ndarray, shape (N, 3)
            Atomic coordinates of the multipole sites, in atomic units.
        atcharges : np.ndarray, shape (N,), optional
            Atomic monopole (charge) moments, in atomic units. Charges
            have no orientation and are identical in both
            representations.
        atdipoles : np.ndarray, shape (N, 3), optional
            Real spherical atomic dipole components (Stone convention).
        atquadrupoles : np.ndarray, shape (N, 5), optional
            Real spherical atomic quadrupole components (Stone
            convention).

        Returns
        -------
        MultipoleExpansion
            Container storing the moments in ``"spherical"``
            representation.
        """
        return cls(
            atcoords=atcoords,
            atcharges=atcharges,
            atdipoles=atdipoles,
            atquadrupoles=atquadrupoles,
            representation="spherical",
        )