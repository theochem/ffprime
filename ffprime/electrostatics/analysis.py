"""Utilities for analyzing electrostatic fields."""

import numpy as np


def project_field(field, direction):
    """
    Project an electric field onto a specified direction.

    Parameters
    ----------
    field : ndarray, shape (..., 3)
        Electric field vector(s).
    direction : array_like, shape (3,)
        Direction vector onto which the field is projected.

    Returns
    -------
    ndarray
        Scalar projection(s) of the field onto the direction.
    """
    field = np.asarray(field, dtype=float)
    direction = np.asarray(direction, dtype=float)

    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Direction vector must have nonzero length.")

    direction = direction / norm

    return np.tensordot(field, direction, axes=([-1], [0]))

def cosine_similarity(field1, field2):
    """
    Compute the cosine similarity between two electric field maps.

    Parameters
    ----------
    field1, field2 : ndarray
        Electric field arrays of identical shape.

    Returns
    -------
    float
        Cosine similarity between -1 and 1.
    """
    field1 = np.asarray(field1, dtype=float)
    field2 = np.asarray(field2, dtype=float)

    if field1.shape != field2.shape:
        raise ValueError("Field arrays must have the same shape.")

    field1 = field1.ravel()
    field2 = field2.ravel()

    norm1 = np.linalg.norm(field1)
    norm2 = np.linalg.norm(field2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute cosine similarity for a zero field.")

    return np.dot(field1, field2) / (norm1 * norm2)

def rms_deviation(field1, field2):
    """
    Compute the root-mean-square deviation between two electric field maps.

    Parameters
    ----------
    field1, field2 : ndarray
        Electric field arrays of identical shape.

    Returns
    -------
    float
        Root-mean-square deviation.
    """
    field1 = np.asarray(field1, dtype=float)
    field2 = np.asarray(field2, dtype=float)

    if field1.shape != field2.shape:
        raise ValueError("Field arrays must have the same shape.")

    return np.sqrt(np.mean((field1 - field2) ** 2))