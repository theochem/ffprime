import numpy as np


def monopole_potential(q: float, r_vec: np.ndarray) -> float:
    r = np.linalg.norm(r_vec)

    if r < 1e-12:
        return 0.0

    return q / r


def monopole_field(q: float, r_vec: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(r_vec)

    if r < 1e-12:
        return np.zeros_like(r_vec)

    return q * r_vec / r**3
    