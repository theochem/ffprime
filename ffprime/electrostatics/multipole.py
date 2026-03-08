import numpy as np


def dipole_potential(p: np.ndarray, r_vec: np.ndarray) -> float:
    """Compute electrostatic potential from dipole moment.
    
    Physics: V = (p · r̂) / r² in atomic units
    
    Parameters
    ----------
    p : np.ndarray
        Dipole moment vector (3,)
    r_vec : np.ndarray
        Position vector(s) (N, 3) or (3,)
        
    Returns
    -------
    float or np.ndarray
        Potential at given position(s)
    """
    r = np.linalg.norm(r_vec, axis=-1 if r_vec.ndim > 1 else 0)
    
    if np.any(r < 1e-12):
        return 0.0 if np.isscalar(r) else np.where(r < 1e-12, 0.0, 
                                                    np.dot(r_vec, p) / r**3)
    
    return np.dot(r_vec, p) / r**3


def dipole_field(p: np.ndarray, r_vec: np.ndarray) -> np.ndarray:
    """Compute electric field from dipole moment.
    
    Physics: E = (1/r³)[3(p·r̂)r̂ - p] in atomic units
    
    Parameters
    ----------
    p : np.ndarray
        Dipole moment vector (3,)
    r_vec : np.ndarray
        Position vector(s) (N, 3) or (3,)
        
    Returns
    -------
    np.ndarray
        Electric field at given position(s), shape (N, 3) or (3,)
    """
    r = np.linalg.norm(r_vec, axis=-1, keepdims=True if r_vec.ndim > 1 else False)
    
    if np.any(r < 1e-12):
        return np.zeros_like(r_vec)
    
    r_hat = r_vec / r
    p_dot_r = np.dot(r_vec, p) if r_vec.ndim == 1 else np.sum(r_vec * p, axis=1, keepdims=True)
    
    return (3 * p_dot_r * r_hat - p) / r**3
