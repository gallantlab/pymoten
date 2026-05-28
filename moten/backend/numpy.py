"""The "numpy" CPU backend, based on NumPy.

To use this backend, call ``moten.backend.set_backend("numpy")``.
"""
import numpy as np

try:
    import scipy.linalg as scipy_linalg
    _has_scipy = True
except ImportError:
    _has_scipy = False

###############################################################################

name = "numpy"

# Constants
pi = np.pi
inf = np.inf
nan = np.nan
float32 = np.float32
float64 = np.float64

# Math functions
sin = np.sin
cos = np.cos
exp = np.exp
sqrt = np.sqrt
log = np.log
abs = np.abs
real = np.real
arctan2 = np.arctan2
mod = np.mod

# Array creation
zeros = np.zeros
zeros_like = np.zeros_like
asarray = np.asarray
arange = np.arange
linspace = np.linspace
meshgrid = np.meshgrid
array = np.array

# Array operations
prod = np.prod
unique = np.unique
ceil = np.ceil
floor = np.floor
allclose = np.allclose
concatenate = np.concatenate
stack = np.stack
matmul = np.matmul
diag = np.diag


def column_stack(arrays):
    """Stack 1-D arrays as columns into a 2-D array."""
    return np.column_stack(arrays)


def to_numpy(array):
    """Convert array to numpy. Identity for numpy backend."""
    return np.asarray(array)


def eigh(matrix):
    """Symmetric eigenvalue decomposition."""
    if _has_scipy:
        return scipy_linalg.eigh(matrix)
    return np.linalg.eigh(matrix)
