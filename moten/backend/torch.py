"""The "torch" CPU backend, based on PyTorch.

To use this backend, call ``moten.backend.set_backend("torch")``.
"""
import math

try:
    import torch
except ImportError as error:
    import sys
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("PyTorch not installed.")
    raise ImportError("PyTorch not installed.") from error

from ._utils import _dtype_to_str

###############################################################################

name = "torch"

# Constants
pi = math.pi
inf = float('inf')
nan = float('nan')
float32 = torch.float32
float64 = torch.float64

# Math functions
sin = torch.sin
cos = torch.cos
exp = torch.exp
sqrt = torch.sqrt
log = torch.log
abs = torch.abs
real = torch.real
arctan2 = torch.atan2


def mod(x, y):
    """Modulo operation compatible with numpy semantics."""
    return torch.fmod(asarray(x), y)


# Array operations
matmul = torch.matmul
stack = torch.stack
diag = torch.diag


def zeros(shape, dtype='float64', device='cpu'):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(array, dtype=None, shape=None):
    if dtype is not None and isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if shape is not None:
        return torch.zeros(shape, dtype=dtype or array.dtype,
                           device=array.device)
    return torch.zeros_like(array, dtype=dtype)


def asarray(x, dtype=None, device='cpu'):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        elif hasattr(x, 'dtype') and hasattr(x.dtype, 'name'):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None and isinstance(x, torch.Tensor):
        device = x.device
    try:
        return torch.as_tensor(x, dtype=dtype, device=device)
    except Exception:
        import numpy as np
        if torch.is_tensor(x) and x.device.type != 'cpu':
            x = x.cpu()
        arr = np.asarray(x, dtype=_dtype_to_str(dtype))
        return torch.as_tensor(arr, dtype=dtype, device=device)


def array(x, dtype=None):
    return asarray(x, dtype=dtype)


def arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


def linspace(start, stop, num, endpoint=True, dtype=torch.float64):
    if endpoint:
        return torch.linspace(start, stop, num, dtype=dtype)
    else:
        # torch.linspace always includes endpoint
        # Replicate numpy endpoint=False behavior
        step = (stop - start) / num
        return torch.linspace(start, stop - step, num, dtype=dtype)


def meshgrid(*tensors, indexing='xy'):
    return torch.meshgrid(*tensors, indexing=indexing)


def prod(x, axis=None):
    if isinstance(x, (tuple, list)):
        x = torch.tensor(x)
    if axis is None:
        return torch.prod(x)
    return torch.prod(x, dim=axis)


def unique(x, axis=None):
    if axis is not None:
        return torch.unique(x, dim=axis)
    return torch.unique(x)


def ceil(x):
    if isinstance(x, (int, float)):
        return math.ceil(x)
    return torch.ceil(x)


def floor(x):
    if isinstance(x, (int, float)):
        return math.floor(x)
    return torch.floor(x)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    a = asarray(a) if not isinstance(a, torch.Tensor) else a
    b = asarray(b) if not isinstance(b, torch.Tensor) else b
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def concatenate(arrays, axis=0):
    return torch.cat(arrays, dim=axis)


def column_stack(arrays):
    """Stack 1-D arrays as columns into a 2-D array."""
    return torch.column_stack(arrays)


def to_numpy(array):
    """Convert torch tensor to numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def eigh(matrix):
    """Symmetric eigenvalue decomposition."""
    return torch.linalg.eigh(matrix)
