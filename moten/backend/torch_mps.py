"""The "torch_mps" GPU backend, based on PyTorch with Apple MPS.

To use this backend, call ``moten.backend.set_backend("torch_mps")``.

Important Notes:
    This backend uses float32 precision due to MPS framework limitations.
    Results may be less precise than other backends.
"""
import warnings

import torch

from ._utils import _dtype_to_str
from .torch import *  # noqa

if not torch.backends.mps.is_available():
    import sys
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("PyTorch with MPS is not available.")
    raise RuntimeError("PyTorch with MPS is not available.")

###############################################################################

name = "torch_mps"

_already_warned = [False]


def _check_dtype_torch_mps(dtype):
    """Warn that X will be cast from float64 to float32."""
    if _dtype_to_str(dtype) == "float64":
        if not _already_warned[0]:
            warnings.warn(
                "GPU backend torch_mps requires single precision floats "
                f"(float32), got input in {dtype}. Data will be automatically "
                "cast to float32.", UserWarning)
            _already_warned[0] = True
        return "float32"
    return dtype


def asarray(x, dtype=None, device="mps"):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        elif hasattr(x, 'dtype') and hasattr(x.dtype, 'name'):
            dtype = x.dtype.name
        else:
            # Plain Python lists/scalars default to float32 on MPS
            # since MPS doesn't support float64
            dtype = "float32"
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = _check_dtype_torch_mps(dtype)
        dtype = getattr(torch, dtype)
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = "mps"
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


def zeros(shape, dtype='float32', device="mps"):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = _check_dtype_torch_mps(dtype)
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(array, dtype=None, shape=None):
    if dtype is not None and isinstance(dtype, str):
        dtype = _check_dtype_torch_mps(dtype)
        dtype = getattr(torch, dtype)
    device = array.device if hasattr(array, 'device') else 'mps'
    if shape is not None:
        return torch.zeros(shape, dtype=dtype or array.dtype, device=device)
    return torch.zeros_like(array, dtype=dtype)


def linspace(start, stop, num, endpoint=True, dtype=torch.float32):
    if endpoint:
        return torch.linspace(start, stop, num, dtype=dtype, device="mps")
    else:
        step = (stop - start) / num
        return torch.linspace(start, stop - step, num, dtype=dtype, device="mps")


def arange(*args, dtype=torch.float32, **kwargs):
    return torch.arange(*args, dtype=dtype, device="mps", **kwargs)


def meshgrid(*tensors, indexing='xy'):
    return torch.meshgrid(*tensors, indexing=indexing)


def mod(x, y):
    """Modulo operation compatible with numpy semantics."""
    return torch.remainder(asarray(x), y)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    a = asarray(a) if not isinstance(a, torch.Tensor) else a
    b = asarray(b) if not isinstance(b, torch.Tensor) else b
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def to_numpy(array):
    """Convert torch tensor to numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def eigh(matrix):
    """Compute eigendecomposition on CPU and move results back to MPS."""
    input_device = matrix.device
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix.cpu())
    return eigenvalues.to(device=input_device), eigenvectors.to(device=input_device)
