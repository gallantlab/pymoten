"""The "torch_cuda" GPU backend, based on PyTorch with CUDA.

To use this backend, call ``moten.backend.set_backend("torch_cuda")``.
"""
from .torch import *  # noqa
import torch

if not torch.cuda.is_available():
    import sys
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("PyTorch with CUDA is not available.")
    raise RuntimeError("PyTorch with CUDA is not available.")

from ._utils import _dtype_to_str

###############################################################################

name = "torch_cuda"


def asarray(x, dtype=None, device="cuda"):
    if dtype is None:
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
        elif hasattr(x, 'dtype') and hasattr(x.dtype, 'name'):
            dtype = x.dtype.name
    if dtype is not None:
        dtype = _dtype_to_str(dtype)
        dtype = getattr(torch, dtype)
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = "cuda"
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


def zeros(shape, dtype='float64', device="cuda"):
    if isinstance(shape, int):
        shape = (shape, )
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return torch.zeros(shape, dtype=dtype, device=device)


def zeros_like(array, dtype=None, shape=None):
    if dtype is not None and isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    device = array.device if hasattr(array, 'device') else 'cuda'
    if shape is not None:
        return torch.zeros(shape, dtype=dtype or array.dtype, device=device)
    return torch.zeros_like(array, dtype=dtype)


def linspace(start, stop, num, endpoint=True, dtype=torch.float64):
    if endpoint:
        return torch.linspace(start, stop, num, dtype=dtype, device="cuda")
    else:
        step = (stop - start) / num
        return torch.linspace(start, stop - step, num, dtype=dtype, device="cuda")


def arange(*args, dtype=None, **kwargs):
    return torch.arange(*args, dtype=dtype, device="cuda", **kwargs)


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
