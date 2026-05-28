"""Shared test fixtures and helpers for pymoten tests."""
import numpy as np
import pytest

from moten.backend import set_backend


def has_torch():
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def make_test_stimulus(nimages=50, vdim=16, hdim=24, seed=42):
    """Create a small reproducible test stimulus."""
    rng = np.random.RandomState(seed)
    return rng.randn(nimages, vdim, hdim).astype(np.float64)


SMALL_PYRAMID_KWARGS = dict(
    stimulus_vhsize=(16, 24),
    stimulus_fps=15,
    temporal_frequencies=[0, 2],
    spatial_frequencies=[0, 2, 4],
    spatial_directions=[0, 90, 180, 270],
)


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    """Ensure backend is reset to numpy after every test, even on failure."""
    yield
    set_backend("numpy")
