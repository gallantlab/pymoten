"""Backend utilities for pymoten.

Provides set_backend/get_backend for switching between numpy and torch.
"""
import importlib
import types
import warnings
from functools import wraps

ALL_BACKENDS = [
    "numpy",
    "torch",
    "torch_cuda",
    "torch_mps",
]

CURRENT_BACKEND = "numpy"

MATCHING_CPU_BACKEND = {
    "numpy": "numpy",
    "torch": "torch",
    "torch_cuda": "torch",
    "torch_mps": "torch",
}


def set_backend(backend, on_error="raise"):
    """Set the backend using a global variable, and return the backend module.

    Parameters
    ----------
    backend : str or module
        Name or module of the backend.
    on_error : str in {"raise", "warn"}
        Define what is done if the backend fails to be loaded.
        If "warn", this function only warns, and keeps the previous backend.
        If "raise", this function raises on errors.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    global CURRENT_BACKEND

    try:
        if isinstance(backend, types.ModuleType):
            backend = backend.name

        if backend not in ALL_BACKENDS:
            raise ValueError("Unknown backend=%r" % (backend, ))

        module = importlib.import_module(__package__ + "." + backend)
        CURRENT_BACKEND = backend

        if backend == "torch_mps":
            warnings.warn(
                "You are using the torch_mps backend which operates with "
                "float32 precision. Results may be less precise than other "
                "backends due to MPS framework limitations.",
                UserWarning
            )
    except Exception as error:
        if on_error == "raise":
            raise error
        elif on_error == "warn":
            warnings.warn(f"Setting backend to {backend} failed: {str(error)}."
                          f"Falling back to {CURRENT_BACKEND} backend.")
            module = get_backend()
        else:
            raise ValueError('Unknown value on_error=%r' % (on_error, ))

    return module


def get_backend():
    """Get the current backend module.

    Returns
    -------
    module : python module
        Module of the backend.
    """
    module = importlib.import_module(__package__ + "." + CURRENT_BACKEND)
    return module


def benchmark(backend=None, nimages=100, vdim=96, hdim=128, stimulus_fps=24):
    """Benchmark motion energy computation across one or more backends.

    Runs a small motion energy pyramid projection and reports the wall-clock
    time for each backend. Useful for comparing CPU vs GPU performance.

    Parameters
    ----------
    backend : str or None
        Name of a single backend to benchmark, or None to benchmark all
        available backends.
    nimages : int
        Number of video frames in the test stimulus.
    vdim : int
        Vertical dimension of the test stimulus (pixels).
    hdim : int
        Horizontal dimension of the test stimulus (pixels).
    stimulus_fps : int
        Stimulus frame rate.

    Returns
    -------
    results : dict
        Dictionary mapping backend name to a dict with keys:
        - ``duration_seconds``: wall-clock time in seconds
        - ``nimages``: number of frames processed
        - ``vhsize``: ``(vdim, hdim)``
        - ``nfilters``: number of filters in the pyramid

    Examples
    --------
    >>> from moten.backend import benchmark
    >>> results = benchmark("numpy")
    >>> print(f"numpy: {results['numpy']['duration_seconds']:.3f}s")
    """
    import time
    import numpy as np

    original_backend = CURRENT_BACKEND

    if backend is not None:
        backends_to_test = [backend]
    else:
        backends_to_test = list(ALL_BACKENDS)

    # Create stimulus once (always on CPU as numpy)
    rng = np.random.RandomState(0)
    stimulus_np = rng.randn(nimages, vdim, hdim).astype(np.float64)

    results = {}
    for backend_name in backends_to_test:
        try:
            backend_mod = set_backend(backend_name)
        except BaseException:
            # BaseException catches pytest.skip, RuntimeError, ImportError, etc.
            continue

        # Lazy imports inside the loop to avoid import errors
        from moten import pyramids

        stimulus = backend_mod.asarray(stimulus_np)
        pyramid = pyramids.MotionEnergyPyramid(
            stimulus_vhsize=(vdim, hdim),
            stimulus_fps=stimulus_fps,
        )

        # Warm-up run (important for GPU backends)
        pyramid.project_stimulus(stimulus, dtype='float32')

        # Timed run
        start = time.perf_counter()
        pyramid.project_stimulus(stimulus, dtype='float32')
        duration = time.perf_counter() - start

        results[backend_name] = {
            "duration_seconds": duration,
            "nimages": nimages,
            "vhsize": (vdim, hdim),
            "nfilters": pyramid.nfilters,
        }

    # Restore original backend
    set_backend(original_backend)

    return results


def _dtype_to_str(dtype):
    """Cast dtype to string, such as "float32", or "float64"."""
    if isinstance(dtype, str):
        return dtype
    elif hasattr(dtype, "name"):  # works for numpy
        return dtype.name
    elif "torch." in str(dtype):  # works for torch
        return str(dtype)[6:]
    elif dtype is None:
        return None
    elif dtype is bool:
        return "bool"
    else:
        raise NotImplementedError()
