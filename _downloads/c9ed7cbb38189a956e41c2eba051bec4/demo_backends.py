'''
=================================================
 Using different backends (NumPy, PyTorch, GPU)
=================================================

This example shows how to use the different computational backends available in
``pymoten`` to extract motion energy features. Backends let you run the exact
same pyramid projection on the CPU with NumPy (the default) or on the GPU with
PyTorch, which can be substantially faster for large stimuli.

The available backends are:

- ``"numpy"``: CPU backend using NumPy (the default).
- ``"torch"``: CPU backend using PyTorch.
- ``"torch_cuda"``: GPU backend using PyTorch with CUDA (NVIDIA GPUs).
- ``"torch_mps"``: GPU backend using PyTorch with Metal (Apple Silicon GPUs).
  Note that the MPS backend runs in ``float32`` precision only, so results are
  slightly less precise than the other backends.

The numpy backend is always available. The torch backends additionally require
`PyTorch <https://pytorch.org>`_ to be installed, and the GPU backends require a
compatible GPU.
'''

# %%
# Selecting a backend
# ===================
#
# Backends are managed with :func:`moten.backend.set_backend` and
# :func:`moten.backend.get_backend`. The default backend is ``"numpy"``.

import numpy as np
import moten
from moten.backend import set_backend, get_backend, ALL_BACKENDS

print("Available backends:", ALL_BACKENDS)
print("Current backend:", get_backend().name)

# %%
# Computing features with the NumPy backend
# =========================================
#
# Let us create a small synthetic stimulus and a motion energy pyramid. We use
# random noise here so the example runs quickly and reproducibly, but in
# practice you would load a video with :func:`moten.io.video2luminance`.

nimages, vdim, hdim = (100, 72, 128)
stimulus_fps = 24

rng = np.random.RandomState(0)
luminance_images = rng.randn(nimages, vdim, hdim)

pyramid = moten.pyramids.MotionEnergyPyramid(stimulus_vhsize=(vdim, hdim),
                                             stimulus_fps=stimulus_fps)
print(pyramid)

# %%
# With the numpy backend, ``project_stimulus`` works exactly as in the other
# examples and returns a NumPy array of shape ``(nimages, nfilters)``.

set_backend("numpy")
features_numpy = pyramid.project_stimulus(luminance_images)
print(type(features_numpy), features_numpy.shape)

# %%
# Per-filter vs. batched projection
# =================================
#
# Each backend exposes two projection methods. ``project_stimulus`` loops over
# the filters one at a time (lower memory use), while
# ``project_stimulus_batched`` groups filters into batches and computes them with
# large matrix multiplications. The batched version is what unlocks most of the
# GPU speed-up, but it produces the same result on any backend.

features_batched = pyramid.project_stimulus_batched(luminance_images,
                                                    batch_size=128)
print("Per-filter and batched results match:",
      np.allclose(features_numpy, features_batched))

# %%
# Using a GPU backend
# ===================
#
# Switching to a GPU backend is a two-step process:
#
# 1. Call :func:`moten.backend.set_backend` with the backend name. This returns
#    the backend module.
# 2. Move the stimulus onto the device with ``backend.asarray(...)`` before
#    projecting. The result lives on the GPU, so convert it back to a NumPy
#    array with ``backend.to_numpy(...)``.
#
# Here we try the GPU backends in turn. We pass ``on_error="warn"`` so that, on
# machines without a GPU (such as the documentation build server), the call
# simply warns and keeps the current backend instead of raising. This lets the
# example run everywhere while still demonstrating the GPU API.

gpu_backend = None
for name in ["torch_cuda", "torch_mps"]:
    backend = set_backend(name, on_error="warn")
    if backend.name == name:
        gpu_backend = name
        break

if gpu_backend is None:
    print("No GPU backend available; skipping the GPU comparison.")
else:
    print(f"Using GPU backend: {gpu_backend}")

    # Move the stimulus to the GPU.
    stimulus_gpu = backend.asarray(luminance_images)

    # Project on the GPU (batched is recommended for speed).
    features_gpu = pyramid.project_stimulus_batched(stimulus_gpu,
                                                    batch_size=128)

    # Bring the result back to the CPU as a NumPy array.
    features_gpu = backend.to_numpy(features_gpu)

    # Compare against the numpy reference. The GPU result is computed in
    # float32, so we only expect agreement up to single precision.
    max_abs_diff = np.max(np.abs(features_numpy.astype(np.float32)
                                 - features_gpu.astype(np.float32)))
    print(f"Max |numpy - {gpu_backend}|: {max_abs_diff:.2e}")

# %%
# Always reset the backend to numpy when you are done, so that other code is not
# affected by the global backend setting.

set_backend("numpy")

# %%
# Benchmarking backends
# =====================
#
# Finally, ``pymoten`` ships a small helper,
# :func:`moten.backend.benchmark`, that times the per-filter and batched
# projections on one or all backends. It is handy for checking how much speed-up
# a GPU gives on your own hardware.

from moten.backend import benchmark

results = benchmark("numpy", nimages=50, vdim=72, hdim=128)
numpy_result = results["numpy"]
print(f"numpy per-filter: {numpy_result['duration_seconds']:.3f}s")
print(f"numpy batched:    {numpy_result['duration_batched_seconds']:.3f}s")

# %%
# Calling ``benchmark()`` with no arguments times every available backend, so on
# a machine with a GPU you can directly compare CPU and GPU timings.
