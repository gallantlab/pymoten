"""Tests for backend system and cross-backend numerical equivalence.

These tests verify that:
1. The backend switching mechanism works correctly
2. NumPy and PyTorch backends produce numerically equivalent results
3. All core computational functions work with all backends
4. torch_cuda and torch_mps backends (when available) also produce equivalent results
"""
import numpy as np
import pytest

import moten
from moten.backend import set_backend, get_backend, ALL_BACKENDS
from moten import core, utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


def _has_torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_torch_mps():
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


# Backends available in this environment
AVAILABLE_BACKENDS = ["numpy"]
if _has_torch():
    AVAILABLE_BACKENDS.append("torch")
if _has_torch_cuda():
    AVAILABLE_BACKENDS.append("torch_cuda")
if _has_torch_mps():
    AVAILABLE_BACKENDS.append("torch_mps")

# Non-numpy backends available
_NON_NUMPY_BACKENDS = [b for b in AVAILABLE_BACKENDS if b != "numpy"]

# Per-backend skip reasons
_SKIP_REASONS = {
    "torch": "PyTorch not installed",
    "torch_cuda": "PyTorch CUDA not available",
    "torch_mps": "PyTorch MPS not available",
}

_AVAILABILITY_CHECKS = {
    "torch": _has_torch,
    "torch_cuda": _has_torch_cuda,
    "torch_mps": _has_torch_mps,
}


def _skip_if_unavailable(backend_name):
    """Skip the test if backend_name is not available."""
    check = _AVAILABILITY_CHECKS.get(backend_name)
    if check is not None and not check():
        pytest.skip(_SKIP_REASONS[backend_name])


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    """Ensure backend is reset to numpy after every test, even on failure."""
    yield
    set_backend("numpy")


# ---------------------------------------------------------------------------
# Precision tolerances for cross-backend comparisons
# ---------------------------------------------------------------------------

# float64 tolerance (numpy vs torch CPU)
# numpy and torch use different implementations of trig/exp functions,
# leading to ~1e-6 differences in float64 after accumulation through
# matrix multiplications and multi-step computations.
ATOL_F64 = 1e-5
RTOL_F64 = 1e-5

# float32 tolerance (default projection dtype)
ATOL_F32 = 1e-4
RTOL_F32 = 1e-4

# CUDA float64 -- same precision characteristics as CPU torch
ATOL_CUDA_F64 = 1e-5
RTOL_CUDA_F64 = 1e-5

# CUDA float32
ATOL_CUDA_F32 = 1e-4
RTOL_CUDA_F32 = 1e-4

# MPS is float32-only and may have further reduced precision
ATOL_MPS = 5e-3
RTOL_MPS = 1e-2


def _get_tolerances(backend_name):
    """Return (atol, rtol) for float64 comparisons on a given backend."""
    if backend_name == "torch_mps":
        return ATOL_MPS, RTOL_MPS
    if backend_name == "torch_cuda":
        return ATOL_CUDA_F64, RTOL_CUDA_F64
    return ATOL_F64, RTOL_F64


def _get_tolerances_f32(backend_name):
    """Return (atol, rtol) for float32 comparisons on a given backend."""
    if backend_name == "torch_mps":
        return ATOL_MPS, RTOL_MPS
    if backend_name == "torch_cuda":
        return ATOL_CUDA_F32, RTOL_CUDA_F32
    return ATOL_F32, RTOL_F32


# ---------------------------------------------------------------------------
# Small reproducible stimulus
# ---------------------------------------------------------------------------

def _make_test_stimulus(nimages=50, vdim=16, hdim=24, seed=42):
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


# ---------------------------------------------------------------------------
# Reference fixtures (always numpy)
# ---------------------------------------------------------------------------

@pytest.fixture
def numpy_reference_gabor():
    """Compute reference gabor filter with numpy backend."""
    set_backend("numpy")
    vhsize = (16, 24)
    result = core.mk_3d_gabor(
        vhsize,
        stimulus_fps=15,
        centerh=0.75,
        centerv=0.5,
        direction=45.0,
        spatial_freq=4.0,
        spatial_env=0.3,
        temporal_freq=2.0,
        temporal_env=0.3,
        spatial_phase_offset=0.0,
    )
    return tuple(np.array(r) for r in result)


@pytest.fixture
def numpy_reference_projection():
    """Compute reference projection with numpy backend."""
    set_backend("numpy")
    stimulus = _make_test_stimulus()
    pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
    responses = pyramid.project_stimulus(stimulus)
    return np.array(responses)


@pytest.fixture
def numpy_reference_raw_projection():
    """Compute reference raw projection with numpy backend."""
    set_backend("numpy")
    stimulus = _make_test_stimulus()
    pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
    sin_resp, cos_resp = pyramid.raw_project_stimulus(stimulus)
    return np.array(sin_resp), np.array(cos_resp)


# ---------------------------------------------------------------------------
# All non-numpy backends to parametrize over (includes torch, torch_cuda, torch_mps)
# ---------------------------------------------------------------------------

# We parametrize over ALL possible non-numpy backends.
# Each test uses _skip_if_unavailable() so it properly skips when hardware
# is missing instead of erroring.
_ALL_NONDEFAULT_BACKENDS = ["torch", "torch_cuda", "torch_mps"]


# ---------------------------------------------------------------------------
# Basic backend mechanism tests
# ---------------------------------------------------------------------------

class TestBackendMechanism:
    def test_default_backend_is_numpy(self):
        backend = get_backend()
        assert backend.name == "numpy"

    def test_set_and_get_backend(self):
        for name in AVAILABLE_BACKENDS:
            backend = set_backend(name)
            assert backend.name == name
            assert get_backend().name == name

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("nonexistent_backend")

    def test_set_backend_warn_on_error(self):
        with pytest.warns(UserWarning):
            set_backend("nonexistent_backend", on_error="warn")
        assert get_backend().name == "numpy"

    def test_backend_has_required_attributes(self):
        for name in AVAILABLE_BACKENDS:
            backend = set_backend(name)
            required = [
                'name', 'sin', 'cos', 'exp', 'sqrt', 'log', 'abs', 'real',
                'pi', 'inf', 'zeros', 'zeros_like', 'asarray', 'arange',
                'linspace', 'meshgrid', 'prod', 'allclose', 'column_stack',
                'stack', 'to_numpy', 'eigh', 'ceil', 'floor', 'mod',
            ]
            for attr in required:
                assert hasattr(backend, attr), \
                    f"Backend '{name}' missing attribute '{attr}'"

    @pytest.mark.parametrize("backend_name", AVAILABLE_BACKENDS)
    def test_to_numpy_roundtrip(self, backend_name):
        backend = set_backend(backend_name)
        x = backend.asarray([1.0, 2.0, 3.0])
        result = backend.to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_top_level_api(self):
        """Test that set_backend/get_backend are accessible from top-level."""
        assert hasattr(moten, 'set_backend')
        assert hasattr(moten, 'get_backend')


# ---------------------------------------------------------------------------
# Cross-backend equivalence: mk_3d_gabor
# ---------------------------------------------------------------------------

class TestMk3dGaborEquivalence:
    """Test that mk_3d_gabor produces the same results across backends."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_gabor_equivalence(self, backend_name, numpy_reference_gabor):
        _skip_if_unavailable(backend_name)
        atol, rtol = _get_tolerances(backend_name)

        backend = set_backend(backend_name)
        vhsize = (16, 24)
        result = core.mk_3d_gabor(
            vhsize,
            stimulus_fps=15,
            centerh=0.75,
            centerv=0.5,
            direction=45.0,
            spatial_freq=4.0,
            spatial_env=0.3,
            temporal_freq=2.0,
            temporal_env=0.3,
            spatial_phase_offset=0.0,
        )

        for i, (ref, res) in enumerate(zip(numpy_reference_gabor, result)):
            res_np = backend.to_numpy(res)
            np.testing.assert_allclose(
                res_np, ref, atol=atol, rtol=rtol,
                err_msg=f"mk_3d_gabor component {i} mismatch on backend={backend_name}"
            )

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_gabor_zero_temporal_freq(self, backend_name):
        """Test with zero temporal frequency (static filter)."""
        _skip_if_unavailable(backend_name)

        set_backend("numpy")
        vhsize = (16, 24)
        ref = core.mk_3d_gabor(
            vhsize, stimulus_fps=15, temporal_freq=0.0,
            spatial_freq=4.0, temporal_env=0.3,
        )
        ref_np = tuple(np.array(r) for r in ref)

        atol, rtol = _get_tolerances(backend_name)
        backend = set_backend(backend_name)
        result = core.mk_3d_gabor(
            vhsize, stimulus_fps=15, temporal_freq=0.0,
            spatial_freq=4.0, temporal_env=0.3,
        )
        for i, (r, res) in enumerate(zip(ref_np, result)):
            np.testing.assert_allclose(
                backend.to_numpy(res), r, atol=atol, rtol=rtol,
                err_msg=f"Zero temporal freq gabor component {i} mismatch on backend={backend_name}"
            )

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_gabor_multiple_directions(self, backend_name):
        """Test gabor filters at several directions."""
        _skip_if_unavailable(backend_name)

        vhsize = (16, 24)
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        atol, rtol = _get_tolerances(backend_name)

        for direction in directions:
            set_backend("numpy")
            ref = core.mk_3d_gabor(vhsize, stimulus_fps=15, direction=direction)
            ref_np = tuple(np.array(r) for r in ref)

            backend = set_backend(backend_name)
            result = core.mk_3d_gabor(vhsize, stimulus_fps=15, direction=direction)
            for i, (r, res) in enumerate(zip(ref_np, result)):
                np.testing.assert_allclose(
                    backend.to_numpy(res), r, atol=atol, rtol=rtol,
                    err_msg=f"Direction={direction} gabor component {i} mismatch on {backend_name}"
                )

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_gabor_multiple_spatial_frequencies(self, backend_name):
        """Test gabor filters at multiple spatial frequencies."""
        _skip_if_unavailable(backend_name)

        vhsize = (16, 24)
        spatial_freqs = [2, 4, 8, 16]
        atol, rtol = _get_tolerances(backend_name)

        for sf in spatial_freqs:
            set_backend("numpy")
            ref = core.mk_3d_gabor(vhsize, stimulus_fps=15, spatial_freq=sf)
            ref_np = tuple(np.array(r) for r in ref)

            backend = set_backend(backend_name)
            result = core.mk_3d_gabor(vhsize, stimulus_fps=15, spatial_freq=sf)
            for i, (r, res) in enumerate(zip(ref_np, result)):
                np.testing.assert_allclose(
                    backend.to_numpy(res), r, atol=atol, rtol=rtol,
                    err_msg=f"sf={sf} gabor component {i} mismatch on {backend_name}"
                )


# ---------------------------------------------------------------------------
# Cross-backend equivalence: dotspatial_frames
# ---------------------------------------------------------------------------

class TestDotspatialEquivalence:
    """Test that dotspatial_frames produces the same results across backends."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_dotspatial_equivalence(self, backend_name):
        _skip_if_unavailable(backend_name)

        stimulus = _make_test_stimulus()
        vhsize = (stimulus.shape[1], stimulus.shape[2])
        stimulus_2d = stimulus.reshape(stimulus.shape[0], -1)

        set_backend("numpy")
        sgabor_sin, sgabor_cos, _, _ = core.mk_3d_gabor(
            vhsize, stimulus_fps=15, spatial_freq=4.0)
        ref_sin, ref_cos = core.dotspatial_frames(
            sgabor_sin, sgabor_cos, stimulus_2d)
        ref_sin, ref_cos = np.array(ref_sin), np.array(ref_cos)

        atol, rtol = _get_tolerances(backend_name)
        backend = set_backend(backend_name)
        sgabor_sin_t, sgabor_cos_t, _, _ = core.mk_3d_gabor(
            vhsize, stimulus_fps=15, spatial_freq=4.0)
        stimulus_2d_t = backend.asarray(stimulus_2d)
        res_sin, res_cos = core.dotspatial_frames(
            sgabor_sin_t, sgabor_cos_t, stimulus_2d_t)

        np.testing.assert_allclose(
            backend.to_numpy(res_sin), ref_sin, atol=atol, rtol=rtol,
            err_msg=f"dotspatial_frames sin mismatch on {backend_name}")
        np.testing.assert_allclose(
            backend.to_numpy(res_cos), ref_cos, atol=atol, rtol=rtol,
            err_msg=f"dotspatial_frames cos mismatch on {backend_name}")


# ---------------------------------------------------------------------------
# Cross-backend equivalence: dotdelay_frames
# ---------------------------------------------------------------------------

class TestDotdelayEquivalence:
    """Test that dotdelay_frames produces the same results across backends."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_dotdelay_equivalence(self, backend_name):
        _skip_if_unavailable(backend_name)

        stimulus = _make_test_stimulus()
        vhsize = (stimulus.shape[1], stimulus.shape[2])
        stimulus_2d = stimulus.reshape(stimulus.shape[0], -1)

        set_backend("numpy")
        sg_sin, sg_cos, tg_sin, tg_cos = core.mk_3d_gabor(
            vhsize, stimulus_fps=15, spatial_freq=4.0, temporal_freq=2.0)
        ref_sin, ref_cos = core.dotdelay_frames(
            sg_sin, sg_cos, tg_sin, tg_cos, stimulus_2d)
        ref_sin, ref_cos = np.array(ref_sin), np.array(ref_cos)

        atol, rtol = _get_tolerances(backend_name)
        backend = set_backend(backend_name)
        sg_sin_t, sg_cos_t, tg_sin_t, tg_cos_t = core.mk_3d_gabor(
            vhsize, stimulus_fps=15, spatial_freq=4.0, temporal_freq=2.0)
        stimulus_2d_t = backend.asarray(stimulus_2d)
        res_sin, res_cos = core.dotdelay_frames(
            sg_sin_t, sg_cos_t, tg_sin_t, tg_cos_t, stimulus_2d_t)

        np.testing.assert_allclose(
            backend.to_numpy(res_sin), ref_sin, atol=atol, rtol=rtol,
            err_msg=f"dotdelay_frames sin mismatch on {backend_name}")
        np.testing.assert_allclose(
            backend.to_numpy(res_cos), ref_cos, atol=atol, rtol=rtol,
            err_msg=f"dotdelay_frames cos mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_dotdelay_multiple_temporal_frequencies(self, backend_name):
        """Test dotdelay at various temporal frequencies."""
        _skip_if_unavailable(backend_name)

        stimulus = _make_test_stimulus(nimages=30)
        vhsize = (stimulus.shape[1], stimulus.shape[2])
        stimulus_2d = stimulus.reshape(stimulus.shape[0], -1)
        atol, rtol = _get_tolerances(backend_name)

        for tf in [0.0, 1.0, 2.0, 4.0]:
            set_backend("numpy")
            sg_sin, sg_cos, tg_sin, tg_cos = core.mk_3d_gabor(
                vhsize, stimulus_fps=15, spatial_freq=4.0, temporal_freq=tf)
            ref_sin, ref_cos = core.dotdelay_frames(
                sg_sin, sg_cos, tg_sin, tg_cos, stimulus_2d)
            ref_sin, ref_cos = np.array(ref_sin), np.array(ref_cos)

            backend = set_backend(backend_name)
            sg_sin_t, sg_cos_t, tg_sin_t, tg_cos_t = core.mk_3d_gabor(
                vhsize, stimulus_fps=15, spatial_freq=4.0, temporal_freq=tf)
            stimulus_2d_t = backend.asarray(stimulus_2d)
            res_sin, res_cos = core.dotdelay_frames(
                sg_sin_t, sg_cos_t, tg_sin_t, tg_cos_t, stimulus_2d_t)

            np.testing.assert_allclose(
                backend.to_numpy(res_sin), ref_sin, atol=atol, rtol=rtol,
                err_msg=f"dotdelay tf={tf} sin mismatch on {backend_name}")
            np.testing.assert_allclose(
                backend.to_numpy(res_cos), ref_cos, atol=atol, rtol=rtol,
                err_msg=f"dotdelay tf={tf} cos mismatch on {backend_name}")


# ---------------------------------------------------------------------------
# Cross-backend equivalence: full projection
# ---------------------------------------------------------------------------

class TestProjectionEquivalence:
    """Test full pyramid projection equivalence across backends."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_project_stimulus_equivalence(self, backend_name,
                                          numpy_reference_projection):
        _skip_if_unavailable(backend_name)
        atol, rtol = _get_tolerances_f32(backend_name)

        backend = set_backend(backend_name)
        stimulus = _make_test_stimulus()
        stimulus_t = backend.asarray(stimulus)
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        responses = pyramid.project_stimulus(stimulus_t)
        responses_np = backend.to_numpy(responses)

        np.testing.assert_allclose(
            responses_np, numpy_reference_projection, atol=atol, rtol=rtol,
            err_msg=f"project_stimulus mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_raw_project_stimulus_equivalence(self, backend_name,
                                               numpy_reference_raw_projection):
        _skip_if_unavailable(backend_name)
        ref_sin, ref_cos = numpy_reference_raw_projection
        atol, rtol = _get_tolerances_f32(backend_name)

        backend = set_backend(backend_name)
        stimulus = _make_test_stimulus()
        stimulus_t = backend.asarray(stimulus)
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        sin_resp, cos_resp = pyramid.raw_project_stimulus(stimulus_t)

        np.testing.assert_allclose(
            backend.to_numpy(sin_resp), ref_sin, atol=atol, rtol=rtol,
            err_msg=f"raw_project_stimulus sin mismatch on {backend_name}")
        np.testing.assert_allclose(
            backend.to_numpy(cos_resp), ref_cos, atol=atol, rtol=rtol,
            err_msg=f"raw_project_stimulus cos mismatch on {backend_name}")


# ---------------------------------------------------------------------------
# Cross-backend equivalence: utility functions
# ---------------------------------------------------------------------------

class TestUtilsEquivalence:
    """Test that utility functions produce equivalent results."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_log_compress(self, backend_name):
        _skip_if_unavailable(backend_name)

        x_np = np.array([0.1, 0.5, 1.0, 5.0, 100.0])
        set_backend("numpy")
        ref = utils.log_compress(x_np)

        atol, rtol = _get_tolerances(backend_name)
        backend = set_backend(backend_name)
        x_t = backend.asarray(x_np)
        result = utils.log_compress(x_t)
        np.testing.assert_allclose(
            backend.to_numpy(result), ref, atol=atol, rtol=rtol,
            err_msg=f"log_compress mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_sqrt_sum_squares(self, backend_name):
        _skip_if_unavailable(backend_name)

        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        y_np = np.array([5.0, 6.0, 7.0, 8.0])
        set_backend("numpy")
        ref = utils.sqrt_sum_squares(x_np, y_np)

        atol, rtol = _get_tolerances(backend_name)
        backend = set_backend(backend_name)
        x_t = backend.asarray(x_np)
        y_t = backend.asarray(y_np)
        result = utils.sqrt_sum_squares(x_t, y_t)
        np.testing.assert_allclose(
            backend.to_numpy(result), ref, atol=atol, rtol=rtol,
            err_msg=f"sqrt_sum_squares mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_pointwise_square(self, backend_name):
        _skip_if_unavailable(backend_name)

        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        ref = utils.pointwise_square(x_np)

        backend = set_backend(backend_name)
        x_t = backend.asarray(x_np)
        result = utils.pointwise_square(x_t)
        np.testing.assert_allclose(
            backend.to_numpy(result), ref, atol=1e-12, rtol=1e-12,
            err_msg=f"pointwise_square mismatch on {backend_name}")


# ---------------------------------------------------------------------------
# Cross-backend equivalence: individual backend operations
# ---------------------------------------------------------------------------

class TestBackendOperations:
    """Test individual backend operations for correctness."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_linspace_endpoint(self, backend_name):
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        # With endpoint
        result = backend.to_numpy(backend.linspace(0, 1, 5, endpoint=True))
        expected = np.linspace(0, 1, 5, endpoint=True)
        np.testing.assert_allclose(result, expected, atol=1e-12)

        # Without endpoint
        result = backend.to_numpy(backend.linspace(0, 1, 5, endpoint=False))
        expected = np.linspace(0, 1, 5, endpoint=False)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_meshgrid(self, backend_name):
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        x = backend.linspace(0, 1, 4, endpoint=True)
        y = backend.linspace(0, 2, 3, endpoint=True)
        xx, yy = backend.meshgrid(x, y)

        x_np = np.linspace(0, 1, 4, endpoint=True)
        y_np = np.linspace(0, 2, 3, endpoint=True)
        xx_np, yy_np = np.meshgrid(x_np, y_np)

        np.testing.assert_allclose(backend.to_numpy(xx), xx_np, atol=1e-12)
        np.testing.assert_allclose(backend.to_numpy(yy), yy_np, atol=1e-12)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_column_stack(self, backend_name):
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        a = backend.asarray([1.0, 2.0, 3.0])
        b = backend.asarray([4.0, 5.0, 6.0])
        result = backend.to_numpy(backend.column_stack([a, b]))
        expected = np.column_stack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_eigh(self, backend_name):
        """Test eigendecomposition produces equivalent results."""
        _skip_if_unavailable(backend_name)

        # Create a symmetric positive definite matrix
        rng = np.random.RandomState(42)
        A_np = rng.randn(10, 10)
        A_np = A_np @ A_np.T  # make symmetric PD

        set_backend("numpy")
        backend_np = get_backend()
        L_ref, Q_ref = backend_np.eigh(A_np)

        backend = set_backend(backend_name)
        A_t = backend.asarray(A_np)
        L, Q = backend.eigh(A_t)
        L_np = backend.to_numpy(L)
        Q_np = backend.to_numpy(Q)

        # Eigenvalues should match closely
        atol, rtol = _get_tolerances(backend_name)
        np.testing.assert_allclose(L_np, L_ref, atol=atol, rtol=rtol,
                                   err_msg=f"eigh eigenvalues mismatch on {backend_name}")

        # Eigenvectors: check that Q @ diag(L) @ Q.T reconstructs A
        reconstructed = Q_np @ np.diag(L_np) @ Q_np.T
        np.testing.assert_allclose(reconstructed, A_np, atol=1e-8, rtol=1e-6,
                                   err_msg=f"eigh reconstruction mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_trig_functions(self, backend_name):
        """Test sin/cos/exp produce equivalent results."""
        _skip_if_unavailable(backend_name)

        x_np = np.linspace(-np.pi, np.pi, 100)
        atol, rtol = _get_tolerances(backend_name)

        backend = set_backend(backend_name)
        x_t = backend.asarray(x_np)

        np.testing.assert_allclose(
            backend.to_numpy(backend.sin(x_t)), np.sin(x_np), atol=atol, rtol=rtol)
        np.testing.assert_allclose(
            backend.to_numpy(backend.cos(x_t)), np.cos(x_np), atol=atol, rtol=rtol)
        np.testing.assert_allclose(
            backend.to_numpy(backend.exp(x_t)), np.exp(x_np), atol=atol, rtol=rtol)
        np.testing.assert_allclose(
            backend.to_numpy(backend.sqrt(backend.abs(x_t))), np.sqrt(np.abs(x_np)),
            atol=atol, rtol=rtol)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_zeros_and_zeros_like(self, backend_name):
        """Test zeros and zeros_like produce correct arrays."""
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        z = backend.zeros((3, 4), dtype='float64')
        z_np = backend.to_numpy(z)
        assert z_np.shape == (3, 4)
        np.testing.assert_array_equal(z_np, 0.0)

        zl = backend.zeros_like(z)
        zl_np = backend.to_numpy(zl)
        assert zl_np.shape == (3, 4)
        np.testing.assert_array_equal(zl_np, 0.0)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_arange(self, backend_name):
        """Test arange equivalence."""
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        result = backend.to_numpy(backend.arange(0, 10, 2))
        expected = np.arange(0, 10, 2)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_unique(self, backend_name):
        """Test unique equivalence."""
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        x = backend.asarray([3.0, 1.0, 2.0, 1.0, 3.0])
        result = backend.to_numpy(backend.unique(x))
        expected = np.unique([3.0, 1.0, 2.0, 1.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_prod(self, backend_name):
        """Test prod equivalence."""
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        # prod of a tuple (used in core.py for shape computation)
        result = backend.prod((3, 4, 5))
        # may return tensor or scalar
        if hasattr(result, 'item'):
            result = result.item()
        assert result == 60

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_mod(self, backend_name):
        """Test mod equivalence."""
        _skip_if_unavailable(backend_name)

        backend = set_backend(backend_name)
        result = backend.mod(backend.asarray(7.0), 3)
        if hasattr(result, 'item'):
            result = result.item()
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end equivalence tests
# ---------------------------------------------------------------------------

class TestEndToEndEquivalence:
    """End-to-end tests using the high-level API."""

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_stimulus_motion_energy_equivalence(self, backend_name):
        """Test StimulusMotionEnergy produces equivalent results."""
        _skip_if_unavailable(backend_name)

        stimulus = _make_test_stimulus(nimages=30, vdim=16, hdim=24)

        set_backend("numpy")
        sme_np = moten.pyramids.StimulusMotionEnergy(
            stimulus, stimulus_fps=15,
            temporal_frequencies=[0, 2],
            spatial_frequencies=[0, 2],
            spatial_directions=[0, 90],
        )
        ref = np.array(sme_np.project())

        atol, rtol = _get_tolerances_f32(backend_name)
        backend = set_backend(backend_name)
        stimulus_t = backend.asarray(stimulus)
        sme_t = moten.pyramids.StimulusMotionEnergy(
            stimulus_t, stimulus_fps=15,
            temporal_frequencies=[0, 2],
            spatial_frequencies=[0, 2],
            spatial_directions=[0, 90],
        )
        result = sme_t.project()
        result_np = backend.to_numpy(result)

        np.testing.assert_allclose(
            result_np, ref, atol=atol, rtol=rtol,
            err_msg=f"StimulusMotionEnergy.project() mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_core_project_stimulus_equivalence(self, backend_name):
        """Test core.project_stimulus directly."""
        _skip_if_unavailable(backend_name)

        stimulus = _make_test_stimulus(nimages=30, vdim=16, hdim=24)
        vhsize = (16, 24)

        set_backend("numpy")
        pyramid = moten.pyramids.MotionEnergyPyramid(
            stimulus_vhsize=vhsize, stimulus_fps=15,
            temporal_frequencies=[0, 2],
            spatial_frequencies=[0, 2, 4],
            spatial_directions=[0, 90],
        )
        filters = pyramid.filters
        stimulus_2d = stimulus.reshape(stimulus.shape[0], -1)
        ref = np.array(core.project_stimulus(stimulus_2d, filters, vhsize=vhsize))

        atol, rtol = _get_tolerances_f32(backend_name)
        backend = set_backend(backend_name)
        stimulus_2d_t = backend.asarray(stimulus_2d)
        result = core.project_stimulus(stimulus_2d_t, filters, vhsize=vhsize)
        result_np = backend.to_numpy(result)

        np.testing.assert_allclose(
            result_np, ref, atol=atol, rtol=rtol,
            err_msg=f"core.project_stimulus mismatch on {backend_name}")

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_filters_at_vhposition_equivalence(self, backend_name):
        """Test MotionEnergyPyramid.filters_at_vhposition across backends."""
        _skip_if_unavailable(backend_name)

        stimulus = _make_test_stimulus(nimages=30, vdim=16, hdim=24)

        set_backend("numpy")
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        filters_np = pyramid.filters_at_vhposition(0.5, 0.75)
        ref = np.array(pyramid.project_stimulus(stimulus, filters=filters_np))

        atol, rtol = _get_tolerances_f32(backend_name)
        backend = set_backend(backend_name)
        stimulus_t = backend.asarray(stimulus)
        pyramid_t = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        filters_t = pyramid_t.filters_at_vhposition(0.5, 0.75)
        result = pyramid_t.project_stimulus(stimulus_t, filters=filters_t)
        result_np = backend.to_numpy(result)

        np.testing.assert_allclose(
            result_np, ref, atol=atol, rtol=rtol,
            err_msg=f"filters_at_vhposition mismatch on {backend_name}")


# ---------------------------------------------------------------------------
# Benchmark function tests
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Test the benchmark utility function."""

    def test_benchmark_numpy(self):
        from moten.backend import benchmark
        result = benchmark("numpy")
        assert "numpy" in result
        assert result["numpy"]["duration_seconds"] > 0
        assert result["numpy"]["nfilters"] > 0

    @pytest.mark.parametrize("backend_name", _ALL_NONDEFAULT_BACKENDS)
    def test_benchmark_backend(self, backend_name):
        _skip_if_unavailable(backend_name)
        from moten.backend import benchmark
        result = benchmark(backend_name)
        assert backend_name in result
        assert result[backend_name]["duration_seconds"] > 0

    def test_benchmark_all(self):
        from moten.backend import benchmark
        result = benchmark()
        # Should contain at least numpy
        assert "numpy" in result

    def test_benchmark_returns_to_original_backend(self):
        """Benchmark should not change the current backend."""
        set_backend("numpy")
        from moten.backend import benchmark
        benchmark("numpy")
        assert get_backend().name == "numpy"
