"""Tests for batched motion energy computation.

Verifies that the batched implementation produces results equivalent
to the original per-filter implementation across backends and
batch sizes.
"""
import numpy as np
import pytest

import moten
from moten.backend import set_backend, get_backend
from moten import core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_torch():
    try:
        import torch
        return True
    except ImportError:
        return False


@pytest.fixture(autouse=True)
def _reset_backend_after_test():
    yield
    set_backend("numpy")


def _make_test_stimulus(nimages=50, vdim=16, hdim=24, seed=42):
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
# mk_3d_gabor_batched equivalence
# ---------------------------------------------------------------------------

class TestMk3dGaborBatched:
    """Test that mk_3d_gabor_batched matches per-filter mk_3d_gabor."""

    def test_single_filter(self):
        """Batched with B=1 should match the original exactly."""
        set_backend("numpy")
        vhsize = (16, 24)
        params = dict(
            stimulus_fps=15, centerh=0.75, centerv=0.5,
            direction=45.0, spatial_freq=4.0, spatial_env=0.3,
            temporal_freq=2.0, temporal_env=0.3,
            filter_temporal_width=10, aspect_ratio=24 / 16.0,
            spatial_phase_offset=0.0,
        )

        sg_sin_ref, sg_cos_ref, tg_sin_ref, tg_cos_ref = core.mk_3d_gabor(
            vhsize, **params)

        sg_sin, sg_cos, tg_sin, tg_cos = core.mk_3d_gabor_batched(
            vhsize, [params])

        np.testing.assert_allclose(
            sg_sin[0], sg_sin_ref.reshape(-1), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            sg_cos[0], sg_cos_ref.reshape(-1), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            tg_sin[0], tg_sin_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            tg_cos[0], tg_cos_ref, atol=1e-12, rtol=1e-12)

    def test_multiple_filters(self):
        """Batched with multiple filters matches per-filter results."""
        set_backend("numpy")
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        vhsize = (16, 24)
        filters = pyramid.filters

        sg_sin_b, sg_cos_b, tg_sin_b, tg_cos_b = core.mk_3d_gabor_batched(
            vhsize, filters)

        for i, filt in enumerate(filters):
            sg_sin, sg_cos, tg_sin, tg_cos = core.mk_3d_gabor(
                vhsize, **filt)
            np.testing.assert_allclose(
                sg_sin_b[i], sg_sin.reshape(-1), atol=1e-10, rtol=1e-10,
                err_msg=f"spatial sin mismatch for filter {i}")
            np.testing.assert_allclose(
                sg_cos_b[i], sg_cos.reshape(-1), atol=1e-10, rtol=1e-10,
                err_msg=f"spatial cos mismatch for filter {i}")
            np.testing.assert_allclose(
                tg_sin_b[i], tg_sin, atol=1e-10, rtol=1e-10,
                err_msg=f"temporal sin mismatch for filter {i}")
            np.testing.assert_allclose(
                tg_cos_b[i], tg_cos, atol=1e-10, rtol=1e-10,
                err_msg=f"temporal cos mismatch for filter {i}")

    @pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
    def test_multiple_filters_torch(self):
        """Batched gabors match per-filter on torch backend."""
        set_backend("torch")
        backend = get_backend()
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        vhsize = (16, 24)
        filters = pyramid.filters

        sg_sin_b, sg_cos_b, tg_sin_b, tg_cos_b = core.mk_3d_gabor_batched(
            vhsize, filters)

        for i, filt in enumerate(filters):
            sg_sin, sg_cos, tg_sin, tg_cos = core.mk_3d_gabor(
                vhsize, **filt)
            np.testing.assert_allclose(
                backend.to_numpy(sg_sin_b[i]),
                backend.to_numpy(sg_sin).reshape(-1),
                atol=1e-10, rtol=1e-10,
                err_msg=f"spatial sin mismatch for filter {i}")
            np.testing.assert_allclose(
                backend.to_numpy(tg_sin_b[i]),
                backend.to_numpy(tg_sin),
                atol=1e-10, rtol=1e-10,
                err_msg=f"temporal sin mismatch for filter {i}")


# ---------------------------------------------------------------------------
# project_stimulus_batched equivalence
# ---------------------------------------------------------------------------

class TestProjectStimulusBatched:
    """Test equivalence between batched and original project_stimulus."""

    def test_equivalence_numpy(self):
        """Batched matches original on numpy backend."""
        set_backend("numpy")
        stimulus = _make_test_stimulus()
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        ref = pyramid.project_stimulus(stimulus)
        batched = pyramid.project_stimulus_batched(stimulus)

        # The batched version skips the masklimit optimization, so there
        # will be tiny differences from near-zero gabor pixel contributions.
        np.testing.assert_allclose(
            batched, ref, atol=1e-4, rtol=1e-4,
            err_msg="Batched vs original mismatch on numpy")

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 1024])
    def test_batch_sizes_numpy(self, batch_size):
        """Different batch sizes produce the same result."""
        set_backend("numpy")
        stimulus = _make_test_stimulus()
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        ref = pyramid.project_stimulus_batched(stimulus, batch_size=128)
        result = pyramid.project_stimulus_batched(stimulus,
                                                  batch_size=batch_size)

        np.testing.assert_allclose(
            result, ref, atol=1e-10, rtol=1e-10,
            err_msg=f"batch_size={batch_size} gives different results")

    def test_3d_and_2d_stimulus(self):
        """Batched accepts both 3D and 2D stimulus input."""
        set_backend("numpy")
        stimulus_3d = _make_test_stimulus()
        nimages, vdim, hdim = stimulus_3d.shape
        stimulus_2d = stimulus_3d.reshape(nimages, -1)

        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        result_3d = pyramid.project_stimulus_batched(stimulus_3d)
        result_2d = core.project_stimulus_batched(
            stimulus_2d, pyramid.filters, vhsize=(vdim, hdim))

        np.testing.assert_allclose(
            result_2d, result_3d, atol=1e-10, rtol=1e-10)

    def test_subset_filters(self):
        """Batched works with a subset of filters."""
        set_backend("numpy")
        stimulus = _make_test_stimulus()
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        subset = pyramid.filters[:10]
        ref = pyramid.project_stimulus(stimulus, filters=subset)
        batched = pyramid.project_stimulus_batched(stimulus, filters=subset)

        np.testing.assert_allclose(
            batched, ref, atol=1e-4, rtol=1e-4,
            err_msg="Batched subset mismatch")

    def test_output_shape(self):
        """Output has correct shape."""
        set_backend("numpy")
        stimulus = _make_test_stimulus(nimages=30)
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        result = pyramid.project_stimulus_batched(stimulus)
        assert result.shape == (30, pyramid.nfilters)

    @pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
    def test_equivalence_torch(self):
        """Batched matches original on torch backend."""
        set_backend("torch")
        backend = get_backend()
        stimulus = _make_test_stimulus()
        stimulus_t = backend.asarray(stimulus)
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        ref = backend.to_numpy(pyramid.project_stimulus(stimulus_t))
        batched = backend.to_numpy(
            pyramid.project_stimulus_batched(stimulus_t))

        np.testing.assert_allclose(
            batched, ref, atol=1e-4, rtol=1e-4,
            err_msg="Batched vs original mismatch on torch")

    @pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
    @pytest.mark.parametrize("batch_size", [1, 16, 256])
    def test_batch_sizes_torch(self, batch_size):
        """Different batch sizes produce the same result on torch."""
        set_backend("torch")
        backend = get_backend()
        stimulus_t = backend.asarray(_make_test_stimulus())
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)

        ref = backend.to_numpy(
            pyramid.project_stimulus_batched(stimulus_t, batch_size=128))
        result = backend.to_numpy(
            pyramid.project_stimulus_batched(stimulus_t,
                                             batch_size=batch_size))

        np.testing.assert_allclose(
            result, ref, atol=1e-10, rtol=1e-10,
            err_msg=f"torch batch_size={batch_size} gives different results")

    @pytest.mark.skipif(not _has_torch(), reason="PyTorch not installed")
    def test_cross_backend_equivalence(self):
        """Batched numpy and batched torch produce similar results."""
        stimulus = _make_test_stimulus()

        set_backend("numpy")
        pyramid_np = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        ref = pyramid_np.project_stimulus_batched(stimulus)

        set_backend("torch")
        backend = get_backend()
        stimulus_t = backend.asarray(stimulus)
        pyramid_t = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        result = backend.to_numpy(
            pyramid_t.project_stimulus_batched(stimulus_t))

        np.testing.assert_allclose(
            result, ref, atol=1e-4, rtol=1e-4,
            err_msg="Batched cross-backend mismatch (numpy vs torch)")


# ---------------------------------------------------------------------------
# core-level function tests
# ---------------------------------------------------------------------------

class TestCoreBatchedFunction:
    """Test core.project_stimulus_batched directly."""

    def test_matches_core_project_stimulus(self):
        """core.project_stimulus_batched matches core.project_stimulus."""
        set_backend("numpy")
        stimulus = _make_test_stimulus()
        vhsize = (16, 24)
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        filters = pyramid.filters

        ref = core.project_stimulus(stimulus, filters, vhsize=vhsize)
        batched = core.project_stimulus_batched(
            stimulus, filters, vhsize=vhsize)

        np.testing.assert_allclose(
            batched, ref, atol=1e-4, rtol=1e-4,
            err_msg="core-level batched vs original mismatch")

    def test_batch_size_one_matches_full(self):
        """batch_size=1 (degenerate) still produces correct results."""
        set_backend("numpy")
        stimulus = _make_test_stimulus(nimages=20)
        vhsize = (16, 24)
        pyramid = moten.pyramids.MotionEnergyPyramid(**SMALL_PYRAMID_KWARGS)
        filters = pyramid.filters

        ref = core.project_stimulus_batched(
            stimulus, filters, vhsize=vhsize, batch_size=9999)
        result = core.project_stimulus_batched(
            stimulus, filters, vhsize=vhsize, batch_size=1)

        np.testing.assert_allclose(
            result, ref, atol=1e-10, rtol=1e-10)
