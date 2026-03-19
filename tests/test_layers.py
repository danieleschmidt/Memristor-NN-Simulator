"""Tests for MemristiveLayer."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from memristor_nn.layers import MemristiveLayer


def make_layer(in_f=8, out_f=4, **kw):
    rng = np.random.default_rng(7)
    return MemristiveLayer(in_f, out_f, rng=rng, **kw)


class TestMemristiveLayer:

    def test_construction(self):
        layer = make_layer()
        assert layer.in_features == 8
        assert layer.out_features == 4

    def test_set_weights_wrong_shape(self):
        layer = make_layer()
        with pytest.raises(ValueError):
            layer.set_weights(np.ones((4, 8)))  # transposed

    def test_forward_shape_1d(self):
        rng = np.random.default_rng(7)
        layer = make_layer()
        W = rng.normal(0, 0.1, (8, 4))
        layer.set_weights(W)
        x = np.ones(8)
        out = layer(x)
        assert out.shape == (4,)

    def test_forward_shape_2d(self):
        rng = np.random.default_rng(7)
        layer = make_layer()
        W = rng.normal(0, 0.1, (8, 4))
        layer.set_weights(W)
        x = np.ones((10, 8))
        out = layer(x)
        assert out.shape == (10, 4)

    def test_forward_noiseless_close_to_software(self):
        """Noiseless forward should approximate software matmul."""
        rng = np.random.default_rng(99)
        layer = MemristiveLayer(
            16, 8,
            device_kwargs={"d2d_sigma": 0.0, "read_noise_sigma": 0.0},
            rng=np.random.default_rng(99),
        )
        W = rng.normal(0, 0.2, (16, 8))
        b = rng.normal(0, 0.01, 8)
        layer.set_weights(W, b)

        x = rng.normal(0, 0.5, 16)
        expected = x @ W + b
        actual = layer(x, noisy=False)

        # Should be reasonably close (programming is approximate)
        corr = np.corrcoef(expected, actual)[0, 1]
        assert corr > 0.95, f"Noiseless output correlation too low: {corr:.3f}"

    def test_repr(self):
        layer = make_layer()
        r = repr(layer)
        assert "MemristiveLayer" in r
        assert "8→4" in r
