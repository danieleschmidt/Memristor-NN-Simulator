"""Tests for MemristiveNN and SoftwareMLP."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from memristor_nn.network import MemristiveNN, SoftwareMLP


def build_weights(rng, sizes):
    weights = []
    biases = []
    for i in range(len(sizes) - 1):
        weights.append(rng.normal(0, 0.1, (sizes[i], sizes[i + 1])))
        biases.append(np.zeros(sizes[i + 1]))
    return weights, biases


class TestMemristiveNN:

    def test_construction(self):
        net = MemristiveNN([16, 8, 4])
        assert len(net.layers) == 2

    def test_invalid_sizes(self):
        with pytest.raises(ValueError):
            MemristiveNN([16])

    def test_forward_shape(self):
        rng = np.random.default_rng(1)
        sizes = [16, 8, 4]
        net = MemristiveNN(sizes, rng=np.random.default_rng(1))
        W, b = build_weights(rng, sizes)
        net.load_weights(W, b)
        x = rng.normal(0, 0.5, 16)
        out = net(x)
        assert out.shape == (4,)

    def test_forward_batch(self):
        rng = np.random.default_rng(2)
        sizes = [16, 8, 4]
        net = MemristiveNN(sizes, rng=np.random.default_rng(2))
        W, b = build_weights(rng, sizes)
        net.load_weights(W, b)
        x = rng.normal(0, 0.5, (10, 16))
        out = net(x)
        assert out.shape == (10, 4)

    def test_predict(self):
        rng = np.random.default_rng(3)
        sizes = [16, 8, 4]
        net = MemristiveNN(sizes, rng=np.random.default_rng(3))
        W, b = build_weights(rng, sizes)
        net.load_weights(W, b)
        x = rng.normal(0, 0.5, (20, 16))
        preds = net.predict(x)
        assert preds.shape == (20,)
        assert np.all((preds >= 0) & (preds < 4))

    def test_repr(self):
        net = MemristiveNN([16, 8, 4])
        r = repr(net)
        assert "MemristiveNN" in r

    def test_wrong_weight_count(self):
        net = MemristiveNN([16, 8, 4])
        rng = np.random.default_rng(4)
        W = [rng.normal(0, 0.1, (16, 8))]  # only 1 instead of 2
        with pytest.raises(ValueError):
            net.load_weights(W)


class TestSoftwareMLP:

    def test_matches_manual_matmul(self):
        """SoftwareMLP should exactly match manual matmul + relu."""
        rng = np.random.default_rng(10)
        sizes = [8, 4, 2]
        W0 = rng.normal(0, 0.1, (8, 4))
        W1 = rng.normal(0, 0.1, (4, 2))
        b0 = rng.normal(0, 0.01, 4)
        b1 = rng.normal(0, 0.01, 2)

        mlp = SoftwareMLP(sizes, activation="relu")
        mlp.load_weights([W0, W1], [b0, b1])

        x = rng.normal(0, 0.5, 8)
        out = mlp(x)

        h = np.maximum(0, x @ W0 + b0)
        expected = h @ W1 + b1
        np.testing.assert_allclose(out, expected, rtol=1e-10)
