"""Tests for CrossbarArray."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from memristor_nn.crossbar import CrossbarArray
from memristor_nn.device import MemristorDevice


RNG = np.random.default_rng(0)


def make_xbar(rows=4, cols=3, **kw):
    return CrossbarArray(rows, cols, rng=np.random.default_rng(0), **kw)


class TestCrossbarArray:

    def test_construction(self):
        xb = make_xbar(4, 3)
        assert xb.n_rows == 4
        assert xb.n_cols == 3

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            CrossbarArray(0, 3)
        with pytest.raises(ValueError):
            CrossbarArray(3, 0)

    def test_conductance_matrix_shape(self):
        xb = make_xbar(4, 3)
        G = xb.conductance_matrix(noisy=False)
        assert G.shape == (4, 3)

    def test_program_sets_conductances(self):
        xb = make_xbar(4, 3)
        ref = MemristorDevice()
        g_target = np.full((4, 3), ref.g_on * 0.7)
        xb.program(g_target)
        G = xb.conductance_matrix(noisy=False)
        # Not exact (state write is approximate) but should be in right half
        assert np.all(G > ref.g_off)

    def test_vmm_shape_1d(self):
        xb = make_xbar(4, 3)
        v = np.ones(4) * 0.1
        out = xb.vmm(v, noisy=False)
        assert out.shape == (3,)

    def test_vmm_shape_2d(self):
        xb = make_xbar(4, 3)
        v = np.ones((5, 4)) * 0.1
        out = xb.vmm(v, noisy=False)
        assert out.shape == (5, 3)

    def test_vmm_linearity(self):
        """VMM should be linear: f(2v) ≈ 2·f(v) when using same G (noisy=False)."""
        xb = make_xbar(4, 3)
        v = np.random.default_rng(1).random(4) * 0.1
        out1 = xb.vmm(v, noisy=False)
        out2 = xb.vmm(2 * v, noisy=False)
        np.testing.assert_allclose(out2, 2 * out1, rtol=1e-10)

    def test_vmm_wrong_input_shape(self):
        xb = make_xbar(4, 3)
        with pytest.raises(ValueError):
            xb.vmm(np.ones(5), noisy=False)

    def test_program_wrong_shape(self):
        xb = make_xbar(4, 3)
        with pytest.raises(ValueError):
            xb.program(np.ones((3, 4)))

    def test_repr(self):
        xb = make_xbar(4, 3)
        assert "4×3" in repr(xb)
