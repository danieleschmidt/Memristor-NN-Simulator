"""Tests for MemristorDevice."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from memristor_nn.device import MemristorDevice


RNG = np.random.default_rng(42)


def make_device(**kwargs):
    return MemristorDevice(rng=np.random.default_rng(42), **kwargs)


class TestMemristorDevice:

    def test_default_construction(self):
        d = make_device()
        assert d.ron == 1e4
        assert d.roff == 1e6
        assert 0.0 <= d.w <= 1.0

    def test_invalid_ron_roff(self):
        with pytest.raises(ValueError):
            MemristorDevice(ron=1e6, roff=1e4)

    def test_conductance_bounds(self):
        d = make_device()
        # initial state = 0.5
        g = d.conductance
        assert d.g_off <= g <= d.g_on

    def test_write_positive_sets_lrs(self):
        d = make_device()
        d._w = 0.0  # start at HRS
        d.write_voltage(2.0)  # above v_write
        assert d.w > 0.0, "Positive write should increase w toward LRS"

    def test_write_negative_resets_hrs(self):
        d = make_device()
        d._w = 1.0  # start at LRS
        d.write_voltage(-2.0)  # negative, above threshold magnitude
        assert d.w < 1.0, "Negative write should decrease w toward HRS"

    def test_subthreshold_no_switch(self):
        d = make_device()
        w_before = d.w
        d.write_voltage(0.001)  # well below v_write threshold
        assert d.w == w_before, "Sub-threshold should not change state"

    def test_read_conductance_close_to_nominal(self):
        # Over many reads the average should be close to nominal (small noise)
        d = make_device(d2d_sigma=0.0, read_noise_sigma=0.001)
        d._d2d_factor = 1.0  # eliminate d2d variation
        reads = [d.read_conductance() for _ in range(1000)]
        mean_g = np.mean(reads)
        assert abs(mean_g - d.conductance) / d.conductance < 0.05, (
            f"Mean read ({mean_g:.6f}) too far from nominal ({d.conductance:.6f})"
        )

    def test_reset(self):
        d = make_device()
        d.write_voltage(5.0)
        d.reset()
        assert d.w == 0.5

    def test_repr(self):
        d = make_device()
        r = repr(d)
        assert "MemristorDevice" in r
