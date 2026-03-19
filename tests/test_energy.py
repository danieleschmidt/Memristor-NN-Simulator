"""Tests for EnergyModel."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from memristor_nn.network import MemristiveNN
from memristor_nn.energy import EnergyModel


def build_net(sizes=(16, 8, 4), seed=0):
    rng = np.random.default_rng(seed)
    net = MemristiveNN(sizes, rng=np.random.default_rng(seed))
    weights = []
    biases = []
    for i in range(len(sizes) - 1):
        weights.append(rng.normal(0, 0.1, (sizes[i], sizes[i + 1])))
        biases.append(np.zeros(sizes[i + 1]))
    net.load_weights(weights, biases)
    return net, rng


class TestEnergyModel:

    def test_estimate_returns_dict(self):
        net, rng = build_net()
        em = EnergyModel()
        x = rng.normal(0, 0.5, 16)
        result = em.estimate(net, x)
        expected_keys = {
            "crossbar_energy_j", "total_energy_j", "total_energy_nj",
            "total_macs", "cmos_equivalent_j", "savings_factor",
        }
        assert set(result.keys()) == expected_keys

    def test_total_macs_correct(self):
        sizes = (16, 8, 4)
        net, rng = build_net(sizes)
        em = EnergyModel()
        x = rng.normal(0, 0.5, 16)
        result = em.estimate(net, x)
        expected_macs = 16 * 8 + 8 * 4
        assert result["total_macs"] == expected_macs

    def test_energy_positive(self):
        net, rng = build_net()
        em = EnergyModel()
        x = rng.normal(0, 0.5, 16)
        result = em.estimate(net, x)
        assert result["crossbar_energy_j"] >= 0
        assert result["total_energy_j"] >= 0

    def test_peripheral_overhead(self):
        net, rng = build_net()
        em = EnergyModel(peripheral_overhead=5.0)
        x = rng.normal(0, 0.5, 16)
        result = em.estimate(net, x)
        ratio = result["total_energy_j"] / max(result["crossbar_energy_j"], 1e-30)
        assert abs(ratio - 5.0) < 0.01, f"Overhead ratio should be 5.0, got {ratio:.3f}"

    def test_crossbar_energy_method(self):
        em = EnergyModel(t_read=10e-9)
        G = np.array([[1e-4, 2e-4], [3e-4, 4e-4]])
        v = np.array([0.1, 0.2])
        E = em.crossbar_energy(G, v)
        # Manual: P = sum(v_i^2 * G_ij) = 0.01*1e-4 + 0.01*2e-4 + 0.04*3e-4 + 0.04*4e-4
        P_expected = 0.01 * 1e-4 + 0.01 * 2e-4 + 0.04 * 3e-4 + 0.04 * 4e-4
        E_expected = P_expected * 10e-9
        np.testing.assert_allclose(E, E_expected, rtol=1e-10)

    def test_repr(self):
        em = EnergyModel()
        r = repr(em)
        assert "EnergyModel" in r
