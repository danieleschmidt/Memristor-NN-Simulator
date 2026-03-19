"""
EnergyModel: estimates energy consumption per inference for a MemristiveNN.

Physics
-------
During a VMM read, each memristor dissipates power:
    P_ij = V_ij² × G_ij  (Ohm's law: P = V²/R = V²·G)

For a crossbar input vector V ∈ ℝ^n:
    P_j = Σ_i  V_i² · G_ij

Total crossbar energy for one inference sample (single-shot read, time T_read):
    E_xbar = T_read × Σ_ij  V_i² · G_ij

Peripheral overhead (ADC, DAC, digital activation) is modelled as a fixed
multiplier over the crossbar energy.

Comparison
----------
A standard CMOS multiply-accumulate at 7 nm costs ~1 fJ per MAC.
Crossbar VMM is measured in nJ/inference but at much higher parallelism —
the key metric is energy/operation (fJ/MAC), typically 10–100× lower than CMOS.
"""

import numpy as np
from .network import MemristiveNN


class EnergyModel:
    """Estimates energy per inference for a MemristiveNN.

    Parameters
    ----------
    t_read : float
        Read pulse duration in seconds. Default 10 ns.
    v_read : float
        Read voltage applied to rows. Default 0.1 V.
    peripheral_overhead : float
        Multiplier over raw crossbar energy to account for ADC/DAC/digital.
        Default 3.0 (conservative estimate).
    cmos_energy_per_mac_j : float
        Reference CMOS energy per MAC in Joules. Default 1e-15 (1 fJ).
    """

    def __init__(
        self,
        t_read: float = 10e-9,
        v_read: float = 0.1,
        peripheral_overhead: float = 3.0,
        cmos_energy_per_mac_j: float = 1e-15,
    ):
        self.t_read = t_read
        self.v_read = v_read
        self.peripheral_overhead = peripheral_overhead
        self.cmos_energy_per_mac_j = cmos_energy_per_mac_j

    def crossbar_energy(
        self, conductance_matrix: np.ndarray, v_in: np.ndarray
    ) -> float:
        """Energy for one VMM on a single crossbar (Joules).

        Parameters
        ----------
        conductance_matrix : ndarray, shape (n_rows, n_cols)
        v_in : ndarray, shape (n_rows,)

        Returns
        -------
        float : energy in Joules
        """
        G = np.asarray(conductance_matrix)
        v = np.asarray(v_in)
        # Power per device: P_ij = V_i² * G_ij
        # Total power: sum over all devices
        P_total = np.sum((v[:, np.newaxis] ** 2) * G)
        return float(P_total * self.t_read)

    def estimate(
        self, network: MemristiveNN, x: np.ndarray
    ) -> dict:
        """Estimate total inference energy for a MemristiveNN.

        Parameters
        ----------
        network : MemristiveNN
        x : ndarray, shape (in_features,) — single input sample

        Returns
        -------
        dict with keys:
            crossbar_energy_j     : raw crossbar energy (J)
            total_energy_j        : including peripheral overhead (J)
            total_energy_nj       : same in nJ
            total_macs            : total multiply-accumulate ops
            cmos_equivalent_j     : what same MACs cost in CMOS
            savings_factor        : cmos_equivalent / total_energy
        """
        x = np.asarray(x, dtype=float)
        total_xbar_energy = 0.0
        total_macs = 0

        h = x.copy()
        for layer in network.layers:
            in_f = layer.in_features
            out_f = layer.out_features
            total_macs += in_f * out_f

            # Scale input to read voltage
            v_scaled = h * self.v_read

            # Positive crossbar
            G_pos = layer._xbar_pos.conductance_matrix(noisy=False)
            total_xbar_energy += self.crossbar_energy(G_pos, v_scaled)

            # Negative crossbar
            G_neg = layer._xbar_neg.conductance_matrix(noisy=False)
            total_xbar_energy += self.crossbar_energy(G_neg, v_scaled)

            # Propagate ideal output for next layer
            from .network import relu
            g_range = layer._g_on - layer._g_off
            i_pos = v_scaled @ G_pos
            i_neg = v_scaled @ G_neg
            h = (i_pos - i_neg) / (self.v_read * g_range) * layer._w_scale
            if layer != network.layers[-1]:
                h = relu(h)  # simplified: apply relu for energy walk-through

        total_energy = total_xbar_energy * self.peripheral_overhead
        cmos_equiv = total_macs * self.cmos_energy_per_mac_j

        return {
            "crossbar_energy_j": total_xbar_energy,
            "total_energy_j": total_energy,
            "total_energy_nj": total_energy * 1e9,
            "total_macs": total_macs,
            "cmos_equivalent_j": cmos_equiv,
            "savings_factor": cmos_equiv / total_energy if total_energy > 0 else float("inf"),
        }

    def __repr__(self) -> str:
        return (
            f"EnergyModel(t_read={self.t_read*1e9:.0f}ns, "
            f"v_read={self.v_read}V, overhead={self.peripheral_overhead}×)"
        )
