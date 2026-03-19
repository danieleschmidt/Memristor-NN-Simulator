"""
MemristorDevice: single memristor physics model.

A memristor is a two-terminal resistive switching device. Resistance toggles
between a low-resistance state (LRS / Ron) and a high-resistance state (HRS / Roff)
under applied voltage. The state variable w ∈ [0, 1] tracks the internal boundary
position (e.g., oxygen vacancy front in TaOx or HfOx devices).

Switching rule (non-linear):
  dw/dt = α · f(V) · w^p · (1 - w)^q  (window function keeps w in [0,1])

Here we use a simplified but physically motivated discrete-step model:
  - |V| > V_write → switch toward LRS (w→1) or HRS (w→0)
  - Conductance: G = G_off + w * (G_on - G_off)
  - Read operation adds device-to-device variation + cycle-to-cycle read noise
"""

import numpy as np


class MemristorDevice:
    """Single memristor with physics-based resistance switching.

    Parameters
    ----------
    ron : float
        Low-resistance state (Ohms). Default 10 kΩ.
    roff : float
        High-resistance state (Ohms). Default 1 MΩ.
    v_write : float
        Threshold voltage for switching (V). Default 1.0 V.
    v_read : float
        Read voltage magnitude (V). Default 0.1 V.
    d2d_sigma : float
        Device-to-device variation (log-normal sigma). Default 0.05.
    read_noise_sigma : float
        Cycle-to-cycle read noise (Gaussian sigma, relative). Default 0.02.
    rng : numpy.random.Generator or None
        Optional RNG for reproducibility.
    """

    def __init__(
        self,
        ron: float = 1e4,
        roff: float = 1e6,
        v_write: float = 1.0,
        v_read: float = 0.1,
        d2d_sigma: float = 0.05,
        read_noise_sigma: float = 0.02,
        rng: np.random.Generator = None,
    ):
        if ron >= roff:
            raise ValueError(f"ron ({ron}) must be < roff ({roff})")
        if v_write <= 0:
            raise ValueError("v_write must be positive")

        self.ron = float(ron)
        self.roff = float(roff)
        self.v_write = float(v_write)
        self.v_read = float(v_read)
        self.d2d_sigma = float(d2d_sigma)
        self.read_noise_sigma = float(read_noise_sigma)
        self._rng = rng if rng is not None else np.random.default_rng()

        # Internal state: w=1 → fully LRS (Ron), w=0 → fully HRS (Roff)
        self._w: float = 0.5

        # Device-to-device variation baked in at creation (log-normal)
        self._d2d_factor = float(
            self._rng.lognormal(mean=0.0, sigma=self.d2d_sigma)
        )

    # ------------------------------------------------------------------
    # State and conductance
    # ------------------------------------------------------------------

    @property
    def w(self) -> float:
        """Internal state variable, ∈ [0, 1]."""
        return self._w

    @property
    def g_on(self) -> float:
        """Nominal conductance in LRS (S)."""
        return 1.0 / self.ron

    @property
    def g_off(self) -> float:
        """Nominal conductance in HRS (S)."""
        return 1.0 / self.roff

    @property
    def conductance(self) -> float:
        """Ideal conductance (no noise) based on current state (S)."""
        return self.g_off + self._w * (self.g_on - self.g_off)

    def read_conductance(self) -> float:
        """Read conductance with device variation + cycle-to-cycle noise.

        This is what the circuit actually 'sees' during inference.
        """
        g = self.conductance * self._d2d_factor
        # Cycle-to-cycle Gaussian noise
        g *= float(self._rng.normal(1.0, self.read_noise_sigma))
        # Clamp to physical bounds (with variation)
        g_min = self.g_off * 0.5
        g_max = self.g_on * 2.0
        return float(np.clip(g, g_min, g_max))

    # ------------------------------------------------------------------
    # Switching
    # ------------------------------------------------------------------

    def write_voltage(self, v: float) -> None:
        """Apply write voltage; switch state if |v| exceeds threshold.

        Programming uses non-linear sigmoid approach:
          Δw ∝ sigmoid(|V|/V_write - 1) · (target - w)

        Parameters
        ----------
        v : float
            Applied voltage. Positive → set toward LRS; Negative → reset to HRS.
        """
        if abs(v) < self.v_write * 0.1:
            return  # sub-threshold, no effect

        # Normalised overdrive
        overdrive = abs(v) / self.v_write
        # Sigmoid-shaped switching probability
        delta = 1.0 / (1.0 + np.exp(-5.0 * (overdrive - 1.0)))

        if v > 0:
            # SET: drive toward LRS (w → 1)
            self._w += delta * (1.0 - self._w)
        else:
            # RESET: drive toward HRS (w → 0)
            self._w -= delta * self._w

        self._w = float(np.clip(self._w, 0.0, 1.0))

    def reset(self) -> None:
        """Hard reset to initial state (w=0.5)."""
        self._w = 0.5

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        g = self.conductance
        return (
            f"MemristorDevice(G={g*1e6:.2f} µS, w={self._w:.3f}, "
            f"Ron={self.ron/1e3:.1f}kΩ, Roff={self.roff/1e6:.1f}MΩ)"
        )
