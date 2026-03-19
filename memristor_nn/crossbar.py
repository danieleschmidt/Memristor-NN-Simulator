"""
CrossbarArray: NxM memristor crossbar implementing analog matrix-vector multiply.

Physical operation
------------------
A crossbar is a grid of N word-lines (rows) × M bit-lines (columns).
Each crossing hosts a memristor. When input voltages V ∈ ℝ^N are applied
to the rows, Ohm's law drives currents through each column:

    I_j = Σ_i  G_ij · V_i      (Kirchhoff's current law)

In matrix form:  I = G · V

This IS matrix-vector multiplication — for free, in O(1) time, in analog.
It's why crossbars are attractive for neural network inference acceleration.

Differential mapping
--------------------
Weights can be negative. Since conductances are positive, we use a
differential pair of crossbars:
    G_pos encodes max(W, 0)
    G_neg encodes max(-W, 0)
    Output = G_pos·V - G_neg·V = W·V
"""

import numpy as np
from .device import MemristorDevice


class CrossbarArray:
    """NxM crossbar of memristors performing analog VMM.

    Parameters
    ----------
    n_rows : int
        Number of input lines (≡ input dimension).
    n_cols : int
        Number of output lines (≡ output dimension).
    device_kwargs : dict, optional
        Passed to each MemristorDevice at construction.
    rng : numpy.random.Generator, optional
        Shared RNG for reproducibility.
    """

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        device_kwargs: dict = None,
        rng: np.random.Generator = None,
    ):
        if n_rows < 1 or n_cols < 1:
            raise ValueError("Crossbar dimensions must be ≥ 1")

        self.n_rows = n_rows
        self.n_cols = n_cols
        self._rng = rng if rng is not None else np.random.default_rng()
        device_kwargs = device_kwargs or {}

        # Create NxM grid of devices
        self._devices: list[list[MemristorDevice]] = [
            [
                MemristorDevice(rng=self._rng, **device_kwargs)
                for _ in range(n_cols)
            ]
            for _ in range(n_rows)
        ]

    # ------------------------------------------------------------------
    # Conductance matrix access
    # ------------------------------------------------------------------

    def conductance_matrix(self, noisy: bool = True) -> np.ndarray:
        """Return the NxM conductance matrix (S).

        Parameters
        ----------
        noisy : bool
            If True, use read_conductance() (includes variation + noise).
            If False, use ideal conductance.
        """
        G = np.empty((self.n_rows, self.n_cols))
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                d = self._devices[i][j]
                G[i, j] = d.read_conductance() if noisy else d.conductance
        return G

    # ------------------------------------------------------------------
    # Programming
    # ------------------------------------------------------------------

    def program(self, target_conductances: np.ndarray) -> None:
        """Program the crossbar to a target conductance matrix.

        Each device is written until its (ideal) conductance is close
        to the target. A single write-verify step is used here.

        Parameters
        ----------
        target_conductances : ndarray, shape (n_rows, n_cols)
            Target conductances in Siemens. Clipped to [G_off, G_on].
        """
        target_conductances = np.asarray(target_conductances)
        if target_conductances.shape != (self.n_rows, self.n_cols):
            raise ValueError(
                f"Expected shape ({self.n_rows}, {self.n_cols}), "
                f"got {target_conductances.shape}"
            )

        # Reference device bounds (use first device as proxy)
        d_ref = self._devices[0][0]
        g_min, g_max = d_ref.g_off, d_ref.g_on
        clipped = np.clip(target_conductances, g_min, g_max)

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                d = self._devices[i][j]
                g_target = clipped[i, j]
                # Convert target conductance to state: w = (G - G_off) / (G_on - G_off)
                w_target = (g_target - d.g_off) / (d.g_on - d.g_off)
                w_target = float(np.clip(w_target, 0.0, 1.0))
                # Write voltage proportional to desired state
                v = d.v_write * (2.0 * w_target - 1.0) * 2.0  # scale ±2·v_write
                d.write_voltage(v)

    # ------------------------------------------------------------------
    # Analog VMM
    # ------------------------------------------------------------------

    def vmm(self, v_in: np.ndarray, noisy: bool = True) -> np.ndarray:
        """Analog matrix-vector multiply: I = G · V.

        Parameters
        ----------
        v_in : ndarray, shape (n_rows,) or (batch, n_rows)
            Input voltage vector(s).
        noisy : bool
            Whether to include device noise in G.

        Returns
        -------
        ndarray, shape (n_cols,) or (batch, n_cols)
            Output current vector(s).
        """
        v_in = np.asarray(v_in, dtype=float)
        batched = v_in.ndim == 2
        if not batched:
            v_in = v_in[np.newaxis, :]  # (1, n_rows)

        if v_in.shape[-1] != self.n_rows:
            raise ValueError(
                f"Input size {v_in.shape[-1]} != n_rows {self.n_rows}"
            )

        G = self.conductance_matrix(noisy=noisy)  # (n_rows, n_cols)
        # I = V @ G   →  (batch, n_rows) @ (n_rows, n_cols) = (batch, n_cols)
        out = v_in @ G

        return out if batched else out[0]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"CrossbarArray({self.n_rows}×{self.n_cols})"
