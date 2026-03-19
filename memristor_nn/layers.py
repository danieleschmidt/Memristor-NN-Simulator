"""
MemristiveLayer: maps a weight matrix to a crossbar pair and implements
an analog forward pass with realistic noise.

Weight mapping (differential scheme)
--------------------------------------
Real neural network weights W ∈ [-1, 1] (after normalisation).
Since conductances are non-negative, use two crossbars:
    G_pos = max(W_norm, 0) * (G_on - G_off) + G_off
    G_neg = max(-W_norm, 0) * (G_on - G_off) + G_off

Forward pass:  y = (G_pos - G_neg) · x + bias (if any)
             = (G_pos·x) - (G_neg·x)

This cancels the common-mode G_off·x offset.
"""

import numpy as np
from .crossbar import CrossbarArray
from .device import MemristorDevice


class MemristiveLayer:
    """Single fully-connected layer implemented on memristor crossbars.

    Parameters
    ----------
    in_features : int
    out_features : int
    use_bias : bool
        If True, a software bias vector is added (bias not stored on crossbar).
    device_kwargs : dict, optional
        Forwarded to CrossbarArray / MemristorDevice.
    rng : numpy.random.Generator, optional
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        device_kwargs: dict = None,
        rng: np.random.Generator = None,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self._rng = rng if rng is not None else np.random.default_rng()

        # Reference device for bounds
        _ref = MemristorDevice(**(device_kwargs or {}))
        self._g_on = _ref.g_on
        self._g_off = _ref.g_off

        # Two crossbars for differential weight encoding
        self._xbar_pos = CrossbarArray(
            in_features, out_features,
            device_kwargs=device_kwargs,
            rng=self._rng,
        )
        self._xbar_neg = CrossbarArray(
            in_features, out_features,
            device_kwargs=device_kwargs,
            rng=self._rng,
        )

        # Weight matrix and bias (floating-point reference)
        self._weights: np.ndarray = None  # shape (in_features, out_features)
        self._bias: np.ndarray = np.zeros(out_features) if use_bias else None

        # Scaling factor: max abs weight before normalisation
        self._w_scale: float = 1.0

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def set_weights(self, weights: np.ndarray, bias: np.ndarray = None) -> None:
        """Load a weight matrix onto the crossbar pair.

        Parameters
        ----------
        weights : ndarray, shape (in_features, out_features)
            Linear layer weight matrix.
        bias : ndarray, shape (out_features,), optional
        """
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.in_features, self.out_features):
            raise ValueError(
                f"Expected weights shape ({self.in_features}, {self.out_features}), "
                f"got {weights.shape}"
            )

        self._weights = weights.copy()
        if bias is not None:
            self._bias = np.asarray(bias, dtype=float).copy()

        # Normalise weights to [-1, 1]
        w_max = np.max(np.abs(weights))
        self._w_scale = w_max if w_max > 1e-12 else 1.0
        w_norm = weights / self._w_scale  # ∈ [-1, 1]

        # Map to conductances
        g_range = self._g_on - self._g_off
        g_pos = np.maximum(w_norm, 0) * g_range + self._g_off
        g_neg = np.maximum(-w_norm, 0) * g_range + self._g_off

        self._xbar_pos.program(g_pos)
        self._xbar_neg.program(g_neg)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        """Analog forward pass: y = W·x + b (with crossbar noise).

        Parameters
        ----------
        x : ndarray, shape (in_features,) or (batch, in_features)
        noisy : bool
            Pass-through to crossbar VMM.

        Returns
        -------
        ndarray, shape (out_features,) or (batch, out_features)
        """
        x = np.asarray(x, dtype=float)
        batched = x.ndim == 2

        # Scale input to voltage range of read voltage (0.1 V typ)
        ref = MemristorDevice()
        v_read = ref.v_read
        x_scaled = x * v_read  # map activations to small read voltages

        # Differential VMM
        i_pos = self._xbar_pos.vmm(x_scaled, noisy=noisy)
        i_neg = self._xbar_neg.vmm(x_scaled, noisy=noisy)

        # Recover weight-scaled output; undo input scaling and weight normalisation
        # I = G·V = G·(x·v_read), so weight_equiv = I / (v_read) / g_range * w_scale
        g_range = self._g_on - self._g_off
        out = (i_pos - i_neg) / (v_read * g_range) * self._w_scale

        # Add bias
        if self.use_bias and self._bias is not None:
            out = out + self._bias

        return out

    def __call__(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        return self.forward(x, noisy=noisy)

    def __repr__(self) -> str:
        return (
            f"MemristiveLayer({self.in_features}→{self.out_features}, "
            f"bias={self.use_bias})"
        )
