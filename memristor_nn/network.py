"""
MemristiveNN: multi-layer perceptron built from MemristiveLayers.

Acts as a drop-in analog hardware equivalent of a standard software MLP.
Each linear layer runs on a memristor crossbar pair; activations run in
digital peripheral circuits (ADC/DAC chain + digital activation logic).

Usage
-----
    net = MemristiveNN([784, 256, 10], activation='relu')
    net.load_weights(weights_list, biases_list)
    logits = net.forward(x)

Comparison with standard MLP
-----------------------------
A standard (software) MLP is also available via `SoftwareMLP` for accuracy
comparison — same weight matrices, purely floating-point computation.
"""

import numpy as np
from .layers import MemristiveLayer


# ------------------------------------------------------------------
# Activation functions
# ------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "linear": lambda x: x,
}


# ------------------------------------------------------------------
# MemristiveNN
# ------------------------------------------------------------------

class MemristiveNN:
    """MLP implemented on memristive crossbar layers.

    Parameters
    ----------
    layer_sizes : list[int]
        e.g. [128, 64, 10] → one hidden layer of 64 then output of 10.
        First element is input size.
    activation : str or list[str]
        Hidden-layer activation. Last layer always 'linear' unless
        explicitly set via list.
    device_kwargs : dict, optional
    rng : numpy.random.Generator, optional
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        device_kwargs: dict = None,
        rng: np.random.Generator = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output sizes")

        self._rng = rng if rng is not None else np.random.default_rng()
        self.layer_sizes = list(layer_sizes)

        n_layers = len(layer_sizes) - 1

        # Build activation list
        if isinstance(activation, str):
            acts = [activation] * (n_layers - 1) + ["linear"]
        else:
            acts = list(activation)
            if len(acts) != n_layers:
                raise ValueError(
                    f"activation list length {len(acts)} != n_layers {n_layers}"
                )
        self._activations = [ACTIVATIONS[a] for a in acts]
        self._activation_names = acts

        # Build memristive layers
        self.layers: list[MemristiveLayer] = []
        for i in range(n_layers):
            self.layers.append(
                MemristiveLayer(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    device_kwargs=device_kwargs,
                    rng=self._rng,
                )
            )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        weights: list,
        biases: list = None,
    ) -> None:
        """Program crossbars with pre-trained weight matrices.

        Parameters
        ----------
        weights : list of ndarray
            One per layer, shape (in, out).
        biases : list of ndarray or None
            One per layer, shape (out,). None → zero biases.
        """
        if len(weights) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} weight matrices, got {len(weights)}"
            )
        for i, layer in enumerate(self.layers):
            b = biases[i] if biases is not None else None
            layer.set_weights(weights[i], b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        """Run inference through all crossbar layers.

        Parameters
        ----------
        x : ndarray, shape (in_features,) or (batch, in_features)
        noisy : bool

        Returns
        -------
        ndarray
        """
        h = np.asarray(x, dtype=float)
        for layer, act_fn in zip(self.layers, self._activations):
            h = layer(h, noisy=noisy)
            h = act_fn(h)
        return h

    def predict(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        """Return class predictions (argmax of logits)."""
        logits = self.forward(x, noisy=noisy)
        return np.argmax(logits, axis=-1)

    def __call__(self, x: np.ndarray, noisy: bool = True) -> np.ndarray:
        return self.forward(x, noisy=noisy)

    def __repr__(self) -> str:
        parts = []
        for i, (layer, act) in enumerate(
            zip(self.layers, self._activation_names)
        ):
            parts.append(f"  [{i}] {layer}  →  {act}")
        return "MemristiveNN(\n" + "\n".join(parts) + "\n)"


# ------------------------------------------------------------------
# SoftwareMLP — pure numpy reference for accuracy comparison
# ------------------------------------------------------------------

class SoftwareMLP:
    """Standard (non-memristive) MLP in pure numpy for accuracy comparison.

    Shares the same weight/bias loading interface as MemristiveNN.
    """

    def __init__(self, layer_sizes: list, activation: str = "relu"):
        self.layer_sizes = list(layer_sizes)
        n_layers = len(layer_sizes) - 1
        if isinstance(activation, str):
            acts = [activation] * (n_layers - 1) + ["linear"]
        else:
            acts = list(activation)
        self._activations = [ACTIVATIONS[a] for a in acts]
        self._activation_names = acts
        self._weights = [None] * n_layers
        self._biases = [None] * n_layers

    def load_weights(self, weights: list, biases: list = None) -> None:
        for i in range(len(self._weights)):
            self._weights[i] = np.asarray(weights[i], dtype=float)
            self._biases[i] = (
                np.asarray(biases[i], dtype=float)
                if biases is not None
                else np.zeros(self.layer_sizes[i + 1])
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=float)
        for W, b, act_fn in zip(self._weights, self._biases, self._activations):
            h = h @ W + b
            h = act_fn(h)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(x), axis=-1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
