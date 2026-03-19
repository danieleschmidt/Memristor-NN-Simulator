"""
demo_mlp.py — Memristive vs Software MLP comparison

Generates a synthetic classification dataset, trains a simple 2-layer MLP
using gradient descent in numpy (no torch required), then:

1. Loads the trained weights into a MemristiveNN (crossbar simulation)
2. Runs inference with and without noise
3. Compares accuracy: Software MLP vs Memristive (noiseless) vs Memristive (noisy)
4. Estimates inference energy for the crossbar-based network

Run with:
    ~/anaconda3/bin/python3 examples/demo_mlp.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from memristor_nn import MemristorDevice, CrossbarArray, MemristiveLayer, MemristiveNN, EnergyModel
from memristor_nn.network import SoftwareMLP


# ── Utilities ────────────────────────────────────────────────────────────────

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def cross_entropy(logits, labels):
    probs = softmax(logits)
    n = len(labels)
    return -np.mean(np.log(probs[np.arange(n), labels] + 1e-12))

def accuracy(preds, labels):
    return np.mean(preds == labels)


# ── Synthetic Dataset ─────────────────────────────────────────────────────────

def make_dataset(n=1000, d=32, n_classes=4, seed=42):
    rng = np.random.default_rng(seed)
    # Each class is a Gaussian blob
    centers = rng.normal(0, 2, (n_classes, d))
    X, y = [], []
    per_class = n // n_classes
    for c in range(n_classes):
        X.append(rng.normal(centers[c], 0.8, (per_class, d)))
        y.extend([c] * per_class)
    X = np.vstack(X).astype(float)
    y = np.array(y)
    # Shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ── Simple Numpy MLP Trainer ──────────────────────────────────────────────────

class NumpyTrainer:
    """Tiny SGD trainer for a 2-layer MLP (no framework needed)."""

    def __init__(self, in_dim, hidden, out_dim, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        scale = lambda n_in: np.sqrt(2.0 / n_in)
        self.W1 = rng.normal(0, scale(in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale(hidden), (hidden, out_dim))
        self.b2 = np.zeros(out_dim)
        self.lr = lr

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2

    def backward(self, X, y):
        n = len(y)
        probs = softmax(self.Z2)
        dZ2 = probs.copy()
        dZ2[np.arange(n), y] -= 1.0
        dZ2 /= n

        dW2 = self.A1.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=50, batch_size=64):
        rng = np.random.default_rng(0)
        n = len(y)
        for epoch in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                self.forward(X[batch])
                self.backward(X[batch], y[batch])
            if (epoch + 1) % 10 == 0:
                logits = self.forward(X)
                loss = cross_entropy(logits, y)
                acc = accuracy(np.argmax(logits, axis=-1), y)
                print(f"  epoch {epoch+1:3d}  loss={loss:.4f}  train_acc={acc:.3f}")


# ── Main Demo ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Memristive Neural Network Simulator — 2-Layer MLP Demo")
    print("=" * 65)

    # ── 1. Device physics intro ───────────────────────────────────────
    print("\n[1] Single Memristor Device\n")
    d = MemristorDevice(ron=1e4, roff=1e6, v_write=1.0,
                        rng=np.random.default_rng(0))
    print(f"  Initial:   {d}")
    d.write_voltage(2.0)   # SET → LRS
    print(f"  After +2V: {d}")
    d.write_voltage(-2.0)  # RESET → HRS
    print(f"  After -2V: {d}")

    # ── 2. Crossbar VMM ───────────────────────────────────────────────
    print("\n[2] CrossbarArray Analog VMM (4×3)\n")
    xb = CrossbarArray(4, 3, rng=np.random.default_rng(1))
    v = np.array([0.1, 0.2, 0.15, 0.05])
    G = xb.conductance_matrix(noisy=False)
    i_ideal = xb.vmm(v, noisy=False)
    i_noisy = xb.vmm(v, noisy=True)
    print(f"  Input V: {v}")
    print(f"  G (ideal): shape {G.shape}, mean G={G.mean()*1e6:.2f} µS")
    print(f"  I (ideal): {i_ideal}")
    print(f"  I (noisy): {i_noisy}")
    print(f"  SNR: {np.abs(i_ideal).mean() / np.abs(i_noisy - i_ideal).mean():.1f}")

    # ── 3. Train MLP ──────────────────────────────────────────────────
    print("\n[3] Training 2-Layer Software MLP (32→64→4)\n")
    X, y = make_dataset(n=2000, d=32, n_classes=4, seed=0)
    split = 1600
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    trainer = NumpyTrainer(in_dim=32, hidden=64, out_dim=4, lr=0.05)
    trainer.train(X_train, y_train, epochs=60, batch_size=128)

    # Software MLP accuracy
    logits_sw = trainer.forward(X_test)
    preds_sw = np.argmax(logits_sw, axis=-1)
    acc_sw = accuracy(preds_sw, y_test)
    print(f"\n  Software MLP test accuracy: {acc_sw:.3f}")

    # ── 4. Load into MemristiveNN ─────────────────────────────────────
    print("\n[4] Loading weights into MemristiveNN (crossbar simulation)\n")

    weights = [trainer.W1, trainer.W2]
    biases  = [trainer.b1, trainer.b2]

    # Noiseless variant
    mem_net_noiseless = MemristiveNN(
        [32, 64, 4], activation="relu",
        device_kwargs={"d2d_sigma": 0.0, "read_noise_sigma": 0.0},
        rng=np.random.default_rng(5),
    )
    mem_net_noiseless.load_weights(weights, biases)
    preds_mem0 = mem_net_noiseless.predict(X_test, noisy=False)
    acc_mem0 = accuracy(preds_mem0, y_test)
    print(f"  Memristive NN (noiseless) test accuracy: {acc_mem0:.3f}")

    # Noisy variant (realistic device variation)
    mem_net_noisy = MemristiveNN(
        [32, 64, 4], activation="relu",
        device_kwargs={"d2d_sigma": 0.05, "read_noise_sigma": 0.02},
        rng=np.random.default_rng(5),
    )
    mem_net_noisy.load_weights(weights, biases)
    preds_mem1 = mem_net_noisy.predict(X_test, noisy=True)
    acc_mem1 = accuracy(preds_mem1, y_test)
    print(f"  Memristive NN (noisy, σ_d2d=5%, σ_read=2%) test accuracy: {acc_mem1:.3f}")

    # ── 5. Accuracy comparison ────────────────────────────────────────
    print("\n[5] Accuracy Comparison\n")
    print(f"  {'Model':<40}  {'Test Acc':>8}")
    print(f"  {'-'*40}  {'-'*8}")
    print(f"  {'Software MLP (float64)':<40}  {acc_sw:>8.3f}")
    print(f"  {'Memristive NN (noiseless crossbar)':<40}  {acc_mem0:>8.3f}")
    print(f"  {'Memristive NN (realistic noise)':<40}  {acc_mem1:>8.3f}")
    delta = acc_mem1 - acc_sw
    print(f"\n  Accuracy delta (noisy memristive vs software): {delta:+.3f}")

    # ── 6. Energy estimate ────────────────────────────────────────────
    print("\n[6] Energy Estimate (single inference)\n")
    em = EnergyModel(t_read=10e-9, v_read=0.1, peripheral_overhead=3.0)
    x0 = X_test[0]
    energy = em.estimate(mem_net_noisy, x0)

    print(f"  Total MACs:               {energy['total_macs']}")
    print(f"  Crossbar energy:          {energy['crossbar_energy_j']*1e9:.4f} nJ")
    print(f"  Total energy (w/ periph): {energy['total_energy_nj']:.4f} nJ")
    print(f"  CMOS equiv. (1fJ/MAC):    {energy['cmos_equivalent_j']*1e15:.2f} fJ")
    print(f"  Energy savings vs CMOS:   {energy['savings_factor']:.2e}×")
    print(f"\n  Note: crossbar VMM reads all weights in parallel in one pulse,")
    print(f"  while CMOS must serially fetch and multiply each weight.")

    print("\n" + "=" * 65)
    print("  Demo complete!")
    print("=" * 65)


if __name__ == "__main__":
    main()
