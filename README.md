# Memristor-NN-Simulator

A physics-based simulator for **memristive crossbar neural networks** — modeling how resistive switching devices can implement neural network inference in analog hardware.

## What This Is

Memristors are two-terminal resistive devices that retain their resistance state after power is removed. A crossbar array of memristors implements **matrix-vector multiplication (VMM) in a single analog step** using Ohm's law and Kirchhoff's current law:

```
I = G · V
```

where **G** is the conductance matrix (stored as device states), **V** is the input voltage vector, and **I** is the output current — the result of the multiply-accumulate operation. This is the core of a fully-connected neural network layer, executed in O(1) time in analog hardware.

This simulator models that hardware at the device, circuit, and system level — complete with realistic noise models, differential weight encoding, and energy estimation.

---

## Architecture

```
MemristorDevice        → single device physics (Ron/Roff, switching, noise)
    └── CrossbarArray  → NxM grid, analog VMM I=G·V
        └── MemristiveLayer  → weight mapping + differential crossbar pair
            └── MemristiveNN → stacked layers + activations (full MLP)
                └── EnergyModel → power = V²×G, energy per inference
```

---

## Physics

### Device Model (`MemristorDevice`)

Each memristor has:
- **Ron** (low-resistance state, LRS): ~10 kΩ — device is "on" (conducting)
- **Roff** (high-resistance state, HRS): ~1 MΩ — device is "off" (insulating)
- **State variable** `w ∈ [0, 1]`: models internal boundary (e.g., oxygen vacancy front in TaOx)
- **Conductance**: `G = G_off + w × (G_on - G_off)`
- **Switching**: non-linear sigmoid response to applied voltage vs. `V_write` threshold
- **Noise**: device-to-device variation (log-normal σ) + cycle-to-cycle read noise (Gaussian)

### Crossbar VMM (`CrossbarArray`)

```
Word-lines (rows) ——→ [G00][G01][G02]
                  ——→ [G10][G11][G12]
                  ——→ [G20][G21][G22]
                         ↓   ↓   ↓
                    Bit-lines (cols)  →  output currents I
```

Each column sums currents via KCL: `I_j = Σ_i G_ij · V_i`

### Weight Encoding (`MemristiveLayer`)

Weights W ∈ ℝ can be negative; conductances are ≥ 0. We use a **differential pair**:

```
G_pos = max(W_norm, 0) × (G_on - G_off) + G_off
G_neg = max(-W_norm, 0) × (G_on - G_off) + G_off
Output = (G_pos · x) - (G_neg · x)
```

This cancels common-mode offsets and correctly handles sign.

### Energy (`EnergyModel`)

During a read pulse of duration `T_read`, each device dissipates:
```
P_ij = V_i² × G_ij   (Watts)
E_xbar = T_read × Σ_ij P_ij   (Joules)
```

Peripheral circuit overhead (ADC, DAC, digital activation) adds a multiplier (default 3×).

---

## Installation

```bash
git clone https://github.com/danieleschmidt/Memristor-NN-Simulator
cd Memristor-NN-Simulator
pip install numpy   # only dependency
```

---

## Quickstart

```python
import numpy as np
from memristor_nn import MemristorDevice, CrossbarArray, MemristiveNN, EnergyModel

# Single device
d = MemristorDevice(ron=1e4, roff=1e6, v_write=1.0)
d.write_voltage(2.0)   # SET → low-resistance state
print(d)               # MemristorDevice(G=99.67 µS, w=0.997, ...)

# Crossbar analog VMM
xb = CrossbarArray(4, 3)
v_in = np.array([0.1, 0.2, 0.15, 0.05])
i_out = xb.vmm(v_in, noisy=True)  # includes device noise

# Full network
net = MemristiveNN([32, 64, 4], activation="relu")
net.load_weights(weight_matrices, bias_vectors)
logits = net.forward(x_test, noisy=True)

# Energy estimate
em = EnergyModel(t_read=10e-9)
energy = em.estimate(net, x_test[0])
print(f"{energy['total_energy_nj']:.4f} nJ per inference")
```

---

## Demo

Run the full demo — trains a 2-layer MLP, loads weights onto simulated crossbars, compares accuracy, and estimates energy:

```bash
python examples/demo_mlp.py
```

**Expected output:**
```
[1] Single Memristor Device
  After +2V: G=99.67 µS, w=0.997
  After -2V: G=1.66 µS, w=0.007

[5] Accuracy Comparison
  Software MLP (float64)                       1.000
  Memristive NN (noiseless crossbar)           1.000
  Memristive NN (noisy, σ_d2d=5%, σ_read=2%)  1.000

[6] Energy Estimate
  Total MACs:               2304
  Crossbar energy:          0.0140 nJ
  Total energy (w/ periph): 0.0421 nJ
```

---

## Tests

```bash
python -m pytest tests/ -v
# 39 passed
```

---

## Module Reference

| Module | Class | Description |
|--------|-------|-------------|
| `device.py` | `MemristorDevice` | Single device physics, switching, noise |
| `crossbar.py` | `CrossbarArray` | NxM crossbar, conductance matrix, VMM |
| `layers.py` | `MemristiveLayer` | Differential crossbar pair, weight mapping |
| `network.py` | `MemristiveNN` | Full MLP on crossbars + activations |
| `network.py` | `SoftwareMLP` | Reference software MLP (numpy, no crossbar) |
| `energy.py` | `EnergyModel` | Per-inference energy estimation |

---

## Why Memristors?

| Property | CMOS SRAM | Memristor |
|----------|-----------|-----------|
| Compute paradigm | Digital, serial | Analog, parallel |
| VMM complexity | O(N²) ops | O(1) analog |
| Non-volatility | No | Yes |
| Energy/MAC | ~1–10 fJ | Potentially 0.01–1 fJ |
| Area | Larger | Denser (4F² cell) |

The catch: analog noise degrades precision (typically 4–8 effective bits). This simulator helps quantify that tradeoff.

---

## References

- Prezioso et al., "Training and operation of an integrated neuromorphic network based on metal-oxide memristors," *Nature* 521, 61–64 (2015)
- Yao et al., "Fully hardware-implemented memristor convolutional neural network," *Nature* 577, 641–646 (2020)
- Waser & Aono, "Nanoionics-based resistive switching memories," *Nature Materials* 6, 833–840 (2007)
- Strukov et al., "The missing memristor found," *Nature* 453, 80–83 (2008)

---

## License

MIT
