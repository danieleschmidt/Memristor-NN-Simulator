"""
Memristor Neural Network Simulator

Simulates how memristor crossbar arrays implement neural networks in analog hardware.
Covers device physics, crossbar VMM, noise models, energy estimation, and demo MLP.
"""

from .device import MemristorDevice
from .crossbar import CrossbarArray
from .layers import MemristiveLayer
from .network import MemristiveNN
from .energy import EnergyModel

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

__all__ = [
    "MemristorDevice",
    "CrossbarArray",
    "MemristiveLayer",
    "MemristiveNN",
    "EnergyModel",
]
