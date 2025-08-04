"""
Memristor Neural Network Simulator

Device-accurate simulator and RTL generator for memristive crossbar accelerators.
Features IEDM 2024 calibrated noise models and comprehensive design space exploration.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

from .core.crossbar import CrossbarArray
from .core.device_models import DeviceModel, DeviceConfig
from .mapping.neural_mapper import map_to_crossbar
from .simulator.simulator import simulate
from .rtl_gen.generator import RTLGenerator, ChiselGenerator
from .analysis.explorer import DesignSpaceExplorer
from .validation.validator import HardwareValidator
from .faults.analyzer import FaultAnalyzer

__all__ = [
    "CrossbarArray",
    "DeviceModel", 
    "DeviceConfig",
    "map_to_crossbar",
    "simulate",
    "RTLGenerator",
    "ChiselGenerator", 
    "DesignSpaceExplorer",
    "HardwareValidator",
    "FaultAnalyzer",
]