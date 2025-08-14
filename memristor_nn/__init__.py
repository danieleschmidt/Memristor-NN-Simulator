"""
Memristor Neural Network Simulator

Device-accurate simulator and RTL generator for memristive crossbar accelerators.
Features IEDM 2024 calibrated noise models and comprehensive design space exploration.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Optional imports for testing without external dependencies
try:
    from .core.device_models import DeviceModel, DeviceConfig
    CORE_AVAILABLE = True
except ImportError:
    DeviceModel = None
    DeviceConfig = None
    CORE_AVAILABLE = False

# Optional imports for testing without torch
try:
    from .core.crossbar import CrossbarArray
    from .mapping.neural_mapper import map_to_crossbar
    from .simulator.simulator import simulate
    from .rtl_gen.generator import RTLGenerator, ChiselGenerator
    from .analysis.explorer import DesignSpaceExplorer
    from .validation.validator import HardwareValidator
    from .faults.analyzer import FaultAnalyzer
    TORCH_AVAILABLE = True
except ImportError:
    CrossbarArray = None
    map_to_crossbar = None  
    simulate = None
    RTLGenerator = None
    ChiselGenerator = None
    DesignSpaceExplorer = None
    HardwareValidator = None
    FaultAnalyzer = None
    TORCH_AVAILABLE = False

__all__ = [
    "RTLGenerator",
    "ChiselGenerator", 
    "DesignSpaceExplorer",
    "HardwareValidator",
    "FaultAnalyzer",
    "TORCH_AVAILABLE",
    "CORE_AVAILABLE",
]

# Add core items if available
if CORE_AVAILABLE:
    __all__.extend(["DeviceModel", "DeviceConfig"])

# Add torch-dependent items if available
if TORCH_AVAILABLE:
    __all__.extend(["CrossbarArray", "map_to_crossbar", "simulate"])