"""Core memristor simulation components."""

from .device_models import DeviceModel, DeviceConfig

# Optional import for torch-dependent components
try:
    from .crossbar import CrossbarArray
    CROSSBAR_AVAILABLE = True
    __all__ = ["CrossbarArray", "DeviceModel", "DeviceConfig"] 
except ImportError:
    CrossbarArray = None
    CROSSBAR_AVAILABLE = False
    __all__ = ["DeviceModel", "DeviceConfig"]