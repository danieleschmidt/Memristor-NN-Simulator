"""Utility functions and logging setup."""

from .logger import setup_logger, get_logger
from .security import sanitize_input, validate_file_path

# Optional torch-dependent validators
try:
    from .validators import validate_tensor, validate_crossbar_params
    VALIDATORS_AVAILABLE = True
    __all__ = ["setup_logger", "get_logger", "validate_tensor", "validate_crossbar_params", "sanitize_input", "validate_file_path"]
except ImportError:
    VALIDATORS_AVAILABLE = False 
    __all__ = ["setup_logger", "get_logger", "sanitize_input", "validate_file_path"]