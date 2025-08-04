"""Input validation and sanitization utilities."""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Any, List
from pathlib import Path
import re


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_tensor(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "tensor"
) -> torch.Tensor:
    """
    Comprehensive tensor validation.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected exact shape
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        dtype: Expected data type
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name for error messages
        
    Returns:
        Validated tensor
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Check for NaN or infinite values
    if torch.isnan(tensor).any():
        raise ValidationError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise ValidationError(f"{name} contains infinite values")
    
    # Shape validation
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise ValidationError(f"{name} shape {tensor.shape} != expected {expected_shape}")
    
    # Dimension validation
    if min_dims is not None and tensor.ndim < min_dims:
        raise ValidationError(f"{name} has {tensor.ndim} dims, expected >= {min_dims}")
    
    if max_dims is not None and tensor.ndim > max_dims:
        raise ValidationError(f"{name} has {tensor.ndim} dims, expected <= {max_dims}")
    
    # Data type validation
    if dtype is not None and tensor.dtype != dtype:
        raise ValidationError(f"{name} dtype {tensor.dtype} != expected {dtype}")
    
    # Value range validation
    if min_value is not None:
        if tensor.min().item() < min_value:
            raise ValidationError(f"{name} contains values < {min_value}")
    
    if max_value is not None:
        if tensor.max().item() > max_value:
            raise ValidationError(f"{name} contains values > {max_value}")
    
    return tensor


def validate_numpy_array(
    array: np.ndarray,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "array"
) -> np.ndarray:
    """
    Comprehensive numpy array validation.
    
    Args:
        array: Input array to validate
        expected_shape: Expected exact shape
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        dtype: Expected data type
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        name: Name for error messages
        
    Returns:
        Validated array
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray, got {type(array)}")
    
    # Check for NaN or infinite values
    if np.isnan(array).any():
        raise ValidationError(f"{name} contains NaN values")
    
    if np.isinf(array).any():
        raise ValidationError(f"{name} contains infinite values")
    
    # Shape validation
    if expected_shape is not None:
        if array.shape != expected_shape:
            raise ValidationError(f"{name} shape {array.shape} != expected {expected_shape}")
    
    # Dimension validation
    if min_dims is not None and array.ndim < min_dims:
        raise ValidationError(f"{name} has {array.ndim} dims, expected >= {min_dims}")
    
    if max_dims is not None and array.ndim > max_dims:
        raise ValidationError(f"{name} has {array.ndim} dims, expected <= {max_dims}")
    
    # Data type validation
    if dtype is not None and array.dtype != dtype:
        raise ValidationError(f"{name} dtype {array.dtype} != expected {dtype}")
    
    # Value range validation
    if min_value is not None:
        if array.min() < min_value:
            raise ValidationError(f"{name} contains values < {min_value}")
    
    if max_value is not None:
        if array.max() > max_value:
            raise ValidationError(f"{name} contains values > {max_value}")
    
    return array


def validate_crossbar_params(
    rows: int,
    cols: int,
    tile_size: Optional[int] = None,
    device_model: str = None
) -> Tuple[int, int, int, str]:
    """
    Validate crossbar array parameters.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        tile_size: Tile size for large arrays
        device_model: Device model name
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If validation fails
    """
    # Validate dimensions
    if not isinstance(rows, int) or rows <= 0:
        raise ValidationError(f"rows must be positive integer, got {rows}")
    
    if not isinstance(cols, int) or cols <= 0:
        raise ValidationError(f"cols must be positive integer, got {cols}")
    
    # Reasonable size limits
    MAX_SIZE = 10000
    if rows > MAX_SIZE or cols > MAX_SIZE:
        raise ValidationError(f"Crossbar size too large (max {MAX_SIZE}x{MAX_SIZE})")
    
    # Validate tile size
    if tile_size is None:
        tile_size = min(128, max(rows, cols))
    elif not isinstance(tile_size, int) or tile_size <= 0:
        raise ValidationError(f"tile_size must be positive integer, got {tile_size}")
    
    # Validate device model
    valid_models = ['IEDM2024_TaOx', 'IEDM2024_HfOx']
    if device_model is not None and device_model not in valid_models:
        raise ValidationError(f"device_model must be one of {valid_models}, got {device_model}")
    
    device_model = device_model or 'IEDM2024_TaOx'
    
    return rows, cols, tile_size, device_model


def validate_positive_number(
    value: Union[int, float],
    name: str = "value",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None
) -> Union[int, float]:
    """
    Validate positive number.
    
    Args:
        value: Number to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    
    return value


def validate_probability(value: float, name: str = "probability") -> float:
    """
    Validate probability value (0 <= p <= 1).
    
    Args:
        value: Probability to validate
        name: Parameter name for error messages
        
    Returns:
        Validated probability
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if not (0 <= value <= 1):
        raise ValidationError(f"{name} must be in [0, 1], got {value}")
    
    return float(value)


def validate_temperature(temperature: float) -> float:
    """
    Validate temperature in Kelvin.
    
    Args:
        temperature: Temperature to validate
        
    Returns:
        Validated temperature
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError(f"temperature must be numeric, got {type(temperature)}")
    
    if np.isnan(temperature) or np.isinf(temperature):
        raise ValidationError(f"temperature must be finite, got {temperature}")
    
    # Absolute zero to reasonable max (1000K)
    if not (0 < temperature <= 1000):
        raise ValidationError(f"temperature must be in (0, 1000] K, got {temperature}")
    
    return float(temperature)


def validate_string_list(
    values: List[str],
    valid_options: List[str],
    name: str = "values"
) -> List[str]:
    """
    Validate list of strings against valid options.
    
    Args:
        values: List of strings to validate
        valid_options: Valid string options
        name: Parameter name for error messages
        
    Returns:
        Validated list
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(values, list):
        raise ValidationError(f"{name} must be a list, got {type(values)}")
    
    if not values:
        raise ValidationError(f"{name} cannot be empty")
    
    for value in values:
        if not isinstance(value, str):
            raise ValidationError(f"{name} must contain strings, got {type(value)}")
        
        if value not in valid_options:
            raise ValidationError(f"{name} contains invalid option '{value}', valid: {valid_options}")
    
    return values


def validate_batch_size(batch_size: int) -> int:
    """
    Validate batch size for simulation.
    
    Args:
        batch_size: Batch size to validate
        
    Returns:
        Validated batch size
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(batch_size, int):
        raise ValidationError(f"batch_size must be integer, got {type(batch_size)}")
    
    if batch_size <= 0:
        raise ValidationError(f"batch_size must be positive, got {batch_size}")
    
    if batch_size > 10000:
        raise ValidationError(f"batch_size too large (max 10000), got {batch_size}")
    
    return batch_size


def validate_device_config_params(config_dict: dict) -> dict:
    """
    Validate device configuration parameters.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If validation fails
    """
    validated = {}
    
    # Probability parameters
    prob_params = ['read_noise_sigma', 'ron_variation', 'roff_variation', 'drift_coefficient', 'stuck_at_rate']
    for param in prob_params:
        if param in config_dict:
            validated[param] = validate_probability(config_dict[param], param)
    
    # Temperature coefficient
    if 'temp_coefficient' in config_dict:
        val = config_dict['temp_coefficient']
        if not isinstance(val, (int, float)) or abs(val) > 0.1:
            raise ValidationError(f"temp_coefficient must be in [-0.1, 0.1], got {val}")
        validated['temp_coefficient'] = val
    
    # Temperature
    if 'temperature' in config_dict:
        validated['temperature'] = validate_temperature(config_dict['temperature'])
    
    return validated