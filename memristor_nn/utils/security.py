"""Security utilities for input sanitization and validation."""

import os
import re
from pathlib import Path
from typing import Any, Union, List, Dict
import hashlib
import secrets


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


def sanitize_input(
    value: Any,
    max_length: int = 1000,
    allowed_chars: str = None
) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        allowed_chars: Regex pattern for allowed characters
        
    Returns:
        Sanitized string
        
    Raises:
        SecurityError: If input is dangerous
    """
    if value is None:
        return ""
    
    # Convert to string
    sanitized = str(value)
    
    # Check length
    if len(sanitized) > max_length:
        raise SecurityError(f"Input too long: {len(sanitized)} > {max_length}")
    
    # Remove dangerous characters
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'onload\s*=',                # Event handlers
        r'onerror\s*=',
        r'onclick\s*=',
        r'[\x00-\x1f\x7f-\x9f]',     # Control characters
    ]
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    # Check allowed characters
    if allowed_chars:
        if not re.match(f'^[{allowed_chars}]*$', sanitized):
            raise SecurityError(f"Input contains invalid characters: {sanitized}")
    
    return sanitized


def validate_file_path(
    file_path: Union[str, Path],
    allowed_extensions: List[str] = None,
    must_exist: bool = False,
    max_size_mb: float = 100.0
) -> Path:
    """
    Validate and sanitize file paths to prevent directory traversal.
    
    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions
        must_exist: Whether file must exist
        max_size_mb: Maximum file size in MB
        
    Returns:
        Validated Path object
        
    Raises:
        SecurityError: If path is dangerous
    """
    if not file_path:
        raise SecurityError("File path cannot be empty")
    
    path = Path(file_path).resolve()
    
    # Check for directory traversal
    if '..' in str(path):
        raise SecurityError(f"Directory traversal detected: {path}")
    
    # Check for absolute paths outside allowed directories
    cwd = Path.cwd().resolve()
    try:
        path.relative_to(cwd)
    except ValueError:
        raise SecurityError(f"Path outside working directory: {path}")
    
    # Check file extension
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise SecurityError(f"Invalid file extension: {path.suffix}")
    
    # Check if file exists (if required)
    if must_exist:
        if not path.exists():
            raise SecurityError(f"File does not exist: {path}")
        
        if not path.is_file():
            raise SecurityError(f"Path is not a file: {path}")
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise SecurityError(f"File too large: {size_mb:.1f}MB > {max_size_mb}MB")
    
    return path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent filesystem attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
        
    Raises:
        SecurityError: If filename is dangerous
    """
    if not filename:
        raise SecurityError("Filename cannot be empty")
    
    # Remove path separators
    sanitized = filename.replace('/', '_').replace('\\', '_')
    
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '_', sanitized)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Prevent reserved names on Windows
    reserved_names = [
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    ]
    
    base_name = sanitized.split('.')[0].upper()
    if base_name in reserved_names:
        sanitized = f"_{sanitized}"
    
    # Ensure minimum length
    if len(sanitized) < 1:
        sanitized = "file"
    
    # Limit length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def validate_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate network configuration to prevent malicious settings.
    
    Args:
        config: Network configuration dictionary
        
    Returns:
        Validated configuration
        
    Raises:
        SecurityError: If configuration is dangerous
    """
    validated = {}
    
    # Allowed configuration keys
    allowed_keys = {
        'rows', 'cols', 'tile_size', 'device_model', 'batch_size',
        'temperature', 'voltage', 'frequency', 'power_budget', 'area_budget'
    }
    
    for key, value in config.items():
        # Check allowed keys
        key_sanitized = sanitize_input(key, max_length=50, allowed_chars=r'a-zA-Z0-9_')
        if key_sanitized not in allowed_keys:
            raise SecurityError(f"Invalid configuration key: {key}")
        
        # Validate values based on type
        if key in ['rows', 'cols', 'tile_size', 'batch_size']:
            if not isinstance(value, int) or value <= 0 or value > 100000:
                raise SecurityError(f"Invalid {key}: {value}")
        elif key in ['temperature', 'voltage', 'frequency', 'power_budget', 'area_budget']:
            if not isinstance(value, (int, float)) or value <= 0 or value > 1e6:
                raise SecurityError(f"Invalid {key}: {value}")
        elif key == 'device_model':
            allowed_models = ['IEDM2024_TaOx', 'IEDM2024_HfOx']
            if value not in allowed_models:
                raise SecurityError(f"Invalid device model: {value}")
        
        validated[key_sanitized] = value
    
    return validated


def generate_secure_token(length: int = 32) -> str:
    """
    Generate cryptographically secure random token.
    
    Args:
        length: Token length in bytes
        
    Returns:
        Hex-encoded secure token
    """
    return secrets.token_hex(length)


def hash_sensitive_data(data: str, salt: str = None) -> str:
    """
    Hash sensitive data with salt for secure storage.
    
    Args:
        data: Data to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Hashed data (hex-encoded)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Combine data and salt
    combined = f"{data}{salt}".encode('utf-8')
    
    # Hash with SHA-256
    hash_obj = hashlib.sha256(combined)
    return f"{salt}:{hash_obj.hexdigest()}"


def verify_hash(data: str, hashed: str) -> bool:
    """
    Verify data against hash.
    
    Args:
        data: Original data
        hashed: Hashed data with salt
        
    Returns:
        True if verification succeeds
    """
    try:
        salt, hash_value = hashed.split(':', 1)
        recomputed = hash_sensitive_data(data, salt)
        return recomputed == hashed
    except ValueError:
        return False


def check_memory_usage(max_mb: float = 1000.0) -> None:
    """
    Check memory usage to prevent DoS attacks.
    
    Args:
        max_mb: Maximum allowed memory usage in MB
        
    Raises:
        SecurityError: If memory usage exceeds limit
    """
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        if memory_mb > max_mb:
            raise SecurityError(f"Memory usage too high: {memory_mb:.1f}MB > {max_mb}MB")
            
    except ImportError:
        # psutil not available, skip check
        pass


def rate_limit_check(operation: str, max_calls: int = 100, window_seconds: int = 60) -> None:
    """
    Simple rate limiting to prevent abuse.
    
    Args:
        operation: Operation identifier
        max_calls: Maximum calls per window
        window_seconds: Time window in seconds
        
    Raises:
        SecurityError: If rate limit exceeded
    """
    import time
    from collections import defaultdict, deque
    
    # Simple in-memory rate limiter (not persistent)
    if not hasattr(rate_limit_check, 'counters'):
        rate_limit_check.counters = defaultdict(deque)
    
    now = time.time()
    counter = rate_limit_check.counters[operation]
    
    # Remove old entries
    while counter and counter[0] < now - window_seconds:
        counter.popleft()
    
    # Check limit
    if len(counter) >= max_calls:
        raise SecurityError(f"Rate limit exceeded for {operation}: {len(counter)} calls in {window_seconds}s")
    
    # Add current call
    counter.append(now)


class SecureConfig:
    """Secure configuration manager."""
    
    def __init__(self):
        self._config = {}
        self._encrypted = False
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value securely."""
        key = sanitize_input(key, max_length=100, allowed_chars=r'a-zA-Z0-9_.')
        
        # Hash sensitive values
        if 'password' in key.lower() or 'secret' in key.lower() or 'token' in key.lower():
            value = hash_sensitive_data(str(value))
        
        self._config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        key = sanitize_input(key, max_length=100, allowed_chars=r'a-zA-Z0-9_.')
        return self._config.get(key, default)
    
    def validate_all(self) -> None:
        """Validate all configuration values."""
        for key, value in self._config.items():
            if isinstance(value, str):
                sanitize_input(value, max_length=10000)
            elif isinstance(value, (int, float)):
                if abs(value) > 1e12:
                    raise SecurityError(f"Configuration value too large: {key}={value}")
    
    def export_safe(self) -> Dict[str, Any]:
        """Export configuration with sensitive values masked."""
        safe_config = {}
        
        for key, value in self._config.items():
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'token']):
                safe_config[key] = '***HIDDEN***'
            else:
                safe_config[key] = value
        
        return safe_config