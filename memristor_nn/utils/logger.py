"""Comprehensive logging setup for memristor_nn package."""

import logging
import sys
import os
from typing import Optional
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "memristor_nn",
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_colors: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging for the package.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        enable_colors: Whether to use colored output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if enable_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        console_format = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Log startup message
    logger.info(f"Logger '{name}' initialized with level {level}")
    
    return logger


def get_logger(name: str = "memristor_nn") -> logging.Logger:
    """Get existing logger or create new one with default settings."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        class_name = self.__class__.__name__
        return get_logger(f"memristor_nn.{class_name}")


class PerformanceLogger:
    """Context manager for performance logging."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = datetime.now() - self.start_time
            duration_ms = duration.total_seconds() * 1000
            
            if exc_type:
                self.logger.error(f"{self.operation_name} failed after {duration_ms:.2f}ms: {exc_val}")
            else:
                self.logger.info(f"{self.operation_name} completed in {duration_ms:.2f}ms")


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    """Log system information for debugging."""
    logger = logger or get_logger()
    
    try:
        import platform
        import psutil
        import torch
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("CUDA not available")
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info("===========================")
        
    except ImportError as e:
        logger.warning(f"Could not log system info: {e}")


def setup_error_handling(logger: Optional[logging.Logger] = None) -> None:
    """Setup global error handling."""
    logger = logger or get_logger()
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    sys.excepthook = handle_exception
    logger.info("Global error handling enabled")