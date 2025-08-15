"""
Advanced error handling utilities for pipeline guard
"""

import time
import logging
import traceback
import functools
import threading
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit breaker triggered
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ErrorMetrics:
    """Error tracking metrics"""
    total_errors: int = 0
    error_rate: float = 0.0
    last_error_time: Optional[datetime] = None
    error_types: Dict[str, int] = field(default_factory=dict)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker is OPEN, calls blocked for {self.timeout}s")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout)
                
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
            
        self.failure_count = 0
        
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "time_until_retry": max(0, self.timeout - (time.time() - (self.last_failure_time or 0)))
        }


class RetryStrategy:
    """
    Configurable retry strategy with exponential backoff
    """
    
    def __init__(self,
                 max_attempts: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to functions"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_retry(func, *args, **kwargs)
        return wrapper
        
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    self.logger.error(f"Function {func.__name__} failed after {self.max_attempts} attempts")
                    break
                    
                delay = self._calculate_delay(attempt)
                self.logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
                
        raise last_exception
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt"""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)  # Add jitter ±25%
            
        return delay


class ErrorHandler:
    """
    Comprehensive error handling and tracking system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = ErrorMetrics()
        self.error_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()
        
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Centralized error handling with context and metrics
        """
        with self.lock:
            error_info = self._extract_error_info(error, context or {})
            self._update_metrics(error_info)
            self._trigger_callbacks(error_info)
            
            # Log the error
            self.logger.error(
                f"Error handled: {error_info['type']} - {error_info['message']}",
                extra={
                    "error_context": error_info['context'],
                    "stack_trace": error_info['stack_trace']
                }
            )
            
            return error_info
            
    def _extract_error_info(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive information from error"""
        error_type = type(error).__name__
        
        return {
            "type": error_type,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "stack_trace": traceback.format_exc(),
            "severity": self._determine_severity(error, context)
        }
        
    def _determine_severity(self, error: Exception, context: Dict[str, Any]) -> str:
        """Determine error severity level"""
        critical_errors = [
            "MemoryError",
            "SystemExit", 
            "KeyboardInterrupt",
            "OSError"
        ]
        
        warning_errors = [
            "TimeoutError",
            "ConnectionError",
            "HTTPError"
        ]
        
        error_type = type(error).__name__
        
        if error_type in critical_errors:
            return "critical"
        elif error_type in warning_errors:
            return "warning"
        elif "pipeline_id" in context:
            return "pipeline_error"
        else:
            return "error"
            
    def _update_metrics(self, error_info: Dict[str, Any]):
        """Update error metrics"""
        self.metrics.total_errors += 1
        self.metrics.last_error_time = datetime.now()
        
        error_type = error_info["type"]
        self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
        
        # Add to recent errors
        self.metrics.recent_errors.append({
            "timestamp": error_info["timestamp"],
            "type": error_type,
            "message": error_info["message"][:200]  # Truncate long messages
        })
        
        # Calculate error rate (errors per hour)
        recent_hour = datetime.now() - timedelta(hours=1)
        recent_error_count = len([
            e for e in self.metrics.recent_errors
            if datetime.fromisoformat(e["timestamp"]) > recent_hour
        ])
        self.metrics.error_rate = recent_error_count
        
    def _trigger_callbacks(self, error_info: Dict[str, Any]):
        """Trigger registered error callbacks"""
        error_type = error_info["type"]
        severity = error_info["severity"]
        
        # Trigger type-specific callbacks
        for callback in self.error_callbacks.get(error_type, []):
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"Error callback failed: {e}")
                
        # Trigger severity-specific callbacks
        for callback in self.error_callbacks.get(severity, []):
            try:
                callback(error_info)
            except Exception as e:
                self.logger.error(f"Severity callback failed: {e}")
                
    def register_callback(self, error_type_or_severity: str, callback: Callable):
        """Register callback for specific error types or severities"""
        self.error_callbacks[error_type_or_severity].append(callback)
        
    def add_circuit_breaker(self, name: str, circuit_breaker: CircuitBreaker):
        """Add a named circuit breaker"""
        self.circuit_breakers[name] = circuit_breaker
        
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name"""
        return self.circuit_breakers.get(name)
        
    def create_resilient_function(self, 
                                 name: str,
                                 func: Callable,
                                 retry_strategy: RetryStrategy = None,
                                 circuit_breaker: CircuitBreaker = None) -> Callable:
        """
        Create a resilient function with retry and circuit breaker
        """
        if retry_strategy:
            func = retry_strategy(func)
            
        if circuit_breaker:
            self.circuit_breakers[name] = circuit_breaker
            func = circuit_breaker(func)
            
        @functools.wraps(func)
        def resilient_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle_error(e, {"function": name, "args": str(args)[:200]})
                raise
                
        return resilient_wrapper
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        return {
            "total_errors": self.metrics.total_errors,
            "error_rate_per_hour": self.metrics.error_rate,
            "last_error_time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
            "error_types": dict(self.metrics.error_types),
            "recent_errors": list(self.metrics.recent_errors),
            "circuit_breakers": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    def reset_metrics(self):
        """Reset error metrics"""
        with self.lock:
            self.metrics = ErrorMetrics()
            
    def export_error_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Export error log for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            error for error in self.metrics.recent_errors
            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
        ]


# Predefined resilient decorators
def resilient_pipeline_operation(max_attempts: int = 3, circuit_breaker: bool = True):
    """
    Decorator for pipeline operations that need resilience
    """
    def decorator(func: Callable) -> Callable:
        retry_strategy = RetryStrategy(max_attempts=max_attempts, initial_delay=2.0)
        cb = CircuitBreaker(failure_threshold=5, timeout=300) if circuit_breaker else None
        
        if cb:
            func = cb(func)
        func = retry_strategy(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Could integrate with global error handler here
                raise
                
        return wrapper
    return decorator


def graceful_degradation(fallback_func: Callable = None):
    """
    Decorator that provides graceful degradation on failure
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Function {func.__name__} failed, using fallback: {e}")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    # Return a safe default based on function name/context
                    return None
        return wrapper
    return decorator