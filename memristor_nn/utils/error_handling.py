"""Advanced error handling and recovery mechanisms."""

import functools
import time
import traceback
from typing import Any, Callable, Optional, Union, Type, List
import logging
from contextlib import contextmanager

from .logger import get_logger
from .security import SecurityError
from .validators import ValidationError


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RetryError(Exception):
    """Exception raised when all retry attempts fail."""
    pass


class GracefulDegradationError(Exception):
    """Exception for graceful degradation scenarios."""
    pass


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    ignore_exceptions: tuple = (ValidationError, SecurityError),
    logger: Optional[logging.Logger] = None
):
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Exception types to retry on
        ignore_exceptions: Exception types to never retry
        logger: Optional logger for retry messages
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(f"retry.{func.__name__}")
            
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except ignore_exceptions as e:
                    # Don't retry these exceptions
                    _logger.warning(f"Non-retryable exception in {func.__name__}: {e}")
                    raise
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        _logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        _logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        break
            
            raise RetryError(f"Function {func.__name__} failed after {max_attempts} attempts: {last_exception}")
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Timeout before attempting to close circuit (seconds)
            expected_exception: Exception type to count as failures
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger("circuit_breaker")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
        self.logger.info("Circuit breaker reset to CLOSED")
    
    def _record_success(self):
        """Record successful operation."""
        if self.state == 'HALF_OPEN':
            self._reset()
    
    def _record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                self.logger.info("Circuit breaker moved to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self._can_attempt():
                raise CircuitBreakerError(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._record_success()
                return result
            except self.expected_exception as e:
                self._record_failure()
                raise
        
        return wrapper


class GracefulDegradation:
    """Graceful degradation for non-critical functionality."""
    
    def __init__(
        self,
        fallback_value: Any = None,
        fallback_func: Optional[Callable] = None,
        exceptions: tuple = (Exception,),
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize graceful degradation.
        
        Args:
            fallback_value: Value to return on failure
            fallback_func: Function to call on failure
            exceptions: Exception types to catch
            logger: Optional logger
        """
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
        self.exceptions = exceptions
        self.logger = logger or get_logger("graceful_degradation")
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except self.exceptions as e:
                self.logger.warning(f"Graceful degradation for {func.__name__}: {e}")
                
                if self.fallback_func:
                    try:
                        return self.fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback function failed: {fallback_error}")
                        return self.fallback_value
                
                return self.fallback_value
        
        return wrapper


@contextmanager
def error_context(operation_name: str, logger: Optional[logging.Logger] = None):
    """Context manager for enhanced error reporting."""
    _logger = logger or get_logger("error_context")
    
    try:
        _logger.debug(f"Starting operation: {operation_name}")
        yield
        _logger.debug(f"Operation completed: {operation_name}")
    except Exception as e:
        _logger.error(f"Operation failed: {operation_name}")
        _logger.error(f"Exception: {type(e).__name__}: {e}")
        _logger.debug(f"Traceback:\n{traceback.format_exc()}")
        raise


class ErrorCollector:
    """Collect and analyze errors for debugging."""
    
    def __init__(self, max_errors: int = 100):
        self.max_errors = max_errors
        self.errors = []
        self.error_counts = {}
        self.logger = get_logger("error_collector")
    
    def record_error(
        self,
        error: Exception,
        context: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Record an error with context."""
        error_info = {
            'timestamp': time.time(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context,
            'metadata': metadata or {},
            'traceback': traceback.format_exc()
        }
        
        # Add to errors list
        self.errors.append(error_info)
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)  # Remove oldest
        
        # Update counts
        error_type = error_info['type']
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(f"Error recorded: {error_type} in {context}")
    
    def get_error_summary(self) -> dict:
        """Get summary of collected errors."""
        if not self.errors:
            return {'total_errors': 0, 'error_types': {}}
        
        recent_errors = [e for e in self.errors if time.time() - e['timestamp'] < 3600]  # Last hour
        
        return {
            'total_errors': len(self.errors),
            'recent_errors': len(recent_errors),
            'error_types': self.error_counts.copy(),
            'latest_error': self.errors[-1] if self.errors else None
        }
    
    def get_frequent_errors(self, min_count: int = 3) -> List[dict]:
        """Get frequently occurring errors."""
        frequent = []
        
        for error_type, count in self.error_counts.items():
            if count >= min_count:
                # Find most recent example
                for error in reversed(self.errors):
                    if error['type'] == error_type:
                        frequent.append({
                            'type': error_type,
                            'count': count,
                            'latest_message': error['message'],
                            'latest_context': error['context']
                        })
                        break
        
        return sorted(frequent, key=lambda x: x['count'], reverse=True)


class HealthChecker:
    """System health monitoring and diagnostics."""
    
    def __init__(self):
        self.checks = {}
        self.logger = get_logger("health_checker")
    
    def register_check(self, name: str, check_func: Callable[[], bool], critical: bool = False):
        """Register a health check function."""
        self.checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_result': None,
            'last_check': None,
            'failure_count': 0
        }
        self.logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def run_checks(self) -> dict:
        """Run all health checks."""
        results = {
            'overall_health': 'HEALTHY',
            'timestamp': time.time(),
            'checks': {}
        }
        
        critical_failures = 0
        
        for name, check_info in self.checks.items():
            try:
                start_time = time.time()
                result = check_info['func']()
                duration = time.time() - start_time
                
                check_info['last_result'] = result
                check_info['last_check'] = time.time()
                
                if result:
                    check_info['failure_count'] = 0
                    status = 'PASS'
                else:
                    check_info['failure_count'] += 1
                    status = 'FAIL'
                    if check_info['critical']:
                        critical_failures += 1
                
                results['checks'][name] = {
                    'status': status,
                    'duration_ms': duration * 1000,
                    'critical': check_info['critical'],
                    'failure_count': check_info['failure_count']
                }
                
            except Exception as e:
                check_info['failure_count'] += 1
                if check_info['critical']:
                    critical_failures += 1
                
                results['checks'][name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'critical': check_info['critical'],
                    'failure_count': check_info['failure_count']
                }
                
                self.logger.error(f"Health check '{name}' failed: {e}")
        
        # Determine overall health
        if critical_failures > 0:
            results['overall_health'] = 'CRITICAL'
        elif any(check['status'] != 'PASS' for check in results['checks'].values()):
            results['overall_health'] = 'DEGRADED'
        
        return results
    
    def get_health_status(self) -> str:
        """Get quick health status."""
        results = self.run_checks()
        return results['overall_health']


# Global error collector instance
_global_error_collector = ErrorCollector()

def get_error_collector() -> ErrorCollector:
    """Get global error collector instance."""
    return _global_error_collector


# Decorator for automatic error collection
def collect_errors(context: str = None):
    """Decorator to automatically collect errors."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_context = context or func.__name__
                _global_error_collector.record_error(e, error_context)
                raise
        return wrapper
    return decorator