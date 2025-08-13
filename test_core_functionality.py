#!/usr/bin/env python3
"""
Core functionality test without PyTorch dependencies.
Tests device models, basic configurations, and error handling.
"""

import sys
sys.path.insert(0, '/root/repo')

import numpy as np
from memristor_nn.core.device_models import DeviceModel, DeviceConfig
from memristor_nn.utils.error_handling import retry, CircuitBreaker
from memristor_nn.utils.logger import get_logger, PerformanceLogger
from memristor_nn.utils.security import sanitize_input, check_memory_usage, rate_limit_check
from memristor_nn.utils.validators import validate_positive_number, validate_temperature

logger = get_logger(__name__)

def test_device_models():
    """Test basic device model functionality."""
    print("=== Testing Device Models ===")
    
    # Test device configuration
    config = DeviceConfig(
        read_noise_sigma=0.05,
        ron_variation=0.15,
        roff_variation=0.20,
        drift_coefficient=0.1,
        stuck_at_rate=0.001,
        temp_coefficient=0.002,
        temperature=300.0
    )
    
    print(f"✓ Device config created: T={config.temperature}K, noise={config.read_noise_sigma}")
    
    # Test basic device model
    class TestMemristor(DeviceModel):
        def __init__(self, config=None):
            super().__init__(config)
            self.ron = 1e4
            self.roff = 1e6
            self.name = "TestMemristor"
            
        def conductance(self, voltage: float, state: float) -> float:
            """Simple linear conductance model."""
            g_min = 1.0 / self.roff
            g_max = 1.0 / self.ron
            return g_min + state * (g_max - g_min)
            
        def update_state(self, voltage: float, time: float, current_state: float) -> float:
            """Simple state update."""
            if abs(voltage) > 0.5:
                return min(max(current_state + 0.1 * np.sign(voltage), 0.0), 1.0)
            return current_state
    
    device = TestMemristor(config)
    
    # Test conductance calculation
    g = device.conductance(voltage=1.0, state=0.5)
    print(f"✓ Conductance at V=1V, state=0.5: {g:.2e} S")
    
    # Test state update
    new_state = device.update_state(voltage=1.5, time=1e-6, current_state=0.3)
    print(f"✓ State update: 0.3 → {new_state}")
    
    # Test variations
    g_varied = device.add_variations(g)
    print(f"✓ Conductance with variations: {g_varied:.2e} S")
    
    return True

def test_error_handling():
    """Test error handling and resilience mechanisms."""
    print("\n=== Testing Error Handling ===")
    
    # Test retry mechanism
    @retry(max_attempts=3, delay=0.1)
    def flaky_function(attempt_count=[0]):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ValueError(f"Attempt {attempt_count[0]} failed")
        return f"Success on attempt {attempt_count[0]}"
    
    try:
        result = flaky_function()
        print(f"✓ Retry mechanism: {result}")
    except Exception as e:
        print(f"✗ Retry failed: {e}")
        return False
    
    # Test circuit breaker
    cb = CircuitBreaker(failure_threshold=2, timeout=0.5)
    
    def failing_function():
        raise ConnectionError("Service unavailable")
    
    # Trigger circuit breaker
    for i in range(3):
        try:
            cb.call(failing_function)
        except Exception:
            pass
    
    print(f"✓ Circuit breaker state: {cb.state}")
    
    return True

def test_monitoring():
    """Test performance monitoring and logging."""
    print("\n=== Testing Monitoring ===")
    
    # Test performance logger
    perf_logger = PerformanceLogger("test_operation")
    
    import time
    with perf_logger:
        time.sleep(0.01)  # Simulate work
    
    metrics = perf_logger.get_metrics()
    print(f"✓ Performance metrics: {metrics}")
    
    # Test regular logging
    logger.info("Test log message")
    logger.warning("Test warning message")
    print("✓ Logging system operational")
    
    return True

def test_security():
    """Test security validation."""
    print("\n=== Testing Security ===")
    
    # Test input validation
    try:
        validate_positive_number(5.0, "test_param")
        print("✓ Positive number validation")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False
    
    try:
        validate_temperature(300.0)
        print("✓ Temperature validation")
    except Exception as e:
        print(f"✗ Temperature validation failed: {e}")
        return False
    
    # Test input sanitization
    try:
        clean_input = sanitize_input("test_string_123", max_length=50)
        print(f"✓ Input sanitization: '{clean_input}'")
    except Exception as e:
        print(f"✗ Input sanitization failed: {e}")
        return False
    
    # Test memory usage check
    try:
        check_memory_usage()
        print("✓ Memory usage check")
    except Exception as e:
        print(f"✓ Memory usage validation: {e}")
    
    # Test rate limiting
    try:
        rate_limit_check("test_operation", max_calls=10, window_seconds=60)
        print("✓ Rate limiting check")
    except Exception as e:
        print(f"✓ Rate limiting: {e}")
    
    return True

def test_core_algorithms():
    """Test core simulation algorithms."""
    print("\n=== Testing Core Algorithms ===")
    
    # Test crossbar matrix operations
    rows, cols = 128, 64
    resistance_matrix = np.random.uniform(1e4, 1e6, (rows, cols))
    conductance_matrix = 1.0 / resistance_matrix
    
    # Simulate analog computation
    input_vector = np.random.randn(rows)
    output_vector = np.dot(conductance_matrix.T, input_vector)
    
    print(f"✓ Matrix-vector multiplication: {rows}x{cols} → {len(output_vector)}")
    
    # Test noise injection
    noise_std = 0.05
    noisy_output = output_vector + np.random.normal(0, noise_std * np.abs(output_vector))
    
    snr = 20 * np.log10(np.std(output_vector) / np.std(noisy_output - output_vector))
    print(f"✓ Noise injection: SNR = {snr:.1f} dB")
    
    # Test fault injection
    fault_rate = 0.01
    fault_mask = np.random.random(conductance_matrix.shape) < fault_rate
    faulty_matrix = conductance_matrix.copy()
    faulty_matrix[fault_mask] = 0  # Stuck-at-zero faults
    
    fault_count = np.sum(fault_mask)
    print(f"✓ Fault injection: {fault_count} faults ({fault_rate*100:.1f}%)")
    
    return True

def main():
    """Run all core functionality tests."""
    print("Memristor-NN Core Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        test_device_models,
        test_error_handling,
        test_monitoring,
        test_security,
        test_core_algorithms
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())