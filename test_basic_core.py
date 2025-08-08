#!/usr/bin/env python3
"""Basic test of core functionality without external dependencies."""

import sys
sys.path.insert(0, '/root/repo')

# Test basic imports and device models
def test_basic_functionality():
    """Test core device models without torch dependencies."""
    print("=== Testing Core Device Models ===")
    
    try:
        from memristor_nn.core.device_models import DeviceConfig, IEDM2024_TaOx, IEDM2024_HfOx
        print("✓ Device models imported successfully")
        
        # Test device creation
        config = DeviceConfig(read_noise_sigma=0.05, temperature=300)
        taox_device = IEDM2024_TaOx(config)
        hfox_device = IEDM2024_HfOx(config)
        
        print(f"✓ Created TaOx device: {taox_device.name}")
        print(f"✓ Created HfOx device: {hfox_device.name}")
        
        # Test conductance calculation
        voltage = 0.5  # 0.5V
        state = 0.7    # 70% programmed state
        
        taox_conductance = taox_device.conductance(voltage, state)
        hfox_conductance = hfox_device.conductance(voltage, state)
        
        print(f"✓ TaOx conductance at {voltage}V, state {state}: {taox_conductance:.2e} S")
        print(f"✓ HfOx conductance at {voltage}V, state {state}: {hfox_conductance:.2e} S")
        
        # Test state update
        time_step = 1e-6  # 1 microsecond
        new_taox_state = taox_device.update_state(voltage, time_step, state)
        new_hfox_state = hfox_device.update_state(voltage, time_step, state)
        
        print(f"✓ TaOx state after {time_step*1e6:.1f}µs: {new_taox_state:.3f}")
        print(f"✓ HfOx state after {time_step*1e6:.1f}µs: {new_hfox_state:.3f}")
        
        # Test resistance with variations
        resistance = taox_device.get_resistance(voltage, state)
        print(f"✓ TaOx resistance with variations: {resistance:.0f} Ω")
        
        print("\n=== Core Device Models Test: PASSED ===")
        return True
        
    except Exception as e:
        print(f"✗ Core device models test failed: {e}")
        return False

def test_logging_and_utils():
    """Test utility functions."""
    print("\n=== Testing Utility Functions ===")
    
    try:
        from memristor_nn.utils.logger import LoggingMixin
        from memristor_nn.utils.validators import validate_positive_number, ValidationError
        from memristor_nn.utils.security import check_memory_usage, SecurityError
        
        print("✓ Utility imports successful")
        
        # Test validation
        try:
            valid_num = validate_positive_number(5.0, "test_param")
            print(f"✓ Validation passed for positive number: {valid_num}")
        except ValidationError:
            print("✗ Validation test failed")
            return False
            
        # Test invalid validation
        try:
            validate_positive_number(-1.0, "test_param")
            print("✗ Validation should have failed for negative number")
            return False
        except ValidationError:
            print("✓ Validation correctly rejected negative number")
        
        # Test logging mixin
        class TestClass(LoggingMixin):
            def test_method(self):
                self.logger.info("Test logging message")
                
        test_obj = TestClass()
        test_obj.test_method()
        print("✓ Logging mixin working")
        
        print("✓ Utility functions test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False

def test_optimization_modules():
    """Test optimization modules without torch."""
    print("\n=== Testing Optimization Modules ===")
    
    try:
        from memristor_nn.optimization.cache_manager import CacheManager, CacheEntry
        from memristor_nn.optimization.memory_optimizer import MemoryOptimizer
        from memristor_nn.optimization.performance_profiler import PerformanceProfiler
        
        print("✓ Optimization module imports successful")
        
        # Test cache manager
        cache = CacheManager(max_size=100, ttl_seconds=3600)
        
        # Test cache operations
        cache.set("test_key", "test_value")
        cached_value = cache.get("test_key")
        
        if cached_value == "test_value":
            print("✓ Cache manager working correctly")
        else:
            print("✗ Cache manager failed")
            return False
        
        # Test memory optimizer
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_stats()
        print(f"✓ Memory optimizer: {stats.get('total_mb', 'unknown')} MB tracked")
        
        # Test performance profiler
        profiler = PerformanceProfiler()
        profiler.start_timing("test_operation")
        # Simulate some work
        import time
        time.sleep(0.001)
        elapsed = profiler.end_timing("test_operation")
        print(f"✓ Performance profiler: {elapsed:.3f}s measured")
        
        print("✓ Optimization modules test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Optimization modules test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("Memristor-NN-Simulator Basic Functionality Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test core functionality
    all_passed &= test_basic_functionality()
    all_passed &= test_logging_and_utils()
    all_passed &= test_optimization_modules()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED")
        print("Core functionality is working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Core functionality needs attention.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)