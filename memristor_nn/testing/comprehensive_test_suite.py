"""Comprehensive test suite for memristor neural network simulation."""

import unittest
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

# Import modules to test
from ..core.device_models import DeviceConfig, IEDM2024_TaOx, IEDM2024_HfOx, create_device
from ..core.crossbar import CrossbarArray
from ..mapping.neural_mapper import map_to_crossbar
from ..simulator.simulator import simulate, benchmark_model
from ..analysis.explorer import DesignSpaceExplorer
from ..faults.analyzer import FaultAnalyzer
from ..utils.validators import ValidationError
from ..utils.security import SecurityError
from ..utils.error_handling import get_error_collector


class TestDeviceModels(unittest.TestCase):
    """Test device model implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = DeviceConfig(
            read_noise_sigma=0.05,
            temperature=300.0
        )
    
    def test_taox_device_creation(self):
        """Test TaOx device model creation."""
        device = IEDM2024_TaOx(self.config)
        
        self.assertEqual(device.name, "IEDM2024_TaOx")
        self.assertEqual(device.ron, 1e4)
        self.assertEqual(device.roff, 1e6)
        self.assertIsNotNone(device.config)
    
    def test_hfox_device_creation(self):
        """Test HfOx device model creation."""
        device = IEDM2024_HfOx(self.config)
        
        self.assertEqual(device.name, "IEDM2024_HfOx")
        self.assertEqual(device.ron, 1e3)
        self.assertEqual(device.roff, 1e5)
    
    def test_device_registry(self):
        """Test device registry functionality."""
        device = create_device("IEDM2024_TaOx", self.config)
        self.assertIsInstance(device, IEDM2024_TaOx)
        
        with self.assertRaises(ValueError):
            create_device("NonexistentDevice")
    
    def test_conductance_calculation(self):
        """Test conductance calculations."""
        device = IEDM2024_TaOx(self.config)
        
        # Test various voltage and state combinations
        for voltage in [0.1, 0.5, 1.0]:
            for state in [0.0, 0.5, 1.0]:
                conductance = device.conductance(voltage, state)
                self.assertGreater(conductance, 0)
                self.assertIsFinite(conductance)
    
    def test_state_updates(self):
        """Test device state update dynamics."""
        device = IEDM2024_TaOx(self.config)
        
        initial_state = 0.5
        voltage = 1.0
        time_step = 1e-6
        
        new_state = device.update_state(voltage, time_step, initial_state)
        
        # State should be bounded [0, 1]
        self.assertGreaterEqual(new_state, 0.0)
        self.assertLessEqual(new_state, 1.0)
        
        # Positive voltage should increase state
        self.assertGreaterEqual(new_state, initial_state)
    
    def test_device_variations(self):
        """Test device-to-device variations."""
        device = IEDM2024_TaOx(self.config)
        
        base_conductance = 1e-4
        variations = []
        
        for _ in range(100):
            varied = device.add_variations(base_conductance)
            variations.append(varied)
        
        # Should have some spread
        std_dev = np.std(variations)
        self.assertGreater(std_dev, 0)
        
        # Should be reasonable range
        self.assertGreater(min(variations), 0)


class TestCrossbarArray(unittest.TestCase):
    """Test crossbar array implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rows = 16
        self.cols = 8
        self.crossbar = CrossbarArray(
            rows=self.rows,
            cols=self.cols,
            device_model="IEDM2024_TaOx"
        )
    
    def test_crossbar_initialization(self):
        """Test crossbar initialization."""
        self.assertEqual(self.crossbar.rows, self.rows)
        self.assertEqual(self.crossbar.cols, self.cols)
        self.assertEqual(self.crossbar.device_model.name, "IEDM2024_TaOx")
        
        # Device states should be initialized
        self.assertEqual(self.crossbar.device_states.shape, (self.rows, self.cols))
    
    def test_conductance_matrix(self):
        """Test conductance matrix computation."""
        conductances = self.crossbar.get_conductance_matrix(read_voltage=0.1)
        
        self.assertEqual(conductances.shape, (self.rows, self.cols))
        self.assertTrue(np.all(conductances > 0))
        self.assertTrue(np.all(np.isfinite(conductances)))
    
    def test_weight_programming_differential(self):
        """Test differential weight programming."""
        weights = np.random.randn(self.rows, self.cols)
        
        self.crossbar.program_weights(weights, programming_scheme="differential")
        
        # Device states should be in [0, 1] range
        self.assertTrue(np.all(self.crossbar.device_states >= 0))
        self.assertTrue(np.all(self.crossbar.device_states <= 1))
    
    def test_weight_programming_offset(self):
        """Test offset weight programming."""
        weights = np.random.randn(self.rows, self.cols)
        
        self.crossbar.program_weights(weights, programming_scheme="offset")
        
        # Device states should be in [0, 1] range
        self.assertTrue(np.all(self.crossbar.device_states >= 0))
        self.assertTrue(np.all(self.crossbar.device_states <= 1))
    
    def test_analog_matmul(self):
        """Test analog matrix multiplication."""
        input_vector = np.random.randn(self.rows)
        
        outputs = self.crossbar.analog_matmul(input_vector)
        
        self.assertEqual(len(outputs), self.cols)
        self.assertTrue(np.all(np.isfinite(outputs)))
    
    def test_power_consumption(self):
        """Test power consumption calculation."""
        power_breakdown = self.crossbar.get_power_consumption()
        
        required_keys = ["static_power_mw", "dynamic_power_mw", "peripheral_power_mw", "total_power_mw"]
        for key in required_keys:
            self.assertIn(key, power_breakdown)
            self.assertGreater(power_breakdown[key], 0)
    
    def test_area_estimation(self):
        """Test area estimation."""
        area_breakdown = self.crossbar.get_area_estimate()
        
        required_keys = ["device_area_mm2", "peripheral_area_mm2", "total_area_mm2"]
        for key in required_keys:
            self.assertIn(key, area_breakdown)
            self.assertGreater(area_breakdown[key], 0)
    
    def test_fault_injection(self):
        """Test fault injection."""
        initial_states = self.crossbar.device_states.copy()
        
        self.crossbar.inject_stuck_faults(fault_rate=0.1)
        
        # Some devices should be affected
        differences = np.sum(self.crossbar.device_states != initial_states)
        self.assertGreater(differences, 0)
    
    def test_drift_application(self):
        """Test temporal drift."""
        initial_states = self.crossbar.device_states.copy()
        
        self.crossbar.apply_drift(time_hours=100)
        
        # States should have changed but remain in bounds
        self.assertTrue(np.all(self.crossbar.device_states >= 0))
        self.assertTrue(np.all(self.crossbar.device_states <= 1))


class TestSimulation(unittest.TestCase):
    """Test simulation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test model
        self.model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        
        # Map to crossbar
        crossbar = CrossbarArray(32, 16, device_model="IEDM2024_TaOx")
        self.mapped_model = map_to_crossbar(self.model, crossbar)
        
        # Test data
        self.test_data = torch.randn(100, 32)
    
    def test_basic_simulation(self):
        """Test basic simulation functionality."""
        results = simulate(
            self.mapped_model,
            self.test_data,
            include_noise=True,
            batch_size=16,
            max_batches=5
        )
        
        # Check result attributes
        self.assertIsInstance(results.accuracy, float)
        self.assertIsInstance(results.energy_pj, float)
        self.assertIsInstance(results.latency_us, float)
        self.assertIsInstance(results.power_mw, float)
        
        # Reasonable value ranges
        self.assertGreaterEqual(results.accuracy, 0.0)
        self.assertLessEqual(results.accuracy, 1.0)
        self.assertGreater(results.energy_pj, 0)
        self.assertGreater(results.latency_us, 0)
        self.assertGreater(results.power_mw, 0)
    
    def test_simulation_without_noise(self):
        """Test simulation without device noise."""
        results = simulate(
            self.mapped_model,
            self.test_data,
            include_noise=False,
            batch_size=16,
            max_batches=3
        )
        
        self.assertIsNotNone(results)
        self.assertGreater(results.inference_count, 0)
    
    def test_temperature_effects(self):
        """Test temperature effects on simulation."""
        results_300k = simulate(
            self.mapped_model,
            self.test_data[:50],
            temperature=300.0,
            max_batches=2
        )
        
        results_350k = simulate(
            self.mapped_model,
            self.test_data[:50],
            temperature=350.0,
            max_batches=2
        )
        
        # Higher temperature should affect performance
        self.assertNotEqual(results_300k.power_mw, results_350k.power_mw)
    
    def test_benchmark_model(self):
        """Test model benchmarking."""
        benchmark_results = benchmark_model(
            self.mapped_model,
            input_shape=(32, 32),
            n_samples=50
        )
        
        required_keys = ["latency_us", "throughput_gops", "power_mw", "energy_pj", "area_mm2"]
        for key in required_keys:
            self.assertIn(key, benchmark_results)
            self.assertGreaterEqual(benchmark_results[key], 0)


class TestValidation(unittest.TestCase):
    """Test input validation and security."""
    
    def test_crossbar_parameter_validation(self):
        """Test crossbar parameter validation."""
        # Invalid dimensions
        with self.assertRaises(ValidationError):
            CrossbarArray(-1, 10)
        
        with self.assertRaises(ValidationError):
            CrossbarArray(10, 0)
        
        # Too large dimensions
        with self.assertRaises(ValidationError):
            CrossbarArray(100000, 100000)
        
        # Invalid device model
        with self.assertRaises(ValidationError):
            CrossbarArray(10, 10, device_model="InvalidDevice")
    
    def test_simulation_parameter_validation(self):
        """Test simulation parameter validation."""
        model = nn.Linear(4, 2)
        crossbar = CrossbarArray(4, 2)
        mapped_model = map_to_crossbar(model, crossbar)
        test_data = torch.randn(10, 4)
        
        # Invalid temperature
        with self.assertRaises(ValidationError):
            simulate(mapped_model, test_data, temperature=-100)
        
        with self.assertRaises(ValidationError):
            simulate(mapped_model, test_data, temperature=2000)
        
        # Invalid batch size
        with self.assertRaises(ValidationError):
            simulate(mapped_model, test_data, batch_size=0)
        
        with self.assertRaises(ValidationError):
            simulate(mapped_model, test_data, batch_size=100000)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""
    
    def test_error_collection(self):
        """Test error collection functionality."""
        error_collector = get_error_collector()
        initial_count = error_collector.get_error_summary()['total_errors']
        
        # Trigger an error
        try:
            CrossbarArray(-1, -1)  # Invalid parameters
        except ValidationError as e:
            error_collector.record_error(e, "test_context")
        
        # Check error was recorded
        new_count = error_collector.get_error_summary()['total_errors']
        self.assertGreater(new_count, initial_count)
    
    def test_graceful_degradation(self):
        """Test graceful degradation under adverse conditions."""
        # This would test the degraded simulation mode
        # when primary simulation fails
        pass


class TestPerformance(unittest.TestCase):
    """Performance and benchmarking tests."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.large_model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        crossbar = CrossbarArray(128, 64, device_model="IEDM2024_TaOx")
        self.mapped_large_model = map_to_crossbar(self.large_model, crossbar)
        self.large_test_data = torch.randn(1000, 128)
    
    def test_simulation_performance(self):
        """Test simulation performance benchmarks."""
        start_time = time.time()
        
        results = simulate(
            self.mapped_large_model,
            self.large_test_data,
            batch_size=64,
            max_batches=10
        )
        
        execution_time = time.time() - start_time
        
        # Performance requirements
        self.assertLess(execution_time, 30.0)  # Should complete in under 30s
        self.assertGreater(results.inference_count, 0)
        
        # Throughput requirements
        throughput = results.inference_count / execution_time
        self.assertGreater(throughput, 10)  # At least 10 inferences/second
    
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run simulation
        simulate(
            self.mapped_large_model,
            self.large_test_data,
            batch_size=32,
            max_batches=5
        )
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        self.assertLess(memory_increase, 500)
    
    def test_scalability(self):
        """Test performance scaling with model size."""
        # Small model
        small_model = nn.Linear(16, 8)
        small_crossbar = CrossbarArray(16, 8)
        small_mapped = map_to_crossbar(small_model, small_crossbar)
        small_data = torch.randn(100, 16)
        
        start_time = time.time()
        small_results = simulate(small_mapped, small_data, max_batches=5)
        small_time = time.time() - start_time
        
        # Large model (already set up)
        start_time = time.time()
        large_results = simulate(self.mapped_large_model, self.large_test_data[:100], max_batches=5)
        large_time = time.time() - start_time
        
        # Performance should scale reasonably
        size_ratio = (128 * 64) / (16 * 8)  # Device count ratio
        time_ratio = large_time / small_time
        
        # Time increase should be sub-quadratic with device count
        self.assertLess(time_ratio, size_ratio)


class TestSecurity(unittest.TestCase):
    """Security and safety tests."""
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        from ..utils.security import sanitize_input, SecurityError
        
        # Safe input
        safe_input = sanitize_input("normal_input_123")
        self.assertEqual(safe_input, "normal_input_123")
        
        # Dangerous input
        with self.assertRaises(SecurityError):
            sanitize_input("x" * 2000)  # Too long
    
    def test_file_path_validation(self):
        """Test file path validation."""
        from ..utils.security import validate_file_path, SecurityError
        
        # Safe path
        with tempfile.NamedTemporaryFile() as tmp:
            safe_path = validate_file_path(tmp.name, must_exist=True)
            self.assertIsInstance(safe_path, Path)
        
        # Directory traversal attempt
        with self.assertRaises(SecurityError):
            validate_file_path("../../../etc/passwd")
    
    def test_memory_limits(self):
        """Test memory limit enforcement."""
        from ..utils.security import check_memory_usage, SecurityError
        
        # Normal usage should pass
        check_memory_usage(max_mb=10000)
        
        # Very low limit should fail
        with self.assertRaises(SecurityError):
            check_memory_usage(max_mb=1)  # 1MB limit


def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive test suite and return results."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDeviceModels,
        TestCrossbarArray,
        TestSimulation,
        TestValidation,
        TestErrorHandling,
        TestPerformance,
        TestSecurity
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Compile results
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        "total_time": total_time,
        "passed": result.wasSuccessful()
    }


if __name__ == "__main__":
    # Run comprehensive tests
    test_results = run_comprehensive_tests()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*60)
    print(f"Tests Run: {test_results['tests_run']}")
    print(f"Failures: {test_results['failures']}")
    print(f"Errors: {test_results['errors']}")
    print(f"Success Rate: {test_results['success_rate']:.1%}")
    print(f"Total Time: {test_results['total_time']:.2f}s")
    print(f"Overall: {'PASS' if test_results['passed'] else 'FAIL'}")
    print("="*60)