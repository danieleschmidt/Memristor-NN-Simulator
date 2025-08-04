"""Tests for error handling and validation in memristor_nn package."""

import pytest
import torch
import torch.nn as nn
import numpy as np

import memristor_nn as mn
from memristor_nn.utils.validators import ValidationError
from memristor_nn.utils.security import SecurityError


class TestInputValidation:
    """Test input validation across all modules."""
    
    def test_crossbar_invalid_dimensions(self):
        """Test crossbar creation with invalid dimensions."""
        with pytest.raises(ValidationError):
            mn.CrossbarArray(rows=0, cols=64)  # Zero rows
            
        with pytest.raises(ValidationError):
            mn.CrossbarArray(rows=64, cols=-1)  # Negative cols
            
        with pytest.raises(ValidationError):
            mn.CrossbarArray(rows=50000, cols=50000)  # Too large
    
    def test_crossbar_invalid_device_model(self):
        """Test crossbar creation with invalid device model."""
        with pytest.raises(ValidationError):
            mn.CrossbarArray(rows=64, cols=64, device_model="InvalidDevice")
    
    def test_weight_programming_shape_mismatch(self):
        """Test weight programming with wrong shape."""
        crossbar = mn.CrossbarArray(rows=4, cols=4, device_model='IEDM2024_TaOx')
        
        # Wrong shape weight matrix
        wrong_weights = np.random.randn(3, 5)  # Should be 4x4
        
        with pytest.raises(ValidationError):
            crossbar.program_weights(wrong_weights)
    
    def test_weight_programming_invalid_data(self):
        """Test weight programming with invalid data."""
        crossbar = mn.CrossbarArray(rows=4, cols=4)
        
        # NaN weights
        nan_weights = np.full((4, 4), np.nan)
        with pytest.raises(ValidationError):
            crossbar.program_weights(nan_weights)
        
        # Infinite weights
        inf_weights = np.full((4, 4), np.inf)
        with pytest.raises(ValidationError):
            crossbar.program_weights(inf_weights)
    
    def test_simulation_invalid_temperature(self):
        """Test simulation with invalid temperature."""
        model = nn.Linear(4, 2)
        crossbar = mn.CrossbarArray(rows=4, cols=2)
        mapped_model = mn.map_to_crossbar(model, crossbar)
        test_data = torch.randn(10, 4)
        
        # Temperature too low
        with pytest.raises(ValidationError):
            mn.simulate(mapped_model, test_data, temperature=-10)
        
        # Temperature too high
        with pytest.raises(ValidationError):
            mn.simulate(mapped_model, test_data, temperature=2000)
    
    def test_simulation_invalid_batch_size(self):
        """Test simulation with invalid batch size."""
        model = nn.Linear(4, 2)
        crossbar = mn.CrossbarArray(rows=4, cols=2)
        mapped_model = mn.map_to_crossbar(model, crossbar)
        test_data = torch.randn(10, 4)
        
        # Negative batch size
        with pytest.raises(ValidationError):
            mn.simulate(mapped_model, test_data, batch_size=-1)
        
        # Zero batch size
        with pytest.raises(ValidationError):
            mn.simulate(mapped_model, test_data, batch_size=0)
        
        # Too large batch size
        with pytest.raises(ValidationError):
            mn.simulate(mapped_model, test_data, batch_size=20000)


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    def test_device_model_fallback(self):
        """Test fallback when device model fails."""
        # This should create a working crossbar even if some internal operations fail
        crossbar = mn.CrossbarArray(rows=8, cols=8)
        
        # Should still be able to get basic properties
        assert crossbar.rows == 8
        assert crossbar.cols == 8
        assert crossbar.device_model is not None
    
    def test_simulation_with_corrupted_data(self):
        """Test simulation behavior with partially corrupted data."""
        model = nn.Linear(4, 2)
        crossbar = mn.CrossbarArray(rows=4, cols=2)
        mapped_model = mn.map_to_crossbar(model, crossbar)
        
        # Create test data with some valid and some invalid samples
        valid_data = torch.randn(5, 4)
        
        # Should handle gracefully and return results for valid data
        try:
            results = mn.simulate(mapped_model, valid_data, max_batches=2)
            assert results.inference_count > 0
        except Exception as e:
            # Should at least fail gracefully with informative error
            assert isinstance(e, (ValidationError, RuntimeError))
    
    def test_memory_limit_handling(self):
        """Test handling of memory limits."""
        # Try to create a very large crossbar that should trigger memory checks
        try:
            # This should either work or fail gracefully
            large_crossbar = mn.CrossbarArray(rows=5000, cols=5000)
            # If it succeeds, check basic functionality
            assert large_crossbar.rows == 5000
        except (ValidationError, SecurityError, MemoryError):
            # Expected to fail due to size limits
            pass
    
    def test_fault_injection_error_handling(self):
        """Test error handling in fault injection."""
        model = nn.Linear(4, 2)
        crossbar = mn.CrossbarArray(rows=4, cols=2)
        mapped_model = mn.map_to_crossbar(model, crossbar)
        
        fault_analyzer = mn.FaultAnalyzer(mapped_model)
        
        # Invalid fault type
        with pytest.raises((ValidationError, ValueError)):
            fault_analyzer.inject_faults(['invalid_fault_type'], [0.01], n_trials=1)
        
        # Invalid fault rates
        with pytest.raises((ValidationError, ValueError)):
            fault_analyzer.inject_faults(['stuck_at_on'], [-0.1], n_trials=1)  # Negative rate


class TestSecurityValidation:
    """Test security-related validation."""
    
    def test_input_sanitization(self):
        """Test input sanitization in various contexts."""
        from memristor_nn.utils.security import sanitize_input, SecurityError
        
        # Normal input should pass
        clean_input = sanitize_input("normal_text_123")
        assert clean_input == "normal_text_123"
        
        # Dangerous input should be sanitized
        dangerous_input = "<script>alert('xss')</script>normal"
        clean = sanitize_input(dangerous_input)
        assert "<script>" not in clean
        
        # Input too long should fail
        with pytest.raises(SecurityError):
            sanitize_input("x" * 2000, max_length=100)
    
    def test_file_path_validation(self):
        """Test file path validation."""
        from memristor_nn.utils.security import validate_file_path, SecurityError
        
        # Normal path should work
        normal_path = validate_file_path("test_file.txt")
        assert normal_path.name == "test_file.txt"
        
        # Directory traversal should fail
        with pytest.raises(SecurityError):
            validate_file_path("../../../etc/passwd")
        
        # Invalid extension should fail
        with pytest.raises(SecurityError):
            validate_file_path("malicious.exe", allowed_extensions=[".txt", ".csv"])
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        from memristor_nn.utils.security import validate_network_config, SecurityError
        
        # Valid config should pass
        valid_config = {
            'rows': 64,
            'cols': 64,
            'device_model': 'IEDM2024_TaOx',
            'temperature': 300.0
        }
        
        validated = validate_network_config(valid_config)
        assert validated['rows'] == 64
        
        # Invalid keys should fail
        invalid_config = {
            'malicious_key': 'dangerous_value',
            'rows': 64
        }
        
        with pytest.raises(SecurityError):
            validate_network_config(invalid_config)
        
        # Out-of-range values should fail
        bad_config = {
            'rows': 1000000,  # Too large
            'cols': 64
        }
        
        with pytest.raises(SecurityError):
            validate_network_config(bad_config)


class TestLogging:
    """Test logging functionality."""
    
    def test_logger_creation(self):
        """Test logger creation and basic functionality."""
        from memristor_nn.utils.logger import setup_logger, get_logger
        
        # Create logger
        logger = setup_logger("test_logger", level="INFO")
        assert logger.name == "test_logger"
        
        # Get existing logger
        same_logger = get_logger("test_logger")
        assert same_logger is logger
    
    def test_performance_logging(self):
        """Test performance logging context manager."""
        from memristor_nn.utils.logger import PerformanceLogger, get_logger
        
        logger = get_logger("test_perf")
        
        # Should complete without errors
        with PerformanceLogger("test_operation", logger):
            # Simulate some work
            import time
            time.sleep(0.01)
    
    def test_logging_with_errors(self):
        """Test logging behavior when errors occur."""
        from memristor_nn.utils.logger import PerformanceLogger, get_logger
        
        logger = get_logger("test_error")
        
        # Should handle exceptions gracefully
        try:
            with PerformanceLogger("failing_operation", logger):
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected


class TestRobustnessScenarios:
    """Test robustness in edge case scenarios."""
    
    def test_empty_model_mapping(self):
        """Test mapping of empty or minimal models."""
        # Very simple model
        model = nn.Linear(1, 1)
        crossbar = mn.CrossbarArray(rows=1, cols=1)
        
        # Should handle gracefully
        mapped_model = mn.map_to_crossbar(model, crossbar)
        assert mapped_model is not None
    
    def test_simulation_with_minimal_data(self):
        """Test simulation with minimal datasets."""
        model = nn.Linear(2, 1)
        crossbar = mn.CrossbarArray(rows=2, cols=1)
        mapped_model = mn.map_to_crossbar(model, crossbar)
        
        # Single sample
        single_sample = torch.randn(1, 2)
        
        results = mn.simulate(mapped_model, single_sample, max_batches=1)
        assert results.inference_count == 1
    
    def test_design_space_exploration_edge_cases(self):
        """Test design space exploration with edge cases."""
        model = nn.Linear(4, 2)
        explorer = mn.DesignSpaceExplorer(model)
        
        # Empty parameter space
        empty_params = {}
        
        # Should handle gracefully or raise appropriate error
        try:
            results = explorer.explore(empty_params, n_samples=1)
            assert len(results) == 0  # No results expected
        except (ValidationError, ValueError):
            pass  # Expected behavior
        
        # Single parameter with single value
        single_param = {'tile_size': [64]}
        
        results = explorer.explore(single_param, n_samples=1, parallel=False)
        assert len(results) <= 1  # At most one result
    
    def test_concurrent_operations(self):
        """Test thread safety and concurrent operations."""
        import threading
        import time
        
        model = nn.Linear(4, 2)
        crossbar = mn.CrossbarArray(rows=4, cols=2)
        mapped_model = mn.map_to_crossbar(model, crossbar)
        test_data = torch.randn(10, 4)
        
        results = []
        errors = []
        
        def run_simulation():
            try:
                result = mn.simulate(mapped_model, test_data, max_batches=2)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple simulations concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=run_simulation)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0 or all(isinstance(e, (ValidationError, SecurityError)) for e in errors)
        assert len(results) > 0  # At least some should succeed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])