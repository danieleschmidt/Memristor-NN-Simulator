"""Basic functionality tests for memristor_nn package."""

import pytest
import torch
import torch.nn as nn
import numpy as np

import memristor_nn as mn


class TestDeviceModels:
    """Test memristor device models."""
    
    def test_device_config_creation(self):
        """Test DeviceConfig creation with defaults."""
        config = mn.DeviceConfig()
        assert config.read_noise_sigma == 0.05
        assert config.ron_variation == 0.15
        assert config.temperature == 300.0
    
    def test_taox_device_creation(self):
        """Test TaOx device model creation."""
        device = mn.create_device('IEDM2024_TaOx')
        assert device.name == 'IEDM2024_TaOx'
        assert device.ron == 1e4
        assert device.roff == 1e6
    
    def test_hfox_device_creation(self):
        """Test HfOx device model creation."""
        device = mn.create_device('IEDM2024_HfOx')
        assert device.name == 'IEDM2024_HfOx'
        assert device.ron == 1e3
        assert device.roff == 1e5
    
    def test_device_conductance(self):
        """Test device conductance calculation."""
        device = mn.create_device('IEDM2024_TaOx')
        
        # Test conductance at different states
        g_low = device.conductance(0.1, 0.0)  # Low resistance state
        g_high = device.conductance(0.1, 1.0)  # High resistance state
        
        assert g_low > g_high  # Lower resistance -> higher conductance
        assert g_low > 0
        assert g_high > 0
    
    def test_device_state_update(self):
        """Test device state dynamics."""
        device = mn.create_device('IEDM2024_TaOx')
        
        initial_state = 0.5
        
        # Positive voltage should increase state
        new_state_pos = device.update_state(1.5, 1e-6, initial_state)
        assert new_state_pos > initial_state
        
        # Negative voltage should decrease state  
        new_state_neg = device.update_state(-1.5, 1e-6, initial_state)
        assert new_state_neg < initial_state


class TestCrossbarArray:
    """Test crossbar array functionality."""
    
    def test_crossbar_creation(self):
        """Test crossbar array creation."""
        crossbar = mn.CrossbarArray(
            rows=64,
            cols=64,
            device_model='IEDM2024_TaOx'
        )
        
        assert crossbar.rows == 64
        assert crossbar.cols == 64
        assert crossbar.device_states.shape == (64, 64)
        assert crossbar.device_model.name == 'IEDM2024_TaOx'
    
    def test_weight_programming(self):
        """Test weight matrix programming."""
        crossbar = mn.CrossbarArray(rows=4, cols=4, device_model='IEDM2024_TaOx')
        
        # Test weight matrix
        weights = np.array([
            [0.5, -0.3, 0.8, -0.1],
            [-0.2, 0.9, -0.6, 0.4],
            [0.7, -0.8, 0.2, -0.9],
            [-0.4, 0.1, -0.7, 0.6]
        ])
        
        # Program weights
        crossbar.program_weights(weights, programming_scheme='offset')
        
        # Check that device states are in valid range
        assert np.all(crossbar.device_states >= 0)
        assert np.all(crossbar.device_states <= 1)
    
    def test_analog_matmul(self):
        """Test analog matrix multiplication."""
        crossbar = mn.CrossbarArray(rows=4, cols=4, device_model='IEDM2024_TaOx')
        
        # Simple identity-like weights
        weights = np.eye(4)
        crossbar.program_weights(weights)
        
        # Test input
        input_vector = np.array([1.0, 0.5, -0.5, -1.0])
        
        # Perform analog computation
        output = crossbar.analog_matmul(input_vector)
        
        assert len(output) == 4
        assert np.all(np.isfinite(output))
    
    def test_power_calculation(self):
        """Test power consumption estimation."""
        crossbar = mn.CrossbarArray(rows=128, cols=128, device_model='IEDM2024_TaOx')
        
        power_stats = crossbar.get_power_consumption()
        
        assert 'static_power_mw' in power_stats
        assert 'dynamic_power_mw' in power_stats
        assert 'total_power_mw' in power_stats
        assert power_stats['total_power_mw'] > 0
    
    def test_area_calculation(self):
        """Test area estimation."""
        crossbar = mn.CrossbarArray(rows=128, cols=128, device_model='IEDM2024_TaOx')
        
        area_stats = crossbar.get_area_estimate()
        
        assert 'device_area_mm2' in area_stats
        assert 'total_area_mm2' in area_stats
        assert area_stats['total_area_mm2'] > 0
    
    def test_fault_injection(self):
        """Test stuck fault injection."""
        crossbar = mn.CrossbarArray(rows=10, cols=10, device_model='IEDM2024_TaOx')
        
        # Store original states
        original_states = crossbar.device_states.copy()
        
        # Inject faults
        crossbar.inject_stuck_faults(fault_rate=0.1)  # 10% fault rate
        
        # Check that some states changed
        changed_devices = np.sum(crossbar.device_states != original_states)
        assert changed_devices > 0  # Some devices should be faulty


class TestNeuralMapping:
    """Test neural network to crossbar mapping."""
    
    def create_simple_model(self):
        """Create simple test model."""
        return nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
    
    def test_mapping_efficiency_estimation(self):
        """Test mapping efficiency estimation."""
        model = self.create_simple_model()
        
        efficiency = mn.estimate_mapping_efficiency(model, crossbar_size=64)
        
        assert 'total_parameters' in efficiency
        assert 'mappable_parameters' in efficiency
        assert 'mapping_coverage' in efficiency
        assert efficiency['mapping_coverage'] <= 1.0
        assert efficiency['mapping_coverage'] >= 0.0
    
    def test_model_mapping(self):
        """Test complete model mapping."""
        model = self.create_simple_model()
        
        crossbar_template = mn.CrossbarArray(
            rows=8,
            cols=8,
            device_model='IEDM2024_TaOx'
        )
        
        mapped_model = mn.map_to_crossbar(model, crossbar_template)
        
        assert mapped_model.original_model == model
        assert len(mapped_model.mapped_layers) > 0
        
        # Check hardware stats
        hw_stats = mapped_model.get_hardware_stats()
        assert hw_stats['total_devices'] > 0
        assert hw_stats['total_power_mw'] > 0
    
    def test_mapped_model_forward(self):
        """Test forward pass through mapped model."""
        model = self.create_simple_model()
        
        crossbar_template = mn.CrossbarArray(
            rows=8,
            cols=8,
            device_model='IEDM2024_TaOx'
        )
        
        mapped_model = mn.map_to_crossbar(model, crossbar_template)
        
        # Test forward pass
        test_input = torch.randn(2, 4)  # Batch of 2 samples
        output = mapped_model.forward(test_input)
        
        assert output.shape[0] == 2  # Batch dimension preserved
        assert output.shape[1] == 2  # Output dimension correct


class TestSimulation:
    """Test simulation functionality."""
    
    def test_simulation_with_tensor(self):
        """Test simulation with tensor input."""
        model = nn.Linear(4, 2)
        crossbar = mn.CrossbarArray(rows=4, cols=2, device_model='IEDM2024_TaOx')
        mapped_model = mn.map_to_crossbar(model, crossbar)
        
        test_data = torch.randn(10, 4)
        
        results = mn.simulate(
            mapped_model,
            test_data,
            include_noise=False,
            max_batches=2
        )
        
        assert hasattr(results, 'accuracy')
        assert hasattr(results, 'power_mw')
        assert hasattr(results, 'latency_us')
        assert results.inference_count > 0
    
    def test_benchmark_model(self):
        """Test model benchmarking."""
        model = nn.Linear(8, 4)
        crossbar = mn.CrossbarArray(rows=8, cols=4, device_model='IEDM2024_TaOx')
        mapped_model = mn.map_to_crossbar(model, crossbar)
        
        benchmark_results = mn.benchmark_model(
            mapped_model,
            input_shape=(1, 8),
            n_samples=50
        )
        
        assert 'latency_us' in benchmark_results
        assert 'throughput_gops' in benchmark_results
        assert 'power_mw' in benchmark_results
        assert benchmark_results['power_mw'] > 0


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        
        # Create crossbar
        crossbar = mn.CrossbarArray(
            rows=16,
            cols=16,
            device_model='IEDM2024_TaOx'
        )
        
        # Map to hardware
        mapped_model = mn.map_to_crossbar(model, crossbar)
        
        # Generate test data
        test_data = torch.randn(100, 10)
        
        # Run simulation
        results = mn.simulate(
            mapped_model,
            test_data,
            include_noise=True,
            temperature=300,
            max_batches=5
        )
        
        # Verify results
        assert results.inference_count > 0
        assert results.power_mw > 0
        assert results.latency_us > 0
        assert 0 <= results.accuracy <= 1.0
    
    def test_multiple_device_comparison(self):
        """Test comparison across multiple device types."""
        model = nn.Linear(6, 4)
        test_data = torch.randn(50, 6)
        
        device_types = ['IEDM2024_TaOx', 'IEDM2024_HfOx']
        results = {}
        
        for device_type in device_types:
            crossbar = mn.CrossbarArray(
                rows=6,
                cols=4,
                device_model=device_type
            )
            
            mapped_model = mn.map_to_crossbar(model, crossbar)
            sim_results = mn.simulate(mapped_model, test_data, max_batches=3)
            
            results[device_type] = sim_results
        
        # Both simulations should complete
        assert len(results) == 2
        for device_type, result in results.items():
            assert result.power_mw > 0
            assert result.inference_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])