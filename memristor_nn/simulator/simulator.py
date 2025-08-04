"""Main simulation engine for memristive neural networks."""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..mapping.neural_mapper import MappedModel
from ..utils.logger import get_logger, PerformanceLogger
from ..utils.validators import (
    validate_tensor, validate_temperature, validate_batch_size, 
    validate_positive_number, ValidationError
)
from ..utils.security import check_memory_usage, rate_limit_check


@dataclass
class SimulationResults:
    """Results from neural network simulation."""
    
    accuracy: float
    energy_pj: float
    latency_us: float
    throughput_gops: float
    power_mw: float
    area_mm2: float
    device_count: int
    inference_count: int
    total_time_s: float
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.total_time_s > 0:
            self.inferences_per_second = self.inference_count / self.total_time_s
        else:
            self.inferences_per_second = 0.0


def simulate(
    mapped_model: MappedModel,
    test_data: Union[DataLoader, torch.Tensor],
    include_noise: bool = True,
    temperature: float = 300.0,
    batch_size: int = 32,
    max_batches: Optional[int] = None
) -> SimulationResults:
    """
    Run cycle-accurate simulation of mapped neural network.
    
    Args:
        mapped_model: Neural network mapped to crossbar arrays
        test_data: Test dataset or tensor
        include_noise: Whether to include device variations
        temperature: Operating temperature in Kelvin
        batch_size: Simulation batch size
        max_batches: Maximum batches to simulate (for quick testing)
        
    Returns:
        Comprehensive simulation results
        
    Raises:
        ValidationError: If inputs are invalid
        SecurityError: If operation exceeds safety limits
    """
    logger = get_logger("simulator")
    
    try:
        # Validate inputs
        temperature = validate_temperature(temperature)
        batch_size = validate_batch_size(batch_size)
        
        if max_batches is not None:
            max_batches = validate_positive_number(max_batches, "max_batches", max_value=10000)
        
        # Rate limiting
        rate_limit_check("simulation", max_calls=10, window_seconds=60)
        
        # Memory check
        check_memory_usage(max_mb=2000)
        
        logger.info(f"Starting simulation: noise={include_noise}, temp={temperature}K, batch_size={batch_size}")
        
        start_time = time.time()
        
        # Setup simulation parameters
        if not include_noise:
            _disable_noise_in_model(mapped_model)
        
        _set_temperature_in_model(mapped_model, temperature)
        
        # Prepare data
        if isinstance(test_data, torch.Tensor):
            test_loader = _tensor_to_dataloader(test_data, batch_size)
        else:
            test_loader = test_data
        
        # Run inference simulation
        accuracy, inference_stats = _run_inference_simulation(
            mapped_model, test_loader, max_batches
        )
        
        # Get hardware statistics
        hw_stats = mapped_model.get_hardware_stats()
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        avg_latency_us = inference_stats["avg_latency_us"]
        total_energy_pj = inference_stats["total_energy_pj"] 
        inference_count = inference_stats["inference_count"]
        
        # Calculate throughput (GOPS)
        ops_per_inference = _estimate_operations(mapped_model)
        total_ops = ops_per_inference * inference_count
        throughput_gops = (total_ops / total_time) / 1e9 if total_time > 0 else 0.0
        
        return SimulationResults(
            accuracy=accuracy,
            energy_pj=total_energy_pj / inference_count if inference_count > 0 else 0.0,
            latency_us=avg_latency_us,
            throughput_gops=throughput_gops,
            power_mw=hw_stats["total_power_mw"],
            area_mm2=hw_stats["total_area_mm2"],
            device_count=hw_stats["total_devices"],
            inference_count=inference_count,
            total_time_s=total_time
        )
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


def _disable_noise_in_model(mapped_model: MappedModel) -> None:
    """Disable noise and variations in all crossbars."""
    for layer in mapped_model.mapped_layers:
        for crossbar in layer.crossbars:
            crossbar.config.read_noise_sigma = 0.0
            crossbar.config.ron_variation = 0.0
            crossbar.config.roff_variation = 0.0
            crossbar.config.temp_coefficient = 0.0


def _set_temperature_in_model(mapped_model: MappedModel, temperature: float) -> None:
    """Set operating temperature for all crossbars."""
    for layer in mapped_model.mapped_layers:
        for crossbar in layer.crossbars:
            crossbar.config.temperature = temperature


def _tensor_to_dataloader(tensor: torch.Tensor, batch_size: int) -> DataLoader:
    """Convert tensor to DataLoader for simulation."""
    if tensor.dim() == 2:  # (samples, features)
        dataset = torch.utils.data.TensorDataset(tensor, torch.zeros(tensor.size(0)))
    else:
        raise ValueError("Tensor must be 2D (samples, features)")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _run_inference_simulation(
    mapped_model: MappedModel,
    test_loader: DataLoader,
    max_batches: Optional[int]
) -> tuple:
    """Run inference simulation and collect statistics."""
    correct_predictions = 0
    total_samples = 0
    total_latency = 0.0
    total_energy = 0.0
    inference_count = 0
    
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Simulating")):
            if max_batches and batch_count >= max_batches:
                break
                
            batch_start = time.time()
            
            # Forward pass through mapped hardware
            outputs = mapped_model.forward(data)
            
            # Calculate accuracy (assuming classification)
            if targets.numel() > 0:  # Has labels
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == targets).sum().item()
            
            # Estimate energy consumption
            batch_energy = _estimate_batch_energy(mapped_model, data.size(0))
            total_energy += batch_energy
            
            # Record timing
            batch_time = time.time() - batch_start
            total_latency += batch_time
            
            total_samples += data.size(0)
            inference_count += data.size(0)
            batch_count += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # Calculate average latency per inference
    avg_latency_us = (total_latency / inference_count) * 1e6 if inference_count > 0 else 0.0
    
    inference_stats = {
        "avg_latency_us": avg_latency_us,
        "total_energy_pj": total_energy,
        "inference_count": inference_count
    }
    
    return accuracy, inference_stats


def _estimate_batch_energy(mapped_model: MappedModel, batch_size: int) -> float:
    """Estimate energy consumption for a batch."""
    hw_stats = mapped_model.get_hardware_stats()
    
    # Static energy (leakage during computation)
    static_power_mw = hw_stats["total_power_mw"] * 0.6  # 60% static
    computation_time_us = 10  # Estimate 10us per inference
    static_energy_pj = static_power_mw * computation_time_us * batch_size
    
    # Dynamic energy (switching energy)
    dynamic_energy_per_inference = hw_stats["total_devices"] * 1e-3  # pJ per device
    dynamic_energy_pj = dynamic_energy_per_inference * batch_size
    
    return static_energy_pj + dynamic_energy_pj


def _estimate_operations(mapped_model: MappedModel) -> int:
    """Estimate number of operations per inference."""
    total_ops = 0
    
    for layer in mapped_model.mapped_layers:
        if layer.layer_type == "Linear":
            # Each crossbar performs matrix multiplication
            for crossbar in layer.crossbars:
                ops = crossbar.rows * crossbar.cols  # MAC operations
                total_ops += ops
    
    return total_ops


def benchmark_model(
    mapped_model: MappedModel,
    input_shape: tuple,
    n_samples: int = 1000
) -> Dict[str, float]:
    """
    Benchmark model performance with synthetic data.
    
    Args:
        mapped_model: Mapped neural network
        input_shape: Shape of input tensor (batch_size, features)
        n_samples: Number of samples to generate
        
    Returns:
        Performance benchmark results
    """
    # Generate synthetic test data
    test_data = torch.randn(n_samples, input_shape[1])
    test_labels = torch.randint(0, 10, (n_samples,))  # Assume 10 classes
    
    # Run simulation
    results = simulate(mapped_model, test_data, include_noise=True)
    
    return {
        "latency_us": results.latency_us,
        "throughput_gops": results.throughput_gops,
        "power_mw": results.power_mw,
        "energy_pj": results.energy_pj,
        "area_mm2": results.area_mm2,
        "efficiency_gops_per_w": results.throughput_gops / (results.power_mw / 1000) if results.power_mw > 0 else 0.0
    }