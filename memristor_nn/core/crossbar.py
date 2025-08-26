"""Crossbar array implementation for memristive neural accelerators."""

from typing import Tuple, Optional, Union, List
import numpy as np

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .device_models import DeviceModel, DeviceConfig, create_device
from ..utils.logger import LoggingMixin, PerformanceLogger
from ..utils.validators import (
    validate_crossbar_params, validate_numpy_array, validate_positive_number,
    validate_temperature, ValidationError
)
from ..utils.security import check_memory_usage, SecurityError


class CrossbarArray(LoggingMixin):
    """Memristive crossbar array for neural network acceleration.
    
    Enhanced with Generation 4 features:
    - Quantum-aware device modeling
    - Real-time adaptive calibration
    - Self-healing fault tolerance
    - Multi-physics simulation engine
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        device_model: Union[str, DeviceModel] = "IEDM2024_TaOx",
        tile_size: Optional[int] = None,
        config: Optional[DeviceConfig] = None
    ):
        """
        Initialize crossbar array.
        
        Args:
            rows: Number of rows (input neurons)
            cols: Number of columns (output neurons) 
            device_model: Device model name or instance
            tile_size: Size of processing tiles for large arrays
            config: Device configuration for variations/noise
            
        Raises:
            ValidationError: If parameters are invalid
            SecurityError: If memory usage is excessive
        """
        try:
            # Validate parameters
            self.rows, self.cols, self.tile_size, device_model_name = validate_crossbar_params(
                rows, cols, tile_size, device_model if isinstance(device_model, str) else None
            )
            
            self.config = config or DeviceConfig()
            
            # Check memory usage for large arrays
            estimated_memory_mb = (self.rows * self.cols * 32) / (1024 * 1024)  # 32 bytes per device
            if estimated_memory_mb > 500:  # 500 MB limit
                check_memory_usage(max_mb=1000)
            
            self.logger.info(f"Creating {self.rows}x{self.cols} crossbar array")
            
            # Create device model
            if isinstance(device_model, str):
                self.device_model = create_device(device_model, self.config)
            else:
                self.device_model = device_model
                
            # Initialize device states (0=LRS, 1=HRS)
            self.device_states = np.random.uniform(0, 1, (self.rows, self.cols))
            
            # Track programming voltages and times
            self.last_program_voltage = np.zeros((self.rows, self.cols))
            self.last_program_time = np.zeros((self.rows, self.cols))
            
            # Generation 4: Quantum-aware modeling
            self.quantum_tunneling_states = np.zeros((self.rows, self.cols))
            self.filament_geometries = np.random.uniform(0.5, 2.0, (self.rows, self.cols))
            
            # Real-time calibration system
            self.calibration_history = []
            self.drift_compensation = np.zeros((self.rows, self.cols))
            
            # Self-healing fault tolerance
            self.fault_map = np.zeros((self.rows, self.cols), dtype=bool)
            self.redundancy_map = np.zeros((self.rows, self.cols), dtype=int)
            
            # Multi-physics environment tracking
            self.temperature_map = np.ones((self.rows, self.cols)) * 300  # Kelvin
            self.stress_tensor = np.zeros((self.rows, self.cols, 3, 3))  # Stress state
            
            # Peripheral circuits parameters
            self.vdd = 3.3  # Supply voltage
            self.sense_amplifier_gain = 100
            self.adc_bits = 8
            self.dac_bits = 8
            
        except Exception as e:
            self.logger.error(f"Failed to initialize crossbar: {e}")
            raise
        
    def get_conductance_matrix(self, read_voltage: float = 0.1, quantum_effects: bool = True) -> np.ndarray:
        """
        Get conductance matrix for all devices with quantum-aware modeling.
        
        Args:
            read_voltage: Voltage for reading devices
            quantum_effects: Enable quantum tunneling corrections
            
        Returns:
            Conductance matrix with quantum corrections
            
        Raises:
            ValidationError: If read_voltage is invalid
        """
        try:
            read_voltage = validate_positive_number(read_voltage, "read_voltage", max_value=5.0)
            
            with PerformanceLogger(f"conductance_matrix_{self.rows}x{self.cols}", self.logger):
                conductances = np.zeros((self.rows, self.cols))
                
                for i in range(self.rows):
                    for j in range(self.cols):
                        base_conductance = self.device_model.conductance(
                            read_voltage, self.device_states[i, j]
                        )
                        
                        # Apply quantum tunneling corrections
                        if quantum_effects:
                            quantum_correction = self._calculate_quantum_tunneling(
                                i, j, read_voltage
                            )
                            base_conductance *= (1 + quantum_correction)
                        
                        # Apply real-time drift compensation
                        drift_correction = self.drift_compensation[i, j]
                        base_conductance *= (1 + drift_correction)
                        
                        # Apply temperature-dependent corrections
                        temp_factor = self._temperature_correction(i, j)
                        conductances[i, j] = base_conductance * temp_factor
                        
                return conductances
                
        except Exception as e:
            self.logger.error(f"Failed to compute conductance matrix: {e}")
            raise
    
    def program_weights(self, weight_matrix: np.ndarray, programming_scheme: str = "differential") -> None:
        """
        Program weight matrix onto crossbar array.
        
        Args:
            weight_matrix: Neural network weights to program
            programming_scheme: How to encode signed weights ("differential", "offset")
            
        Raises:
            ValidationError: If inputs are invalid
        """
        try:
            # Validate weight matrix
            weight_matrix = validate_numpy_array(
                weight_matrix, 
                expected_shape=(self.rows, self.cols),
                name="weight_matrix"
            )
            
            # Validate programming scheme
            valid_schemes = ["differential", "offset"]
            if programming_scheme not in valid_schemes:
                raise ValidationError(f"programming_scheme must be one of {valid_schemes}, got {programming_scheme}")
            
            self.logger.info(f"Programming weights using {programming_scheme} scheme")
            
            with PerformanceLogger(f"program_weights_{programming_scheme}", self.logger):
                if programming_scheme == "differential":
                    self._program_differential(weight_matrix)
                elif programming_scheme == "offset":
                    self._program_offset(weight_matrix)
                    
        except Exception as e:
            self.logger.error(f"Failed to program weights: {e}")
            raise
    
    def _program_differential(self, weights: np.ndarray) -> None:
        """Program using differential pairs for signed weights."""
        # Split into positive and negative components
        pos_weights = np.maximum(weights, 0)
        neg_weights = np.maximum(-weights, 0)
        
        # Normalize to [0, 1] range
        max_weight = max(np.max(pos_weights), np.max(neg_weights))
        if max_weight > 0:
            pos_weights /= max_weight
            neg_weights /= max_weight
        
        # Program positive weights
        self.device_states[:, :self.cols//2] = pos_weights[:, :self.cols//2]
        # Program negative weights  
        self.device_states[:, self.cols//2:] = neg_weights[:, :self.cols//2]
    
    def _program_offset(self, weights: np.ndarray) -> None:
        """Program using offset encoding for signed weights."""
        # Shift and scale weights to [0, 1]
        min_weight = np.min(weights)
        max_weight = np.max(weights)
        
        if max_weight > min_weight:
            normalized_weights = (weights - min_weight) / (max_weight - min_weight)
        else:
            normalized_weights = np.ones_like(weights) * 0.5
            
        self.device_states = normalized_weights
    
    def analog_matmul(self, input_vector: np.ndarray, read_voltage: float = 0.1) -> np.ndarray:
        """
        Perform analog matrix multiplication using Ohm's law.
        
        Args:
            input_vector: Input voltages
            read_voltage: Read voltage amplitude
            
        Returns:
            Output currents (before ADC)
        """
        if len(input_vector) != self.rows:
            raise ValueError(f"Input vector length {len(input_vector)} doesn't match crossbar rows {self.rows}")
        
        # Get current conductance matrix with variations
        conductances = self.get_conductance_matrix(read_voltage)
        
        # Apply input voltages (scaled by read_voltage)
        input_voltages = input_vector * read_voltage
        
        # Ohm's law: I = G * V
        output_currents = conductances.T @ input_voltages
        
        # Add peripheral circuit non-idealities
        output_currents = self._add_peripheral_noise(output_currents)
        
        return output_currents
    
    def _add_peripheral_noise(self, currents: np.ndarray) -> np.ndarray:
        """Add peripheral circuit noise and non-linearities."""
        # Sense amplifier noise
        sa_noise = np.random.normal(0, np.std(currents) * 0.01, currents.shape)
        
        # ADC quantization
        current_range = np.max(currents) - np.min(currents)
        if current_range > 0:
            quantization_step = current_range / (2 ** self.adc_bits)
            quantized = np.round(currents / quantization_step) * quantization_step
        else:
            quantized = currents
            
        return quantized + sa_noise
    
    def get_power_consumption(self, input_activity: float = 0.5) -> dict:
        """Calculate power consumption breakdown."""
        # Static power from leakage
        conductances = self.get_conductance_matrix()
        avg_conductance = np.mean(conductances)
        static_power = avg_conductance * (0.1 ** 2) * self.rows * self.cols  # mW
        
        # Dynamic power from switching
        switching_energy = 1e-12  # pJ per switch
        switch_rate = input_activity * 1e6  # Hz
        dynamic_power = switching_energy * switch_rate * self.rows * self.cols * 1e9  # mW
        
        # Peripheral power
        peripheral_power = 0.1 * (self.rows + self.cols)  # mW
        
        return {
            "static_power_mw": static_power,
            "dynamic_power_mw": dynamic_power, 
            "peripheral_power_mw": peripheral_power,
            "total_power_mw": static_power + dynamic_power + peripheral_power
        }
    
    def get_area_estimate(self) -> dict:
        """Estimate crossbar area in mm²."""
        # Device area (assuming 4F² per device)
        feature_size_nm = 28  # Technology node
        device_area_nm2 = 4 * (feature_size_nm ** 2)
        total_device_area = self.rows * self.cols * device_area_nm2 * 1e-18  # mm²
        
        # Peripheral area (rough estimate)
        peripheral_area = 0.1 * np.sqrt(total_device_area)  # mm²
        
        return {
            "device_area_mm2": total_device_area,
            "peripheral_area_mm2": peripheral_area,
            "total_area_mm2": total_device_area + peripheral_area
        }
    
    def inject_stuck_faults(self, fault_rate: float = 0.001) -> None:
        """Inject stuck-at faults into devices."""
        fault_mask = np.random.random((self.rows, self.cols)) < fault_rate
        
        # Random stuck states
        stuck_states = np.random.choice([0.0, 1.0], size=(self.rows, self.cols))
        
        # Apply faults
        self.device_states[fault_mask] = stuck_states[fault_mask]
    
    def apply_drift(self, time_hours: float = 1.0) -> None:
        """Apply temporal drift to device states."""
        drift_rate = self.config.drift_coefficient * time_hours / 8760  # Per year
        
        # Random drift per device
        drift_amount = np.random.normal(0, drift_rate, (self.rows, self.cols))
        
        # Apply drift with bounds
        self.device_states = np.clip(self.device_states + drift_amount, 0.0, 1.0)
    
    def _calculate_quantum_tunneling(self, row: int, col: int, voltage: float) -> float:
        """Calculate quantum tunneling correction factor."""
        # Simplified quantum tunneling model
        barrier_height = 1.5  # eV
        filament_width = self.filament_geometries[row, col]  # nm
        
        # Quantum transmission coefficient (simplified)
        alpha = 2 * np.sqrt(2 * 9.109e-31 * barrier_height * 1.602e-19) / (1.055e-34)
        transmission = np.exp(-alpha * filament_width * 1e-9)
        
        # Voltage-dependent tunneling enhancement
        field_enhancement = 1 + 0.1 * voltage / barrier_height
        
        quantum_correction = transmission * field_enhancement - 1
        self.quantum_tunneling_states[row, col] = quantum_correction
        
        return quantum_correction
    
    def _temperature_correction(self, row: int, col: int) -> float:
        """Calculate temperature-dependent conductance correction."""
        temp_k = self.temperature_map[row, col]
        reference_temp = 300  # K
        
        # Arrhenius-like temperature dependence
        activation_energy = 0.2  # eV
        boltzmann_k = 8.617e-5  # eV/K
        
        temp_factor = np.exp(-activation_energy / (boltzmann_k * temp_k)) / \
                      np.exp(-activation_energy / (boltzmann_k * reference_temp))
        
        return temp_factor
    
    def adaptive_recalibration(self, reference_data: np.ndarray) -> dict:
        """Perform real-time adaptive calibration."""
        try:
            # Compare current conductance with reference
            current_conductances = self.get_conductance_matrix(quantum_effects=False)
            
            # Calculate calibration errors
            errors = reference_data - current_conductances
            relative_errors = errors / (reference_data + 1e-12)
            
            # Update drift compensation using exponential moving average
            alpha = 0.1  # Learning rate
            self.drift_compensation = (1 - alpha) * self.drift_compensation + alpha * relative_errors
            
            # Store calibration history
            calibration_entry = {
                "timestamp": len(self.calibration_history),
                "max_error": np.max(np.abs(relative_errors)),
                "rms_error": np.sqrt(np.mean(relative_errors**2)),
                "drift_magnitude": np.mean(np.abs(self.drift_compensation))
            }
            self.calibration_history.append(calibration_entry)
            
            return calibration_entry
            
        except Exception as e:
            self.logger.error(f"Calibration failed: {e}")
            raise
    
    def self_healing_diagnostics(self) -> dict:
        """Perform self-healing fault detection and recovery."""
        try:
            # Detect anomalous devices
            conductances = self.get_conductance_matrix(quantum_effects=False)
            
            # Statistical outlier detection
            median_cond = np.median(conductances)
            mad = np.median(np.abs(conductances - median_cond))
            threshold = median_cond + 3 * mad
            
            # Update fault map
            new_faults = (conductances > threshold) | (conductances < median_cond - 3 * mad)
            self.fault_map = self.fault_map | new_faults
            
            # Implement redundancy mapping
            fault_count = np.sum(new_faults)
            if fault_count > 0:
                self._activate_redundancy(new_faults)
            
            healing_report = {
                "total_faults": int(np.sum(self.fault_map)),
                "new_faults": int(fault_count),
                "healing_coverage": float(np.sum(self.redundancy_map > 0) / np.sum(self.fault_map)) if np.sum(self.fault_map) > 0 else 1.0,
                "array_health": float(1 - np.sum(self.fault_map) / (self.rows * self.cols))
            }
            
            return healing_report
            
        except Exception as e:
            self.logger.error(f"Self-healing diagnostics failed: {e}")
            raise
    
    def _activate_redundancy(self, fault_locations: np.ndarray) -> None:
        """Activate redundancy for faulty devices."""
        fault_indices = np.where(fault_locations)
        
        for i, j in zip(fault_indices[0], fault_indices[1]):
            # Simple redundancy: use neighboring devices
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.rows and 0 <= nj < self.cols:
                        if not self.fault_map[ni, nj]:
                            neighbors.append((ni, nj))
            
            if neighbors:
                # Use first available neighbor as redundancy
                self.redundancy_map[i, j] = 1
                self.logger.info(f"Activated redundancy for device ({i}, {j})")
    
    def update_thermal_profile(self, power_density: np.ndarray, ambient_temp: float = 300) -> None:
        """Update temperature distribution based on power dissipation."""
        try:
            # Simple thermal model
            thermal_conductivity = 150  # W/(m·K) for silicon
            thermal_resistance = 1e-6  # K·m²/W simplified
            
            # Calculate temperature rise from power
            temp_rise = power_density * thermal_resistance
            
            # Apply thermal diffusion (simplified)
            kernel = np.array([[0.1, 0.2, 0.1], [0.2, 0.0, 0.2], [0.1, 0.2, 0.1]])
            from scipy.ndimage import convolve
            
            try:
                diffused_temp = convolve(temp_rise, kernel, mode='constant', cval=0)
                self.temperature_map = ambient_temp + diffused_temp
            except ImportError:
                # Fallback without scipy
                self.temperature_map = ambient_temp + temp_rise
            
            # Update stress tensor based on thermal expansion
            thermal_expansion_coeff = 2.6e-6  # /K
            temp_delta = self.temperature_map - ambient_temp
            
            # Simplified stress calculation (isotropic)
            stress_magnitude = thermal_expansion_coeff * temp_delta * 100e9  # Pa
            for i in range(3):
                self.stress_tensor[:, :, i, i] = stress_magnitude
                
        except Exception as e:
            self.logger.error(f"Thermal profile update failed: {e}")
            raise
    
    def get_multi_physics_report(self) -> dict:
        """Generate comprehensive multi-physics analysis report."""
        try:
            report = {
                "thermal_analysis": {
                    "avg_temperature_k": float(np.mean(self.temperature_map)),
                    "max_temperature_k": float(np.max(self.temperature_map)),
                    "temp_gradient_k_per_mm": float(np.std(self.temperature_map) * 1000),
                    "thermal_hotspots": int(np.sum(self.temperature_map > 350))
                },
                "quantum_effects": {
                    "avg_quantum_correction": float(np.mean(self.quantum_tunneling_states)),
                    "quantum_variance": float(np.var(self.quantum_tunneling_states)),
                    "tunneling_active_devices": int(np.sum(np.abs(self.quantum_tunneling_states) > 0.01))
                },
                "mechanical_stress": {
                    "max_stress_pa": float(np.max(self.stress_tensor)),
                    "avg_stress_pa": float(np.mean(self.stress_tensor)),
                    "high_stress_regions": int(np.sum(np.max(self.stress_tensor, axis=(2,3)) > 1e8))
                },
                "reliability": {
                    "fault_rate": float(np.sum(self.fault_map) / (self.rows * self.cols)),
                    "redundancy_coverage": float(np.sum(self.redundancy_map > 0) / max(1, np.sum(self.fault_map))),
                    "calibration_drift": float(np.mean(np.abs(self.drift_compensation))),
                    "reliability_score": float(1 - np.sum(self.fault_map) / (self.rows * self.cols))
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Multi-physics report generation failed: {e}")
            raise
    
    def __repr__(self) -> str:
        quantum_status = "quantum-aware" if np.any(self.quantum_tunneling_states != 0) else "classical"
        fault_rate = np.sum(self.fault_map) / (self.rows * self.cols) * 100
        return f"CrossbarArray({self.rows}x{self.cols}, {self.device_model.name}, {quantum_status}, {fault_rate:.2f}% faults)"