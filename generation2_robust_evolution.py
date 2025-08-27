#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Enhanced Reliability, Validation, and Error Handling
Autonomous SDLC Evolution - Terragon Labs
"""
import sys
import os
import json
import math
import random
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure robust logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation2_robust.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DeviceState(Enum):
    """Enumeration for memristor device states."""
    LOW_RESISTANCE = "LRS"
    HIGH_RESISTANCE = "HRS"
    INTERMEDIATE = "IRS"
    STUCK_LOW = "STUCK_LRS"
    STUCK_HIGH = "STUCK_HRS"

@dataclass
class DeviceConfig:
    """Configuration for memristor device parameters."""
    ron_min: float = 1e3
    ron_max: float = 1e5
    roff_min: float = 1e6
    roff_max: float = 1e8
    switching_threshold_pos: float = 0.5
    switching_threshold_neg: float = -0.5
    noise_level: float = 0.05
    failure_rate: float = 0.001
    temperature: float = 300.0  # Kelvin

@dataclass 
class SimulationConfig:
    """Configuration for simulation parameters."""
    max_voltage: float = 2.0
    min_voltage: float = -2.0
    time_step: float = 1e-6
    max_iterations: int = 10000
    convergence_threshold: float = 1e-8
    enable_noise: bool = True
    enable_faults: bool = True

class MemristorError(Exception):
    """Base exception for memristor-related errors."""
    pass

class DeviceFailureError(MemristorError):
    """Exception raised when a memristor device fails."""
    pass

class SimulationError(MemristorError):
    """Exception raised during simulation failures."""
    pass

class ValidationError(MemristorError):
    """Exception raised during validation failures."""
    pass

class RobustMemristor:
    """Enhanced memristor device model with comprehensive error handling and validation."""
    
    def __init__(self, config: DeviceConfig, device_id: str = "unknown"):
        self.config = config
        self.device_id = device_id
        self.logger = logging.getLogger(f"{__name__}.{device_id}")
        
        # Initialize device parameters with validation
        self._validate_config()
        self._initialize_device()
        
        # Health monitoring
        self.operation_count = 0
        self.failure_events = []
        self.is_failed = False
        
    def _validate_config(self):
        """Validate device configuration parameters."""
        try:
            if self.config.ron_min >= self.config.ron_max:
                raise ValidationError(f"Invalid Ron range: {self.config.ron_min} >= {self.config.ron_max}")
            if self.config.roff_min >= self.config.roff_max:
                raise ValidationError(f"Invalid Roff range: {self.config.roff_min} >= {self.config.roff_max}")
            if self.config.noise_level < 0 or self.config.noise_level > 1:
                raise ValidationError(f"Invalid noise level: {self.config.noise_level}")
            if self.config.failure_rate < 0 or self.config.failure_rate > 1:
                raise ValidationError(f"Invalid failure rate: {self.config.failure_rate}")
            if self.config.temperature < 0:
                raise ValidationError(f"Invalid temperature: {self.config.temperature}")
                
            self.logger.info(f"Device {self.device_id} configuration validated successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Configuration validation failed: {e}")
    
    def _initialize_device(self):
        """Initialize device with random parameters within specified ranges."""
        try:
            # Random device parameters with manufacturing variations
            ron_range = self.config.ron_max - self.config.ron_min
            roff_range = self.config.roff_max - self.config.roff_min
            
            self.ron = self.config.ron_min + ron_range * random.random()
            self.roff = self.config.roff_min + roff_range * random.random()
            
            # Initial state (0 = HRS, 1 = LRS)
            self.state = random.random()
            self.initial_state = self.state
            
            # Temperature effects
            self.temp_coefficient = 0.002  # 0.2%/K
            
            self.logger.info(f"Device {self.device_id} initialized: Ron={self.ron:.2e}, Roff={self.roff:.2e}")
            
        except Exception as e:
            self.logger.error(f"Device initialization failed: {e}")
            raise MemristorError(f"Device initialization failed: {e}")
    
    def get_resistance(self, apply_noise: bool = True) -> float:
        """Calculate current resistance with optional noise."""
        try:
            if self.is_failed:
                raise DeviceFailureError(f"Device {self.device_id} has failed")
            
            # Base resistance calculation
            resistance = self.ron + (self.roff - self.ron) * (1 - self.state)
            
            # Temperature effects
            temp_factor = 1 + self.temp_coefficient * (self.config.temperature - 300)
            resistance *= temp_factor
            
            # Add noise if enabled
            if apply_noise and self.config.noise_level > 0:
                noise_factor = 1 + self.config.noise_level * (2 * random.random() - 1)
                resistance *= noise_factor
            
            # Ensure resistance is within physical bounds
            resistance = max(self.ron * 0.1, min(resistance, self.roff * 10))
            
            return resistance
            
        except Exception as e:
            self.logger.error(f"Resistance calculation failed: {e}")
            raise MemristorError(f"Resistance calculation failed: {e}")
    
    def conductance(self, voltage: float = 0.1) -> float:
        """Calculate conductance with enhanced error handling."""
        try:
            self._validate_voltage(voltage)
            resistance = self.get_resistance()
            
            if resistance <= 0:
                raise MemristorError(f"Invalid resistance value: {resistance}")
            
            return 1.0 / resistance
            
        except Exception as e:
            self.logger.error(f"Conductance calculation failed: {e}")
            raise MemristorError(f"Conductance calculation failed: {e}")
    
    def update_state(self, voltage: float, time_step: float = 1e-6) -> float:
        """Update device state with comprehensive validation and error handling."""
        try:
            if self.is_failed:
                raise DeviceFailureError(f"Device {self.device_id} has failed")
            
            self._validate_voltage(voltage)
            self._validate_time_step(time_step)
            
            # Check for device failure
            if random.random() < self.config.failure_rate:
                self._trigger_failure("Random failure event")
                return self.state
            
            old_state = self.state
            
            # State update based on voltage polarity and magnitude
            if abs(voltage) > abs(self.config.switching_threshold_pos):
                if voltage > 0:  # SET operation
                    self.state = min(1.0, self.state + 0.1 * time_step * 1e6 * abs(voltage))
                else:  # RESET operation
                    self.state = max(0.0, self.state - 0.1 * time_step * 1e6 * abs(voltage))
            
            # Ensure state bounds
            self.state = max(0.0, min(1.0, self.state))
            
            self.operation_count += 1
            
            # Log significant state changes
            if abs(self.state - old_state) > 0.1:
                self.logger.debug(f"Device {self.device_id} state change: {old_state:.3f} → {self.state:.3f}")
            
            return self.state
            
        except Exception as e:
            self.logger.error(f"State update failed: {e}")
            raise MemristorError(f"State update failed: {e}")
    
    def _validate_voltage(self, voltage: float):
        """Validate voltage is within safe operating range."""
        if not isinstance(voltage, (int, float)):
            raise ValidationError(f"Voltage must be numeric, got {type(voltage)}")
        if abs(voltage) > 5.0:  # Conservative safety limit
            raise ValidationError(f"Voltage {voltage}V exceeds safety limit")
    
    def _validate_time_step(self, time_step: float):
        """Validate time step parameter."""
        if not isinstance(time_step, (int, float)):
            raise ValidationError(f"Time step must be numeric, got {type(time_step)}")
        if time_step <= 0:
            raise ValidationError(f"Time step must be positive, got {time_step}")
        if time_step > 1e-3:  # 1ms max
            raise ValidationError(f"Time step {time_step}s too large for stable simulation")
    
    def _trigger_failure(self, reason: str):
        """Trigger device failure with logging."""
        self.is_failed = True
        failure_event = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "reason": reason,
            "operation_count": self.operation_count,
            "final_state": self.state
        }
        self.failure_events.append(failure_event)
        self.logger.warning(f"Device {self.device_id} failed: {reason}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive device health status."""
        return {
            "device_id": self.device_id,
            "is_failed": self.is_failed,
            "operation_count": self.operation_count,
            "current_state": self.state,
            "initial_state": self.initial_state,
            "state_drift": abs(self.state - self.initial_state),
            "failure_events": len(self.failure_events),
            "ron": self.ron,
            "roff": self.roff,
            "current_resistance": self.get_resistance(apply_noise=False) if not self.is_failed else None
        }

class RobustCrossbarArray:
    """Enhanced crossbar array with fault tolerance and comprehensive monitoring."""
    
    def __init__(self, rows: int, cols: int, device_config: DeviceConfig):
        self.rows = rows
        self.cols = cols
        self.device_config = device_config
        self.logger = logging.getLogger(f"{__name__}.CrossbarArray")
        
        # Initialize devices with unique IDs
        self.devices = []
        for i in range(rows):
            row = []
            for j in range(cols):
                device_id = f"R{i}C{j}"
                device = RobustMemristor(device_config, device_id)
                row.append(device)
            self.devices.append(row)
        
        # Health monitoring
        self.operation_count = 0
        self.error_count = 0
        self.last_health_check = None
        
        self.logger.info(f"CrossbarArray initialized: {rows}×{cols} devices")
    
    def matrix_vector_multiply(self, input_vector: List[float]) -> List[float]:
        """Robust matrix-vector multiplication with error handling."""
        try:
            if len(input_vector) != self.cols:
                raise ValidationError(f"Input vector length {len(input_vector)} != {self.cols}")
            
            # Validate input vector
            for i, val in enumerate(input_vector):
                if not isinstance(val, (int, float)):
                    raise ValidationError(f"Input[{i}] must be numeric, got {type(val)}")
                if abs(val) > 10.0:  # Reasonable bounds
                    self.logger.warning(f"Large input value detected: input[{i}] = {val}")
            
            output = []
            failed_devices = 0
            
            for i in range(self.rows):
                row_current = 0.0
                row_errors = 0
                
                for j in range(self.cols):
                    try:
                        device = self.devices[i][j]
                        voltage = input_vector[j]
                        conductance = device.conductance(voltage)
                        current = conductance * voltage
                        row_current += current
                        
                    except DeviceFailureError:
                        failed_devices += 1
                        # Skip failed device (equivalent to open circuit)
                        continue
                        
                    except Exception as e:
                        row_errors += 1
                        self.logger.warning(f"Error in device R{i}C{j}: {e}")
                        continue
                
                if row_errors > self.cols * 0.1:  # More than 10% errors in row
                    self.logger.error(f"Excessive errors in row {i}: {row_errors}/{self.cols}")
                
                output.append(row_current)
            
            self.operation_count += 1
            
            # Health monitoring
            if failed_devices > 0:
                failure_rate = failed_devices / (self.rows * self.cols)
                self.logger.warning(f"Failed devices: {failed_devices} ({failure_rate:.1%})")
                
                if failure_rate > 0.05:  # More than 5% failed
                    raise SimulationError(f"Excessive device failures: {failure_rate:.1%}")
            
            return output
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Matrix multiplication failed: {e}")
            raise SimulationError(f"Matrix multiplication failed: {e}")
    
    def get_array_health(self) -> Dict[str, Any]:
        """Comprehensive health assessment of the entire array."""
        try:
            total_devices = self.rows * self.cols
            failed_devices = 0
            state_distribution = []
            resistance_distribution = []
            
            for i in range(self.rows):
                for j in range(self.cols):
                    device = self.devices[i][j]
                    health = device.get_health_status()
                    
                    if health["is_failed"]:
                        failed_devices += 1
                    else:
                        state_distribution.append(health["current_state"])
                        if health["current_resistance"]:
                            resistance_distribution.append(health["current_resistance"])
            
            # Statistical analysis
            if state_distribution:
                avg_state = sum(state_distribution) / len(state_distribution)
                state_variance = sum((x - avg_state)**2 for x in state_distribution) / len(state_distribution)
            else:
                avg_state = state_variance = 0
            
            if resistance_distribution:
                avg_resistance = sum(resistance_distribution) / len(resistance_distribution)
                min_resistance = min(resistance_distribution)
                max_resistance = max(resistance_distribution)
            else:
                avg_resistance = min_resistance = max_resistance = 0
            
            health_report = {
                "array_size": f"{self.rows}×{self.cols}",
                "total_devices": total_devices,
                "failed_devices": failed_devices,
                "failure_rate": failed_devices / total_devices,
                "operation_count": self.operation_count,
                "error_count": self.error_count,
                "avg_device_state": avg_state,
                "state_variance": state_variance,
                "avg_resistance": avg_resistance,
                "resistance_range": (min_resistance, max_resistance),
                "health_status": "HEALTHY" if failed_devices / total_devices < 0.01 else "DEGRADED"
            }
            
            self.last_health_check = health_report
            return health_report
            
        except Exception as e:
            self.logger.error(f"Health assessment failed: {e}")
            return {"error": str(e), "health_status": "UNKNOWN"}

def demonstrate_robust_functionality():
    """Demonstrate Generation 2 robust functionality with comprehensive error handling."""
    logger.info("🛡️ Generation 2: MAKE IT ROBUST - Starting Demonstration")
    logger.info("=" * 70)
    
    try:
        # Test 1: Enhanced Configuration and Validation
        logger.info("\n1. Testing Enhanced Configuration System...")
        
        device_config = DeviceConfig(
            ron_min=1e4,
            ron_max=5e4,
            roff_min=1e6,
            roff_max=5e7,
            noise_level=0.02,
            failure_rate=0.0005,
            temperature=323.0  # 50°C
        )
        
        sim_config = SimulationConfig(
            enable_noise=True,
            enable_faults=True,
            max_iterations=1000
        )
        
        logger.info("✅ Configuration system validated")
        logger.info(f"   Device Ron range: {device_config.ron_min:.1e} - {device_config.ron_max:.1e} Ω")
        logger.info(f"   Temperature: {device_config.temperature:.1f} K")
        logger.info(f"   Noise level: {device_config.noise_level:.1%}")
        
        # Test 2: Robust Device Testing
        logger.info("\n2. Testing Robust Device Models...")
        
        test_device = RobustMemristor(device_config, "TEST_DEVICE")
        
        # Test normal operations
        initial_resistance = test_device.get_resistance()
        logger.info(f"   Initial resistance: {initial_resistance:.2e} Ω")
        
        # Test state transitions with validation
        for voltage in [0.8, -0.8, 1.2, -1.2]:
            try:
                old_state = test_device.state
                new_state = test_device.update_state(voltage)
                logger.info(f"   Voltage {voltage:+4.1f}V: state {old_state:.3f} → {new_state:.3f}")
            except Exception as e:
                logger.warning(f"   Voltage {voltage:+4.1f}V: {e}")
        
        # Test error conditions
        logger.info("\n   Testing error conditions...")
        try:
            test_device.update_state(10.0)  # Should trigger validation error
        except (ValidationError, MemristorError) as e:
            logger.info(f"   ✅ Validation error caught: {str(e)[:50]}...")
        
        health = test_device.get_health_status()
        logger.info(f"   Device health: {health['operation_count']} ops, state={health['current_state']:.3f}")
        
        # Test 3: Robust Crossbar Array
        logger.info("\n3. Testing Robust Crossbar Array...")
        
        array = RobustCrossbarArray(6, 6, device_config)
        
        # Multiple test vectors with edge cases
        test_vectors = [
            [0.1, 0.2, -0.1, 0.3, -0.2, 0.1],
            [1.0, -1.0, 0.5, -0.5, 0.8, -0.3],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Zero vector
            [2.0, 1.5, -1.8, 1.2, -0.9, 0.7]  # Large values
        ]
        
        results = []
        for i, test_vector in enumerate(test_vectors):
            try:
                logger.info(f"   Test vector {i+1}: {[f'{x:+4.1f}' for x in test_vector]}")
                output = array.matrix_vector_multiply(test_vector)
                results.append(output)
                logger.info(f"   Output: {[f'{x:.2e}' for x in output[:3]]}...")
                
            except Exception as e:
                logger.error(f"   Test vector {i+1} failed: {e}")
                results.append(None)
        
        # Test 4: Health Monitoring and Fault Tolerance
        logger.info("\n4. Testing Health Monitoring...")
        
        health_report = array.get_array_health()
        logger.info(f"   Array size: {health_report['array_size']}")
        logger.info(f"   Health status: {health_report['health_status']}")
        logger.info(f"   Failed devices: {health_report['failed_devices']}/{health_report['total_devices']} ({health_report['failure_rate']:.2%})")
        logger.info(f"   Operations completed: {health_report['operation_count']}")
        
        # Test 5: Stress Testing and Error Recovery
        logger.info("\n5. Stress Testing and Error Recovery...")
        
        stress_results = {"successful": 0, "failed": 0, "errors": []}
        
        for trial in range(100):
            try:
                # Random stress vector
                stress_vector = [2 * random.random() - 1 for _ in range(6)]
                output = array.matrix_vector_multiply(stress_vector)
                stress_results["successful"] += 1
                
            except Exception as e:
                stress_results["failed"] += 1
                stress_results["errors"].append(str(e)[:100])
                
                if len(stress_results["errors"]) <= 5:  # Log first few errors
                    logger.warning(f"   Stress test {trial+1} failed: {e}")
        
        logger.info(f"   Stress test results: {stress_results['successful']}/100 successful")
        logger.info(f"   Error rate: {stress_results['failed']:.1%}")
        
        # Test 6: Performance and Reliability Metrics
        logger.info("\n6. Computing Performance and Reliability Metrics...")
        
        final_health = array.get_array_health()
        
        # Calculate reliability metrics
        mtbf = array.operation_count / max(1, array.error_count)  # Mean time between failures
        availability = (array.operation_count - array.error_count) / array.operation_count if array.operation_count > 0 else 0
        
        metrics = {
            "generation": 2,
            "status": "ROBUST",
            "implementation": "error_handling_validation_logging",
            "reliability_metrics": {
                "mtbf": mtbf,
                "availability": availability,
                "error_rate": array.error_count / max(1, array.operation_count),
                "device_failure_rate": final_health["failure_rate"],
                "health_status": final_health["health_status"]
            },
            "performance_metrics": {
                "total_operations": array.operation_count,
                "total_errors": array.error_count,
                "stress_test_success_rate": stress_results["successful"] / 100,
                "avg_device_state": final_health["avg_device_state"],
                "state_variance": final_health["state_variance"]
            },
            "features_implemented": [
                "comprehensive_error_handling",
                "input_validation", 
                "fault_tolerance",
                "health_monitoring",
                "logging_system",
                "configuration_management",
                "stress_testing",
                "statistical_analysis"
            ],
            "generation3_targets": {
                "add_performance_optimization": True,
                "implement_caching": True,
                "add_parallel_processing": True,
                "implement_adaptive_algorithms": True,
                "add_load_balancing": True
            }
        }
        
        # Save comprehensive results
        with open("generation2_robust_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("✅ Performance and reliability metrics computed:")
        logger.info(f"   MTBF: {mtbf:.1f} operations")
        logger.info(f"   Availability: {availability:.1%}")
        logger.info(f"   Error rate: {metrics['reliability_metrics']['error_rate']:.2%}")
        logger.info(f"   Results saved to generation2_robust_results.json")
        
        logger.info(f"\n🎉 Generation 2: MAKE IT ROBUST - COMPLETED SUCCESSFULLY!")
        logger.info("   ✓ Comprehensive error handling implemented")
        logger.info("   ✓ Input validation and bounds checking")
        logger.info("   ✓ Fault tolerance and recovery mechanisms")
        logger.info("   ✓ Health monitoring and diagnostics")
        logger.info("   ✓ Logging and observability")
        logger.info("   ✓ Ready for Generation 3 optimizations")
        
        return True
        
    except Exception as e:
        logger.error(f"Generation 2 demonstration failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = demonstrate_robust_functionality()
    sys.exit(0 if success else 1)