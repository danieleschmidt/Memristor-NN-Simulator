"""
Advanced fault tolerance mechanisms for memristive neural networks.

Implements:
- Error detection and correction codes
- Redundant computation strategies
- Adaptive routing around faults
- Dynamic weight reallocation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import time

from ..utils.logger import LoggingMixin
from ..utils.validators import validate_numpy_array, validate_positive_number


class FaultType(Enum):
    """Types of faults that can occur in memristor arrays."""
    STUCK_AT_HIGH = "stuck_at_high"
    STUCK_AT_LOW = "stuck_at_low"
    RESISTIVE_DRIFT = "resistive_drift"
    OPEN_CIRCUIT = "open_circuit"
    SHORT_CIRCUIT = "short_circuit"
    TEMPORAL_NOISE = "temporal_noise"


class ErrorCorrectionCode(Enum):
    """Error correction code types."""
    HAMMING = "hamming"
    BCH = "bch"
    REED_SOLOMON = "reed_solomon"
    LDPC = "ldpc"


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance mechanisms."""
    enable_ecc: bool = True
    ecc_type: ErrorCorrectionCode = ErrorCorrectionCode.HAMMING
    redundancy_factor: int = 3
    adaptive_routing: bool = True
    weight_reallocation: bool = True
    error_threshold: float = 0.05
    monitoring_interval: float = 1.0  # seconds
    auto_healing: bool = True
    

class FaultToleranceManager(LoggingMixin):
    """Comprehensive fault tolerance and error recovery system."""
    
    def __init__(
        self,
        crossbar_array,
        config: Optional[FaultToleranceConfig] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize fault tolerance manager.
        
        Args:
            crossbar_array: Target crossbar array
            config: Fault tolerance configuration
            enable_monitoring: Enable continuous monitoring
        """
        super().__init__()
        self.crossbar = crossbar_array
        self.config = config or FaultToleranceConfig()
        
        # Fault tracking
        self.detected_faults = {}
        self.fault_history = []
        self.error_statistics = {fault_type: 0 for fault_type in FaultType}
        
        # Redundancy management
        self.redundancy_groups = {}
        self.voting_weights = {}
        
        # Adaptive routing
        self.routing_table = self._initialize_routing_table()
        self.bypass_map = np.zeros((crossbar_array.rows, crossbar_array.cols), dtype=bool)
        
        # Error correction
        if self.config.enable_ecc:
            self.ecc_encoder = self._initialize_ecc()
            self.ecc_decoder = self._initialize_ecc_decoder()
        
        # Monitoring thread
        self.monitoring_active = enable_monitoring
        self.monitoring_thread = None
        if enable_monitoring:
            self._start_monitoring()
            
        self.logger.info(f"Fault tolerance initialized with {self.config.ecc_type.value} ECC")
    
    def _initialize_routing_table(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """Initialize adaptive routing table."""
        routing_table = {}
        
        for i in range(self.crossbar.rows):
            for j in range(self.crossbar.cols):
                # Create multiple routing paths for each connection
                primary_path = [(i, j)]
                
                # Add alternative paths through neighboring devices
                alternative_paths = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.crossbar.rows and 0 <= nj < self.crossbar.cols:
                            alternative_paths.append((ni, nj))
                
                routing_table[(i, j)] = [primary_path] + [[path] for path in alternative_paths[:3]]
        
        return routing_table
    
    def _initialize_ecc(self) -> Callable:
        """Initialize error correction coding encoder."""
        if self.config.ecc_type == ErrorCorrectionCode.HAMMING:
            return self._hamming_encode
        elif self.config.ecc_type == ErrorCorrectionCode.BCH:
            return self._bch_encode
        else:
            # Default to simple parity
            return self._parity_encode
    
    def _initialize_ecc_decoder(self) -> Callable:
        """Initialize error correction coding decoder."""
        if self.config.ecc_type == ErrorCorrectionCode.HAMMING:
            return self._hamming_decode
        elif self.config.ecc_type == ErrorCorrectionCode.BCH:
            return self._bch_decode
        else:
            # Default to simple parity
            return self._parity_decode
    
    def detect_faults(
        self,
        test_vectors: Optional[np.ndarray] = None,
        reference_outputs: Optional[np.ndarray] = None
    ) -> Dict[Tuple[int, int], FaultType]:
        """
        Detect faults in the crossbar array.
        
        Args:
            test_vectors: Input test patterns
            reference_outputs: Expected outputs for fault-free operation
            
        Returns:
            Dictionary mapping device coordinates to fault types
        """
        try:
            newly_detected_faults = {}
            
            if test_vectors is not None and reference_outputs is not None:
                # Functional testing approach
                newly_detected_faults.update(
                    self._functional_fault_detection(test_vectors, reference_outputs)
                )
            
            # Parametric testing (always performed)
            newly_detected_faults.update(self._parametric_fault_detection())
            
            # Update fault tracking
            for device_coord, fault_type in newly_detected_faults.items():
                if device_coord not in self.detected_faults:
                    self.detected_faults[device_coord] = []
                
                self.detected_faults[device_coord].append({
                    "fault_type": fault_type,
                    "detection_time": time.time(),
                    "detection_method": "automated"
                })
                
                self.error_statistics[fault_type] += 1
            
            # Update bypass map
            for device_coord in newly_detected_faults:
                self.bypass_map[device_coord] = True
            
            self.logger.info(f"Detected {len(newly_detected_faults)} new faults")
            
            return newly_detected_faults
            
        except Exception as e:
            self.logger.error(f"Fault detection failed: {e}")
            raise
    
    def _functional_fault_detection(
        self, 
        test_vectors: np.ndarray, 
        reference_outputs: np.ndarray
    ) -> Dict[Tuple[int, int], FaultType]:
        """Detect faults using functional testing."""
        faults = {}
        
        for test_idx, (input_vec, expected_output) in enumerate(zip(test_vectors, reference_outputs)):
            # Apply test vector
            actual_output = self.crossbar.analog_matmul(input_vec)
            
            # Check for significant deviations
            error = np.abs(actual_output - expected_output)
            relative_error = error / (np.abs(expected_output) + 1e-12)
            
            # Identify faulty devices contributing to errors
            if np.any(relative_error > self.config.error_threshold):
                # Simplified fault localization
                conductance_matrix = self.crossbar.get_conductance_matrix()
                
                for i in range(self.crossbar.rows):
                    for j in range(self.crossbar.cols):
                        device_contribution = conductance_matrix[i, j] * input_vec[i]
                        
                        # Check for stuck-at faults
                        if conductance_matrix[i, j] > 0.95:
                            faults[(i, j)] = FaultType.STUCK_AT_HIGH
                        elif conductance_matrix[i, j] < 0.05:
                            faults[(i, j)] = FaultType.STUCK_AT_LOW
        
        return faults
    
    def _parametric_fault_detection(self) -> Dict[Tuple[int, int], FaultType]:
        """Detect faults using parametric testing."""
        faults = {}
        
        # Get current conductance matrix
        conductances = self.crossbar.get_conductance_matrix()
        
        # Statistical analysis
        mean_conductance = np.mean(conductances)
        std_conductance = np.std(conductances)
        
        # Detect outliers (potential faults)
        z_scores = np.abs(conductances - mean_conductance) / (std_conductance + 1e-12)
        
        fault_locations = np.where(z_scores > 3.0)  # 3-sigma rule
        
        for i, j in zip(fault_locations[0], fault_locations[1]):
            conductance = conductances[i, j]
            
            if conductance > mean_conductance + 3 * std_conductance:
                faults[(i, j)] = FaultType.STUCK_AT_HIGH
            elif conductance < mean_conductance - 3 * std_conductance:
                faults[(i, j)] = FaultType.STUCK_AT_LOW
            else:
                faults[(i, j)] = FaultType.RESISTIVE_DRIFT
        
        return faults
    
    def apply_error_correction(
        self, 
        input_data: np.ndarray, 
        output_data: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Apply error correction to input/output data.
        
        Args:
            input_data: Raw input data
            output_data: Raw output data (potentially corrupted)
            
        Returns:
            Tuple of (corrected_output, correction_statistics)
        """
        try:
            if not self.config.enable_ecc:
                return output_data, {"corrections_applied": 0}
            
            # Encode input with error correction
            encoded_input = self.ecc_encoder(input_data)
            
            # Attempt to decode and correct output
            corrected_output, corrections = self.ecc_decoder(output_data, encoded_input)
            
            correction_stats = {
                "corrections_applied": corrections,
                "error_rate": corrections / len(output_data) if len(output_data) > 0 else 0,
                "ecc_overhead": len(encoded_input) / len(input_data) if len(input_data) > 0 else 0
            }
            
            return corrected_output, correction_stats
            
        except Exception as e:
            self.logger.error(f"Error correction failed: {e}")
            return output_data, {"corrections_applied": 0, "error": str(e)}
    
    def _hamming_encode(self, data: np.ndarray) -> np.ndarray:
        """Simple Hamming code encoding."""
        # Convert to binary representation
        data_bits = np.unpackbits(data.astype(np.uint8))
        
        # Add parity bits (simplified 7,4 Hamming code)
        encoded_data = []
        for i in range(0, len(data_bits), 4):
            block = data_bits[i:i+4]
            if len(block) < 4:
                block = np.pad(block, (0, 4 - len(block)))
            
            # Calculate parity bits
            p1 = block[0] ^ block[1] ^ block[3]
            p2 = block[0] ^ block[2] ^ block[3]
            p3 = block[1] ^ block[2] ^ block[3]
            
            # Hamming (7,4) code: p1, p2, d1, p3, d2, d3, d4
            encoded_block = [p1, p2, block[0], p3, block[1], block[2], block[3]]
            encoded_data.extend(encoded_block)
        
        return np.array(encoded_data, dtype=np.uint8)
    
    def _hamming_decode(
        self, 
        received_data: np.ndarray, 
        original_encoded: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Simple Hamming code decoding with error correction."""
        corrections = 0
        corrected_data = []
        
        # Process in 7-bit blocks
        for i in range(0, len(received_data), 7):
            if i + 6 >= len(received_data):
                break
                
            block = received_data[i:i+7]
            
            # Extract parity and data bits
            p1, p2, d1, p3, d2, d3, d4 = block
            
            # Calculate syndrome
            s1 = p1 ^ d1 ^ d2 ^ d4
            s2 = p2 ^ d1 ^ d3 ^ d4
            s3 = p3 ^ d2 ^ d3 ^ d4
            
            syndrome = s3 * 4 + s2 * 2 + s1
            
            if syndrome != 0:
                # Error detected and correctable
                if syndrome == 3:  # Error in d1
                    d1 = 1 - d1
                elif syndrome == 5:  # Error in d2
                    d2 = 1 - d2
                elif syndrome == 6:  # Error in d3
                    d3 = 1 - d3
                elif syndrome == 7:  # Error in d4
                    d4 = 1 - d4
                
                corrections += 1
            
            corrected_data.extend([d1, d2, d3, d4])
        
        return np.array(corrected_data, dtype=np.uint8), corrections
    
    def _parity_encode(self, data: np.ndarray) -> np.ndarray:
        """Simple parity bit encoding."""
        data_bits = np.unpackbits(data.astype(np.uint8))
        encoded_data = []
        
        # Add parity bit for every 8 bits
        for i in range(0, len(data_bits), 8):
            block = data_bits[i:i+8]
            parity = np.sum(block) % 2
            encoded_data.extend(list(block))
            encoded_data.append(parity)
        
        return np.array(encoded_data, dtype=np.uint8)
    
    def _parity_decode(
        self, 
        received_data: np.ndarray, 
        original_encoded: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Simple parity checking (detection only)."""
        corrections = 0
        corrected_data = []
        
        # Process in 9-bit blocks (8 data + 1 parity)
        for i in range(0, len(received_data), 9):
            if i + 8 >= len(received_data):
                break
                
            block = received_data[i:i+8]
            received_parity = received_data[i+8]
            
            calculated_parity = np.sum(block) % 2
            
            if calculated_parity != received_parity:
                # Error detected but not correctable with simple parity
                self.logger.warning(f"Parity error detected in block starting at {i}")
                corrections += 1
            
            corrected_data.extend(block)
        
        return np.array(corrected_data, dtype=np.uint8), corrections
    
    def _bch_encode(self, data: np.ndarray) -> np.ndarray:
        \"\"\"Placeholder for BCH encoding (would require galois field arithmetic).\"\"\"
        # For now, fall back to Hamming
        return self._hamming_encode(data)
    
    def _bch_decode(
        self, 
        received_data: np.ndarray, 
        original_encoded: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        \"\"\"Placeholder for BCH decoding.\"\"\"
        # For now, fall back to Hamming
        return self._hamming_decode(received_data, original_encoded)
    
    def configure_redundancy(
        self, 
        critical_weights: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, int]:
        \"\"\"
        Configure redundant computation for critical components.
        
        Args:
            critical_weights: List of critical weight coordinates
            
        Returns:
            Redundancy configuration statistics
        \"\"\"
        try:
            if critical_weights is None:
                # Automatically identify critical weights (simplified)
                conductances = self.crossbar.get_conductance_matrix()
                weight_importance = np.abs(conductances)
                
                # Select top weights by magnitude
                flat_indices = np.argsort(weight_importance.flatten())[::-1]
                top_indices = flat_indices[:min(100, len(flat_indices) // 10)]
                
                critical_weights = [
                    np.unravel_index(idx, weight_importance.shape)
                    for idx in top_indices
                ]
            
            # Create redundancy groups
            redundancy_groups = {}
            group_id = 0
            
            for weight_coord in critical_weights:
                if weight_coord not in self.detected_faults:
                    # Create redundancy group
                    redundancy_groups[group_id] = {
                        "primary": weight_coord,
                        "replicas": self._find_replica_locations(weight_coord),
                        "voting_scheme": "majority"
                    }
                    group_id += 1
            
            self.redundancy_groups = redundancy_groups
            
            stats = {
                "total_redundancy_groups": len(redundancy_groups),
                "critical_weights_protected": len(critical_weights),
                "average_replicas_per_group": np.mean([
                    len(group["replicas"]) for group in redundancy_groups.values()
                ]) if redundancy_groups else 0,
                "redundancy_overhead": len(redundancy_groups) * self.config.redundancy_factor
            }
            
            self.logger.info(f"Configured {len(redundancy_groups)} redundancy groups")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Redundancy configuration failed: {e}")
            raise
    
    def _find_replica_locations(
        self, 
        primary_coord: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        \"\"\"Find suitable locations for weight replicas.\"\"\"
        i, j = primary_coord
        replicas = []
        
        # Search for unused neighboring locations
        search_radius = 2
        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                if di == 0 and dj == 0:
                    continue
                
                ni, nj = i + di, j + dj
                if (0 <= ni < self.crossbar.rows and 
                    0 <= nj < self.crossbar.cols and
                    (ni, nj) not in self.detected_faults and
                    not self.bypass_map[ni, nj]):
                    
                    replicas.append((ni, nj))
                    
                    if len(replicas) >= self.config.redundancy_factor - 1:
                        break
            
            if len(replicas) >= self.config.redundancy_factor - 1:
                break
        
        return replicas
    
    def adaptive_weight_reallocation(
        self, 
        original_weights: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        \"\"\"
        Reallocate weights to avoid faulty devices.
        
        Args:
            original_weights: Original weight matrix
            
        Returns:
            Tuple of (reallocated_weights, reallocation_statistics)
        \"\"\"
        try:
            reallocated_weights = original_weights.copy()
            
            # Identify weights that need reallocation
            reallocation_count = 0
            total_weight_magnitude = 0
            
            for fault_coord in self.detected_faults:
                i, j = fault_coord
                if i < reallocated_weights.shape[0] and j < reallocated_weights.shape[1]:
                    original_weight = reallocated_weights[i, j]
                    total_weight_magnitude += abs(original_weight)
                    
                    # Find alternative location
                    alternative_coords = self._find_alternative_location(fault_coord)
                    
                    if alternative_coords:
                        # Distribute weight among alternatives
                        weight_per_alternative = original_weight / len(alternative_coords)
                        
                        for alt_i, alt_j in alternative_coords:
                            reallocated_weights[alt_i, alt_j] += weight_per_alternative
                        
                        # Zero out the faulty location
                        reallocated_weights[i, j] = 0
                        reallocation_count += 1
            
            # Calculate reallocation statistics
            weight_change = np.sum(np.abs(reallocated_weights - original_weights))
            
            stats = {
                "devices_reallocated": reallocation_count,
                "total_weight_change": float(weight_change),
                "weight_preservation_ratio": float(
                    1.0 - weight_change / (np.sum(np.abs(original_weights)) + 1e-12)
                ),
                "successful_reallocations": reallocation_count,
                "failed_reallocations": len(self.detected_faults) - reallocation_count
            }
            
            return reallocated_weights, stats
            
        except Exception as e:
            self.logger.error(f"Weight reallocation failed: {e}")
            return original_weights, {"error": str(e)}
    
    def _find_alternative_location(
        self, 
        fault_coord: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        \"\"\"Find alternative locations for a faulty device.\"\"\"
        i, j = fault_coord
        alternatives = []
        
        # Search in expanding rings around the fault
        for radius in range(1, 4):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if abs(di) + abs(dj) != radius:  # Only ring boundary
                        continue
                    
                    ni, nj = i + di, j + dj
                    if (0 <= ni < self.crossbar.rows and 
                        0 <= nj < self.crossbar.cols and
                        (ni, nj) not in self.detected_faults and
                        not self.bypass_map[ni, nj]):
                        
                        alternatives.append((ni, nj))
            
            if alternatives:
                break  # Use closest alternatives
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _start_monitoring(self) -> None:
        \"\"\"Start continuous fault monitoring thread.\"\"\"
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Perform periodic fault detection
                    self.detect_faults()
                    
                    # Update routing table if needed
                    if self.config.adaptive_routing:
                        self._update_routing_table()
                    
                    # Sleep until next monitoring cycle
                    time.sleep(self.config.monitoring_interval)
                    
                except Exception as e:
                    self.logger.error(f\"Monitoring thread error: {e}\")
                    time.sleep(self.config.monitoring_interval * 2)  # Back off on error
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info(\"Fault monitoring thread started\")
    
    def _update_routing_table(self) -> None:
        \"\"\"Update routing table to avoid faulty devices.\"\"\"
        for device_coord in self.detected_faults:
            # Mark device for bypass in all routes
            for route_key, paths in self.routing_table.items():
                updated_paths = []
                for path in paths:
                    if device_coord not in path:
                        updated_paths.append(path)
                
                if not updated_paths:
                    # Find new path avoiding the fault
                    new_path = self._find_bypass_path(route_key[0], route_key[1], device_coord)
                    if new_path:
                        updated_paths = [new_path]
                
                self.routing_table[route_key] = updated_paths
    
    def _find_bypass_path(
        self, 
        start_coord: Tuple[int, int], 
        end_coord: Tuple[int, int], 
        avoid_coord: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        \"\"\"Find path from start to end while avoiding a specific coordinate.\"\"\"
        # Simple path finding (could be enhanced with A* or similar)
        si, sj = start_coord
        ei, ej = end_coord
        ai, aj = avoid_coord
        
        # Try direct path first
        if (si, sj) != avoid_coord and (ei, ej) != avoid_coord:
            return [(si, sj), (ei, ej)]
        
        # Find intermediate point
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                intermediate = (si + di, sj + dj)
                if (intermediate != avoid_coord and 
                    0 <= intermediate[0] < self.crossbar.rows and
                    0 <= intermediate[1] < self.crossbar.cols):
                    return [(si, sj), intermediate, (ei, ej)]
        
        return None
    
    def stop_monitoring(self) -> None:
        \"\"\"Stop continuous monitoring.\"\"\"
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info(\"Fault monitoring stopped\")
    
    def get_fault_tolerance_report(self) -> Dict:
        \"\"\"Generate comprehensive fault tolerance status report.\"\"\"
        try:
            return {
                \"fault_detection\": {
                    \"total_faults_detected\": len(self.detected_faults),
                    \"fault_types_breakdown\": dict(self.error_statistics),
                    \"fault_detection_rate\": len(self.fault_history) / max(1, time.time() - (self.fault_history[0]['detection_time'] if self.fault_history else time.time())),
                    \"most_common_fault\": max(self.error_statistics, key=self.error_statistics.get).value if any(self.error_statistics.values()) else \"none\"
                },
                \"error_correction\": {
                    \"ecc_enabled\": self.config.enable_ecc,
                    \"ecc_type\": self.config.ecc_type.value,
                    \"correction_overhead\": 0.33 if self.config.enable_ecc else 0  # Approximate
                },
                \"redundancy\": {
                    \"redundancy_groups\": len(self.redundancy_groups),
                    \"redundancy_factor\": self.config.redundancy_factor,
                    \"critical_components_protected\": sum(1 for group in self.redundancy_groups.values())
                },
                \"adaptive_systems\": {
                    \"adaptive_routing_enabled\": self.config.adaptive_routing,
                    \"weight_reallocation_enabled\": self.config.weight_reallocation,
                    \"auto_healing_enabled\": self.config.auto_healing,
                    \"monitoring_active\": self.monitoring_active
                },
                \"system_health\": {
                    \"healthy_devices\": int(np.sum(~self.bypass_map)),
                    \"total_devices\": int(self.crossbar.rows * self.crossbar.cols),
                    \"system_availability\": float(np.sum(~self.bypass_map) / (self.crossbar.rows * self.crossbar.cols)),
                    \"fault_coverage\": float(len(self.detected_faults) / max(1, len(self.detected_faults) + np.sum(~self.bypass_map)))
                }
            }
            
        except Exception as e:
            self.logger.error(f\"Report generation failed: {e}\")
            raise