"""
Advanced reliability analysis for memristive crossbar arrays.

Implements comprehensive reliability models including:
- Time-dependent dielectric breakdown (TDDB)
- Electromigration effects
- Thermal cycling stress
- Statistical lifetime prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
from pathlib import Path

from ..utils.logger import LoggingMixin
from ..utils.validators import validate_positive_number, validate_numpy_array


@dataclass
class ReliabilityModel:
    """Statistical reliability model parameters."""
    weibull_shape: float = 2.0  # Shape parameter (beta)
    weibull_scale: float = 1000.0  # Scale parameter (eta) in hours
    activation_energy: float = 0.8  # eV
    voltage_acceleration: float = 2.5  # Voltage acceleration factor
    temperature_coefficient: float = 0.1  # /K
    electromigration_factor: float = 0.01  # A/μm²
    
    # Confidence intervals
    confidence_levels: List[float] = field(default_factory=lambda: [0.90, 0.95, 0.99])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "weibull_shape": self.weibull_shape,
            "weibull_scale": self.weibull_scale,
            "activation_energy": self.activation_energy,
            "voltage_acceleration": self.voltage_acceleration,
            "temperature_coefficient": self.temperature_coefficient,
            "electromigration_factor": self.electromigration_factor,
            "confidence_levels": self.confidence_levels
        }


class ReliabilityAnalyzer(LoggingMixin):
    """Advanced reliability analysis and prediction engine."""
    
    def __init__(
        self,
        crossbar_array,
        model: Optional[ReliabilityModel] = None,
        enable_physics_models: bool = True
    ):
        """
        Initialize reliability analyzer.
        
        Args:
            crossbar_array: Target crossbar array to analyze
            model: Reliability model parameters
            enable_physics_models: Enable physics-based degradation models
        """
        super().__init__()
        self.crossbar = crossbar_array
        self.model = model or ReliabilityModel()
        self.enable_physics = enable_physics_models
        
        # Initialize reliability tracking
        self.device_ages = np.zeros((crossbar_array.rows, crossbar_array.cols))
        self.stress_history = []
        self.failure_times = {}
        self.degradation_states = np.ones((crossbar_array.rows, crossbar_array.cols))
        
        # Physics-based degradation tracking
        if self.enable_physics:
            self.tddb_damage = np.zeros((crossbar_array.rows, crossbar_array.cols))
            self.electromigration_damage = np.zeros((crossbar_array.rows, crossbar_array.cols))
            self.thermal_fatigue = np.zeros((crossbar_array.rows, crossbar_array.cols))
        
        self.logger.info("Reliability analyzer initialized with physics-based models")
    
    def predict_lifetime_distribution(
        self, 
        operating_conditions: Dict[str, float],
        monte_carlo_samples: int = 10000
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Predict lifetime distribution using Monte Carlo simulation.
        
        Args:
            operating_conditions: Temperature, voltage, current density
            monte_carlo_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with lifetime statistics and confidence intervals
        """
        try:
            temp_k = operating_conditions.get("temperature_k", 350)
            voltage_v = operating_conditions.get("voltage_v", 1.0)
            current_density = operating_conditions.get("current_density_a_per_cm2", 1e4)
            
            # Validate inputs
            temp_k = validate_positive_number(temp_k, "temperature_k", min_value=200, max_value=500)
            voltage_v = validate_positive_number(voltage_v, "voltage_v", max_value=10.0)
            current_density = validate_positive_number(current_density, "current_density", max_value=1e6)
            
            self.logger.info(f"Predicting lifetime for T={temp_k}K, V={voltage_v}V, J={current_density:.0f}A/cm²")
            
            # Calculate acceleration factors
            temp_acceleration = np.exp(-self.model.activation_energy * 1.602e-19 / 
                                    (1.381e-23 * temp_k))
            voltage_acceleration = (voltage_v / 1.0) ** self.model.voltage_acceleration
            current_acceleration = (current_density / 1e4) ** 1.5
            
            # Base lifetime with acceleration
            base_lifetime = self.model.weibull_scale / (
                temp_acceleration * voltage_acceleration * current_acceleration
            )
            
            # Monte Carlo simulation
            lifetimes = np.random.weibull(
                self.model.weibull_shape, 
                size=monte_carlo_samples
            ) * base_lifetime
            
            # Calculate statistics
            results = {
                "mean_lifetime_hours": float(np.mean(lifetimes)),
                "median_lifetime_hours": float(np.median(lifetimes)),
                "std_lifetime_hours": float(np.std(lifetimes)),
                "min_lifetime_hours": float(np.min(lifetimes)),
                "max_lifetime_hours": float(np.max(lifetimes)),
                "failure_rate_per_hour": float(1.0 / np.mean(lifetimes)),
                "confidence_intervals": {}
            }
            
            # Calculate confidence intervals
            for confidence in self.model.confidence_levels:
                lower_percentile = (1 - confidence) * 50
                upper_percentile = 100 - lower_percentile
                
                results["confidence_intervals"][f"{confidence:.0%}"] = {
                    "lower_bound": float(np.percentile(lifetimes, lower_percentile)),
                    "upper_bound": float(np.percentile(lifetimes, upper_percentile))
                }
            
            # Add physics-based corrections if enabled
            if self.enable_physics:
                physics_corrections = self._calculate_physics_corrections(
                    temp_k, voltage_v, current_density
                )
                results.update(physics_corrections)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Lifetime prediction failed: {e}")
            raise
    
    def _calculate_physics_corrections(
        self, 
        temp_k: float, 
        voltage_v: float, 
        current_density: float
    ) -> Dict[str, float]:
        """Calculate physics-based reliability corrections."""
        
        # Time-dependent dielectric breakdown (TDDB)
        electric_field = voltage_v / 5e-9  # V/m (assuming 5nm oxide)
        tddb_time_to_failure = 1e10 * np.exp(-0.5 * electric_field / 1e8)  # Simplified model
        
        # Electromigration
        em_activation_energy = 0.9  # eV for Cu interconnects
        em_current_exponent = 2.0
        em_time_to_failure = (1e15 / (current_density ** em_current_exponent)) * \
                           np.exp(em_activation_energy * 1.602e-19 / (1.381e-23 * temp_k))
        
        # Thermal cycling (Coffin-Manson model)
        delta_temp = temp_k - 300  # Temperature swing from ambient
        thermal_cycles_to_failure = 1e6 * (delta_temp / 50) ** -2.0  # Simplified
        
        return {
            "tddb_time_to_failure_hours": float(tddb_time_to_failure / 3600),
            "electromigration_ttf_hours": float(em_time_to_failure / 3600),
            "thermal_cycling_ttf_hours": float(thermal_cycles_to_failure * 24),  # Assuming daily cycles
            "dominant_failure_mode": self._identify_dominant_mode(
                tddb_time_to_failure, em_time_to_failure, thermal_cycles_to_failure * 24 * 3600
            )
        }
    
    def _identify_dominant_mode(self, tddb_ttf: float, em_ttf: float, thermal_ttf: float) -> str:
        """Identify the dominant failure mechanism."""
        failure_modes = {
            "TDDB": tddb_ttf,
            "Electromigration": em_ttf,
            "Thermal_Cycling": thermal_ttf
        }
        
        return min(failure_modes, key=failure_modes.get)
    
    def update_stress_conditions(
        self,
        voltage_stress: np.ndarray,
        current_stress: np.ndarray,
        temperature_map: np.ndarray,
        duration_hours: float
    ) -> Dict[str, float]:
        """
        Update stress conditions and accumulate damage.
        
        Args:
            voltage_stress: Per-device voltage stress
            current_stress: Per-device current stress
            temperature_map: Temperature distribution
            duration_hours: Duration of stress application
            
        Returns:
            Damage accumulation report
        """
        try:
            # Validate inputs
            voltage_stress = validate_numpy_array(
                voltage_stress, 
                expected_shape=(self.crossbar.rows, self.crossbar.cols),
                name="voltage_stress"
            )
            current_stress = validate_numpy_array(
                current_stress,
                expected_shape=(self.crossbar.rows, self.crossbar.cols), 
                name="current_stress"
            )
            temperature_map = validate_numpy_array(
                temperature_map,
                expected_shape=(self.crossbar.rows, self.crossbar.cols),
                name="temperature_map"
            )
            
            duration_hours = validate_positive_number(duration_hours, "duration_hours", max_value=1e6)
            
            # Update device ages
            self.device_ages += duration_hours
            
            # Calculate and accumulate damage if physics models enabled
            if self.enable_physics:
                self._accumulate_tddb_damage(voltage_stress, temperature_map, duration_hours)
                self._accumulate_em_damage(current_stress, temperature_map, duration_hours)
                self._accumulate_thermal_damage(temperature_map, duration_hours)
            
            # Update degradation states
            total_damage = np.zeros_like(self.degradation_states)
            
            if self.enable_physics:
                total_damage = (self.tddb_damage + 
                              self.electromigration_damage + 
                              self.thermal_fatigue)
            
            # Update degradation (damage reduces device performance)
            self.degradation_states = np.maximum(
                0.1,  # Minimum degradation state (10% of original performance)
                1.0 - total_damage
            )
            
            # Store stress history
            stress_entry = {
                "timestamp": len(self.stress_history),
                "duration_hours": duration_hours,
                "avg_voltage": float(np.mean(voltage_stress)),
                "max_voltage": float(np.max(voltage_stress)),
                "avg_current_density": float(np.mean(current_stress)),
                "max_temperature_k": float(np.max(temperature_map)),
                "total_damage": float(np.mean(total_damage))
            }
            self.stress_history.append(stress_entry)
            
            # Generate damage report
            damage_report = {
                "total_accumulated_damage": float(np.mean(total_damage)),
                "max_damage_device": float(np.max(total_damage)),
                "devices_above_50_percent_damage": int(np.sum(total_damage > 0.5)),
                "estimated_remaining_lifetime_hours": float(np.mean(
                    self.model.weibull_scale * (1 - total_damage)
                )),
                "degradation_variance": float(np.var(self.degradation_states))
            }
            
            if self.enable_physics:
                damage_report.update({
                    "tddb_contribution": float(np.mean(self.tddb_damage)),
                    "em_contribution": float(np.mean(self.electromigration_damage)),
                    "thermal_contribution": float(np.mean(self.thermal_fatigue))
                })
            
            return damage_report
            
        except Exception as e:
            self.logger.error(f"Stress condition update failed: {e}")
            raise
    
    def _accumulate_tddb_damage(
        self, 
        voltage_stress: np.ndarray, 
        temperature_map: np.ndarray, 
        duration_hours: float
    ) -> None:
        """Accumulate time-dependent dielectric breakdown damage."""
        # Electric field calculation (simplified)
        oxide_thickness = 5e-9  # 5 nm
        electric_field = voltage_stress / oxide_thickness
        
        # TDDB damage rate (simplified power law model)
        damage_rate = 1e-10 * (electric_field / 1e8) ** 3 * \
                     np.exp(-0.8 * 1.602e-19 / (1.381e-23 * temperature_map))
        
        # Accumulate damage
        self.tddb_damage += damage_rate * duration_hours
    
    def _accumulate_em_damage(
        self, 
        current_stress: np.ndarray, 
        temperature_map: np.ndarray, 
        duration_hours: float
    ) -> None:
        """Accumulate electromigration damage."""
        # Current density effect
        damage_rate = 1e-12 * (current_stress / 1e4) ** 2 * \
                     np.exp(-0.9 * 1.602e-19 / (1.381e-23 * temperature_map))
        
        # Accumulate damage
        self.electromigration_damage += damage_rate * duration_hours
    
    def _accumulate_thermal_damage(
        self, 
        temperature_map: np.ndarray, 
        duration_hours: float
    ) -> None:
        """Accumulate thermal fatigue damage."""
        # Temperature cycling stress
        temp_swing = np.abs(temperature_map - 300)  # From room temperature
        damage_rate = 1e-9 * (temp_swing / 50) ** 2  # Simplified thermal fatigue
        
        # Accumulate damage
        self.thermal_fatigue += damage_rate * duration_hours
    
    def generate_reliability_report(self, save_path: Optional[Path] = None) -> Dict:
        """Generate comprehensive reliability analysis report."""
        try:
            report = {
                "reliability_analysis": {
                    "total_devices": int(self.crossbar.rows * self.crossbar.cols),
                    "analysis_duration_hours": float(np.max(self.device_ages)),
                    "average_device_age_hours": float(np.mean(self.device_ages)),
                    "stress_cycles_applied": len(self.stress_history)
                },
                "degradation_summary": {
                    "mean_degradation_state": float(np.mean(self.degradation_states)),
                    "worst_degradation_state": float(np.min(self.degradation_states)),
                    "degradation_std": float(np.std(self.degradation_states)),
                    "devices_below_80_percent": int(np.sum(self.degradation_states < 0.8)),
                    "devices_below_50_percent": int(np.sum(self.degradation_states < 0.5))
                },
                "model_parameters": self.model.to_dict(),
                "stress_history": self.stress_history[-10:] if self.stress_history else []  # Last 10 entries
            }
            
            # Add physics-based damage breakdown if available
            if self.enable_physics:
                report["physics_based_analysis"] = {
                    "tddb_damage": {
                        "mean": float(np.mean(self.tddb_damage)),
                        "max": float(np.max(self.tddb_damage)),
                        "devices_critical": int(np.sum(self.tddb_damage > 0.8))
                    },
                    "electromigration_damage": {
                        "mean": float(np.mean(self.electromigration_damage)),
                        "max": float(np.max(self.electromigration_damage)),
                        "devices_critical": int(np.sum(self.electromigration_damage > 0.8))
                    },
                    "thermal_fatigue": {
                        "mean": float(np.mean(self.thermal_fatigue)),
                        "max": float(np.max(self.thermal_fatigue)),
                        "devices_critical": int(np.sum(self.thermal_fatigue > 0.8))
                    }
                }
            
            # Calculate reliability metrics
            healthy_devices = np.sum(self.degradation_states > 0.8)
            total_devices = self.crossbar.rows * self.crossbar.cols
            
            report["reliability_metrics"] = {
                "current_reliability": float(healthy_devices / total_devices),
                "mean_time_to_failure_hours": float(self.model.weibull_scale),
                "failure_rate_fit": float(1.0 / self.model.weibull_scale) if self.model.weibull_scale > 0 else 0.0,
                "projected_10_year_survival": float(
                    np.sum(self.degradation_states > 0.5) / total_devices
                )
            }
            
            # Save report if path provided
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                self.logger.info(f"Reliability report saved to {save_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def get_device_health_map(self) -> np.ndarray:
        """Get per-device health status map."""
        return self.degradation_states.copy()
    
    def predict_next_failure(self, confidence: float = 0.95) -> Dict[str, Union[float, Tuple[int, int]]]:
        """Predict the time and location of next device failure."""
        try:
            # Find device closest to failure
            damage_scores = 1.0 - self.degradation_states
            max_damage_idx = np.unravel_index(np.argmax(damage_scores), damage_scores.shape)
            max_damage = damage_scores[max_damage_idx]
            
            # Estimate time to failure for most vulnerable device
            remaining_health = self.degradation_states[max_damage_idx]
            current_age = self.device_ages[max_damage_idx]
            
            # Simple extrapolation based on current damage rate
            if len(self.stress_history) > 1:
                recent_damage_rate = (self.stress_history[-1]["total_damage"] - 
                                    self.stress_history[-2]["total_damage"]) / \
                                   self.stress_history[-1]["duration_hours"]
                
                if recent_damage_rate > 0:
                    time_to_failure = remaining_health / recent_damage_rate
                else:
                    time_to_failure = float('inf')
            else:
                # Use Weibull model as fallback
                time_to_failure = self.model.weibull_scale * remaining_health
            
            # Calculate confidence bounds
            confidence_multiplier = {0.90: 0.8, 0.95: 0.6, 0.99: 0.4}.get(confidence, 0.6)
            
            return {
                "next_failure_device": max_damage_idx,
                "estimated_ttf_hours": float(time_to_failure),
                "confidence_lower_bound": float(time_to_failure * confidence_multiplier),
                "confidence_upper_bound": float(time_to_failure / confidence_multiplier),
                "current_damage_level": float(max_damage),
                "device_age_hours": float(current_age)
            }
            
        except Exception as e:
            self.logger.error(f"Next failure prediction failed: {e}")
            raise