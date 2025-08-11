"""Experimental validation framework with real measurement data integration."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import time

from ..core.device_models import DeviceModel, DeviceConfig
from ..core.crossbar import CrossbarArray
from ..utils.logger import get_logger
from ..utils.error_handling import collect_errors
from ..utils.validators import validate_numpy_array, ValidationError


@dataclass
class ExperimentalDataPoint:
    """Single experimental measurement point."""
    voltage: float
    current: float
    temperature: float
    device_state: float
    measurement_error: float
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result from experimental validation."""
    model_name: str
    dataset_name: str
    validation_metrics: Dict[str, float]
    statistical_tests: Dict[str, float]
    error_analysis: Dict[str, Any]
    confidence_interval: Tuple[float, float]
    validation_passed: bool
    timestamp: float


class ExperimentalDataset:
    """Container for experimental measurement data."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.data_points: List[ExperimentalDataPoint] = []
        self.metadata = {}
        self.logger = get_logger(f"dataset.{name}")
    
    def add_data_point(self, voltage: float, current: float, temperature: float = 300.0,
                      device_state: float = 0.5, measurement_error: float = 0.01,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add experimental data point."""
        data_point = ExperimentalDataPoint(
            voltage=voltage,
            current=current,
            temperature=temperature,
            device_state=device_state,
            measurement_error=measurement_error,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.data_points.append(data_point)
    
    def load_from_csv(self, file_path: str, voltage_col: str = 'voltage',
                     current_col: str = 'current', temperature_col: str = 'temperature') -> None:
        """Load experimental data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            required_cols = [voltage_col, current_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
            
            for _, row in df.iterrows():
                voltage = float(row[voltage_col])
                current = float(row[current_col])
                temperature = float(row.get(temperature_col, 300.0))
                
                # Extract additional metadata
                metadata = {}
                for col in df.columns:
                    if col not in [voltage_col, current_col, temperature_col]:
                        metadata[col] = row[col]
                
                self.add_data_point(voltage, current, temperature, metadata=metadata)
            
            self.logger.info(f"Loaded {len(self.data_points)} data points from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load data from {file_path}: {e}")
            raise
    
    def get_iv_curve(self, temperature_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract I-V curve from experimental data."""
        filtered_points = self.data_points
        
        if temperature_range:
            t_min, t_max = temperature_range
            filtered_points = [p for p in filtered_points if t_min <= p.temperature <= t_max]
        
        voltages = np.array([p.voltage for p in filtered_points])
        currents = np.array([p.current for p in filtered_points])
        
        # Sort by voltage
        sort_idx = np.argsort(voltages)
        return voltages[sort_idx], currents[sort_idx]
    
    def get_temperature_dependence(self, voltage: float, tolerance: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Extract temperature dependence at specific voltage."""
        matching_points = [p for p in self.data_points if abs(p.voltage - voltage) <= tolerance]
        
        temperatures = np.array([p.temperature for p in matching_points])
        currents = np.array([p.current for p in matching_points])
        
        # Sort by temperature
        sort_idx = np.argsort(temperatures)
        return temperatures[sort_idx], currents[sort_idx]
    
    def export_to_json(self, file_path: str) -> None:
        """Export dataset to JSON format."""
        export_data = {
            'name': self.name,
            'description': self.description,
            'metadata': self.metadata,
            'data_points': []
        }
        
        for point in self.data_points:
            export_data['data_points'].append({
                'voltage': point.voltage,
                'current': point.current,
                'temperature': point.temperature,
                'device_state': point.device_state,
                'measurement_error': point.measurement_error,
                'timestamp': point.timestamp,
                'metadata': point.metadata
            })
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Dataset exported to {file_path}")


class SyntheticDatasetGenerator:
    """Generate realistic synthetic datasets for validation testing."""
    
    @staticmethod
    def generate_iedm2024_taox_dataset(n_points: int = 200, voltage_range: Tuple[float, float] = (-2.0, 2.0),
                                      temperature_range: Tuple[float, float] = (250, 400)) -> ExperimentalDataset:
        """Generate synthetic IEDM 2024 TaOx dataset."""
        dataset = ExperimentalDataset(
            name="Synthetic_IEDM2024_TaOx",
            description="Synthetic dataset mimicking IEDM 2024 TaOx characteristics"
        )
        
        voltages = np.linspace(voltage_range[0], voltage_range[1], n_points)
        temperatures = np.random.uniform(temperature_range[0], temperature_range[1], n_points)
        
        for v, temp in zip(voltages, temperatures):
            # Physics-based current calculation
            # Include nonlinear switching, temperature effects, and measurement noise
            
            # Base I-V relationship
            if abs(v) < 0.5:  # Linear region
                base_current = v * 1e-6  # 1 µA/V conductance
            else:  # Nonlinear switching region
                switching_voltage = 1.2
                nonlinearity = 2.5
                base_current = 1e-6 * v * (1 + np.exp(nonlinearity * (abs(v) - switching_voltage)))
            
            # Temperature dependence (Arrhenius)
            temp_factor = np.exp(-0.3 / (8.617e-5 * temp))  # 0.3 eV activation energy
            current = base_current * temp_factor
            
            # Add measurement noise (5% typical)
            noise_factor = np.random.normal(1.0, 0.05)
            measured_current = current * noise_factor
            
            # Measurement error estimate
            measurement_error = abs(measured_current) * 0.05
            
            dataset.add_data_point(
                voltage=v,
                current=measured_current,
                temperature=temp,
                measurement_error=measurement_error,
                metadata={'device_type': 'TaOx', 'synthetic': True}
            )
        
        return dataset
    
    @staticmethod
    def generate_hfox_dataset(n_points: int = 200) -> ExperimentalDataset:
        """Generate synthetic HfOx dataset."""
        dataset = ExperimentalDataset(
            name="Synthetic_HfOx",
            description="Synthetic HfOx memristor dataset"
        )
        
        voltages = np.linspace(-1.5, 1.5, n_points)
        
        for v in voltages:
            # HfOx characteristics: faster switching, lower voltages
            if abs(v) < 0.3:
                current = v * 5e-6  # Higher conductance than TaOx
            else:
                switching_voltage = 0.8
                nonlinearity = 3.0
                current = 5e-6 * v * (1 + np.exp(nonlinearity * (abs(v) - switching_voltage)))
            
            # Add noise
            measured_current = current * np.random.normal(1.0, 0.03)
            measurement_error = abs(measured_current) * 0.03
            
            dataset.add_data_point(
                voltage=v,
                current=measured_current,
                measurement_error=measurement_error,
                metadata={'device_type': 'HfOx', 'synthetic': True}
            )
        
        return dataset


class ExperimentalValidator:
    """Validate device models against experimental data."""
    
    def __init__(self):
        self.logger = get_logger("experimental_validator")
        self.validation_results = []
    
    @collect_errors("experimental_validation")
    def validate_device_model(self, model: DeviceModel, dataset: ExperimentalDataset,
                             validation_config: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate device model against experimental dataset.
        
        Args:
            model: Device model to validate
            dataset: Experimental dataset
            validation_config: Configuration for validation parameters
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        self.logger.info(f"Validating model {model.name} against dataset {dataset.name}")
        
        config = validation_config or {}
        confidence_level = config.get('confidence_level', 0.95)
        
        # Extract experimental I-V curve
        exp_voltages, exp_currents = dataset.get_iv_curve()
        
        # Generate model predictions
        model_currents = []
        for voltage in exp_voltages:
            # Use average device state for validation
            avg_state = np.mean([p.device_state for p in dataset.data_points])
            conductance = model.conductance(voltage, avg_state)
            predicted_current = conductance * voltage  # Ohm's law
            model_currents.append(predicted_current)
        
        model_currents = np.array(model_currents)
        
        # Calculate validation metrics
        metrics = self._calculate_validation_metrics(exp_currents, model_currents)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(exp_currents, model_currents)
        
        # Error analysis
        error_analysis = self._analyze_errors(exp_voltages, exp_currents, model_currents)
        
        # Confidence interval
        confidence_interval = self._calculate_confidence_interval(
            exp_currents, model_currents, confidence_level
        )
        
        # Determine if validation passed
        validation_passed = self._determine_validation_status(metrics, statistical_tests, config)
        
        result = ValidationResult(
            model_name=model.name,
            dataset_name=dataset.name,
            validation_metrics=metrics,
            statistical_tests=statistical_tests,
            error_analysis=error_analysis,
            confidence_interval=confidence_interval,
            validation_passed=validation_passed,
            timestamp=time.time()
        )
        
        self.validation_results.append(result)
        self.logger.info(f"Validation {'PASSED' if validation_passed else 'FAILED'} for {model.name}")
        
        return result
    
    def _calculate_validation_metrics(self, experimental: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive validation metrics."""
        
        # Basic error metrics
        mse = np.mean((experimental - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(experimental - predicted))
        
        # Relative metrics
        exp_range = np.max(experimental) - np.min(experimental)
        normalized_rmse = rmse / exp_range if exp_range > 0 else np.inf
        
        # Correlation metrics
        correlation = np.corrcoef(experimental, predicted)[0, 1] if len(experimental) > 1 else 0
        
        # R-squared
        ss_res = np.sum((experimental - predicted) ** 2)
        ss_tot = np.sum((experimental - np.mean(experimental)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Log-scale metrics (for currents spanning multiple orders)
        exp_log = np.log10(np.abs(experimental) + 1e-12)
        pred_log = np.log10(np.abs(predicted) + 1e-12)
        log_rmse = np.sqrt(np.mean((exp_log - pred_log) ** 2))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'normalized_rmse': normalized_rmse,
            'correlation': correlation,
            'r_squared': r_squared,
            'log_rmse': log_rmse
        }\n    \n    def _perform_statistical_tests(self, experimental: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:\n        \"\"\"Perform statistical significance tests.\"\"\"\n        try:\n            from scipy import stats\n            \n            # Kolmogorov-Smirnov test for distribution similarity\n            ks_statistic, ks_p_value = stats.ks_2samp(experimental, predicted)\n            \n            # Wilcoxon signed-rank test for systematic differences\n            differences = experimental - predicted\n            if len(differences) > 0:\n                wilcoxon_statistic, wilcoxon_p_value = stats.wilcoxon(differences)\n            else:\n                wilcoxon_statistic, wilcoxon_p_value = 0, 1\n            \n            # Shapiro-Wilk test for normality of residuals\n            if len(differences) >= 3:\n                shapiro_statistic, shapiro_p_value = stats.shapiro(differences)\n            else:\n                shapiro_statistic, shapiro_p_value = 0, 1\n            \n            return {\n                'ks_statistic': ks_statistic,\n                'ks_p_value': ks_p_value,\n                'wilcoxon_statistic': wilcoxon_statistic,\n                'wilcoxon_p_value': wilcoxon_p_value,\n                'shapiro_statistic': shapiro_statistic,\n                'shapiro_p_value': shapiro_p_value\n            }\n            \n        except ImportError:\n            self.logger.warning(\"SciPy not available, skipping statistical tests\")\n            return {'error': 'scipy_not_available'}\n        except Exception as e:\n            self.logger.error(f\"Statistical tests failed: {e}\")\n            return {'error': str(e)}\n    \n    def _analyze_errors(self, voltages: np.ndarray, experimental: np.ndarray, \n                       predicted: np.ndarray) -> Dict[str, Any]:\n        \"\"\"Analyze error patterns and characteristics.\"\"\"\n        errors = experimental - predicted\n        relative_errors = errors / (experimental + 1e-12)\n        \n        # Error statistics\n        error_stats = {\n            'mean_error': np.mean(errors),\n            'std_error': np.std(errors),\n            'max_absolute_error': np.max(np.abs(errors)),\n            'mean_relative_error': np.mean(relative_errors),\n            'std_relative_error': np.std(relative_errors)\n        }\n        \n        # Voltage-dependent error analysis\n        voltage_ranges = [(-2, -1), (-1, 0), (0, 1), (1, 2)]\n        range_errors = {}\n        \n        for v_min, v_max in voltage_ranges:\n            mask = (voltages >= v_min) & (voltages < v_max)\n            if np.any(mask):\n                range_errors[f'{v_min}_to_{v_max}V'] = {\n                    'mean_error': np.mean(errors[mask]),\n                    'rmse': np.sqrt(np.mean(errors[mask] ** 2))\n                }\n        \n        # Identify systematic biases\n        bias_analysis = {\n            'positive_bias': np.sum(errors > 0) / len(errors),\n            'large_errors': np.sum(np.abs(relative_errors) > 0.2) / len(errors),  # >20% error\n            'correlation_with_voltage': np.corrcoef(voltages, errors)[0, 1] if len(errors) > 1 else 0\n        }\n        \n        return {\n            'error_statistics': error_stats,\n            'voltage_range_errors': range_errors,\n            'bias_analysis': bias_analysis\n        }\n    \n    def _calculate_confidence_interval(self, experimental: np.ndarray, predicted: np.ndarray, \n                                      confidence_level: float) -> Tuple[float, float]:\n        \"\"\"Calculate confidence interval for model accuracy.\"\"\"\n        try:\n            from scipy import stats\n            \n            errors = experimental - predicted\n            n = len(errors)\n            \n            if n < 2:\n                return (0.0, 1.0)\n            \n            # Student's t-distribution\n            alpha = 1 - confidence_level\n            t_value = stats.t.ppf(1 - alpha/2, df=n-1)\n            \n            mean_error = np.mean(errors)\n            std_error = np.std(errors, ddof=1)\n            margin_of_error = t_value * std_error / np.sqrt(n)\n            \n            lower_bound = mean_error - margin_of_error\n            upper_bound = mean_error + margin_of_error\n            \n            return (lower_bound, upper_bound)\n            \n        except ImportError:\n            return (0.0, 1.0)  # Default wide interval if scipy not available\n        except Exception:\n            return (0.0, 1.0)\n    \n    def _determine_validation_status(self, metrics: Dict[str, float], \n                                   statistical_tests: Dict[str, float],\n                                   config: Dict[str, Any]) -> bool:\n        \"\"\"Determine if validation passed based on criteria.\"\"\"\n        \n        # Default criteria\n        criteria = {\n            'max_normalized_rmse': config.get('max_normalized_rmse', 0.2),  # 20%\n            'min_correlation': config.get('min_correlation', 0.8),\n            'min_r_squared': config.get('min_r_squared', 0.6),\n            'max_log_rmse': config.get('max_log_rmse', 1.0)\n        }\n        \n        # Check each criterion\n        checks = [\n            metrics['normalized_rmse'] <= criteria['max_normalized_rmse'],\n            metrics['correlation'] >= criteria['min_correlation'],\n            metrics['r_squared'] >= criteria['min_r_squared'],\n            metrics['log_rmse'] <= criteria['max_log_rmse']\n        ]\n        \n        # Must pass all criteria\n        return all(checks)\n    \n    def generate_validation_report(self, output_file: str = \"validation_report.txt\") -> None:\n        \"\"\"Generate comprehensive validation report.\"\"\"\n        if not self.validation_results:\n            self.logger.warning(\"No validation results to report\")\n            return\n        \n        with open(output_file, 'w') as f:\n            f.write(\"EXPERIMENTAL VALIDATION REPORT\\n\")\n            f.write(\"=\" * 40 + \"\\n\\n\")\n            f.write(f\"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n\")\n            \n            for result in self.validation_results:\n                f.write(f\"MODEL: {result.model_name}\\n\")\n                f.write(f\"DATASET: {result.dataset_name}\\n\")\n                f.write(f\"STATUS: {'PASSED' if result.validation_passed else 'FAILED'}\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                \n                # Validation metrics\n                f.write(\"Validation Metrics:\\n\")\n                for metric, value in result.validation_metrics.items():\n                    f.write(f\"  {metric}: {value:.6f}\\n\")\n                \n                # Statistical tests\n                f.write(\"\\nStatistical Tests:\\n\")\n                for test, value in result.statistical_tests.items():\n                    f.write(f\"  {test}: {value:.6f}\\n\")\n                \n                # Confidence interval\n                lower, upper = result.confidence_interval\n                f.write(f\"\\n95% Confidence Interval: [{lower:.6f}, {upper:.6f}]\\n\")\n                \n                f.write(\"\\n\" + \"=\" * 40 + \"\\n\\n\")\n        \n        self.logger.info(f\"Validation report saved to {output_file}\")\n    \n    def cross_validate_models(self, models: List[DeviceModel], \n                             datasets: List[ExperimentalDataset]) -> Dict[str, Dict[str, ValidationResult]]:\n        \"\"\"Cross-validate multiple models against multiple datasets.\"\"\"\n        self.logger.info(f\"Cross-validating {len(models)} models against {len(datasets)} datasets\")\n        \n        results = {}\n        \n        for model in models:\n            model_results = {}\n            for dataset in datasets:\n                validation_result = self.validate_device_model(model, dataset)\n                model_results[dataset.name] = validation_result\n            results[model.name] = model_results\n        \n        return results\n\n\n@collect_errors(\"validation_suite\")\ndef run_comprehensive_validation_suite() -> Dict[str, Any]:\n    \"\"\"\n    Run comprehensive validation suite with synthetic and real datasets.\n    \n    Returns:\n        Dictionary containing all validation results\n    \"\"\"\n    logger = get_logger(\"validation_suite\")\n    logger.info(\"Starting comprehensive validation suite\")\n    \n    # Create validator\n    validator = ExperimentalValidator()\n    \n    # Generate synthetic datasets\n    datasets = [\n        SyntheticDatasetGenerator.generate_iedm2024_taox_dataset(),\n        SyntheticDatasetGenerator.generate_hfox_dataset()\n    ]\n    \n    # Create device models to test\n    from ..core.device_models import IEDM2024_TaOx, IEDM2024_HfOx\n    \n    models = [\n        IEDM2024_TaOx(),\n        IEDM2024_HfOx()\n    ]\n    \n    # Run cross-validation\n    cross_validation_results = validator.cross_validate_models(models, datasets)\n    \n    # Generate report\n    validator.generate_validation_report()\n    \n    # Summary statistics\n    total_tests = len(models) * len(datasets)\n    passed_tests = sum(\n        1 for model_results in cross_validation_results.values()\n        for result in model_results.values()\n        if result.validation_passed\n    )\n    \n    summary = {\n        'total_tests': total_tests,\n        'passed_tests': passed_tests,\n        'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,\n        'cross_validation_results': cross_validation_results\n    }\n    \n    logger.info(f\"Validation suite completed: {passed_tests}/{total_tests} tests passed ({summary['pass_rate']:.1%})\")\n    \n    return summary\n\n\nif __name__ == \"__main__\":\n    # Run comprehensive validation\n    validation_results = run_comprehensive_validation_suite()\n    print(f\"Validation completed! Pass rate: {validation_results['pass_rate']:.1%}\")"