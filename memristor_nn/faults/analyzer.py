"""Fault injection and reliability analysis for memristive accelerators."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from ..mapping.neural_mapper import MappedModel
from ..simulator.simulator import simulate


class FaultType(Enum):
    """Types of faults that can be injected."""
    STUCK_AT_ON = "stuck_at_on"
    STUCK_AT_OFF = "stuck_at_off"
    DRIFT = "drift"
    READ_NOISE = "read_noise"
    WRITE_FAILURE = "write_failure"


@dataclass
class FaultInjectionResult:
    """Results from fault injection experiment."""
    
    fault_type: FaultType
    fault_rate: float
    accuracy_degradation: float
    power_increase: float
    latency_increase: float
    functional_devices: int
    total_devices: int
    trial_id: int


class FaultAnalyzer:
    """Comprehensive fault injection and reliability analysis."""
    
    def __init__(self, mapped_model: MappedModel):
        """
        Initialize fault analyzer.
        
        Args:
            mapped_model: Neural network mapped to crossbar arrays
        """
        self.mapped_model = mapped_model
        self.baseline_performance = None
        self.fault_results = []
        
    def inject_faults(
        self,
        fault_types: List[str],
        fault_rates: np.ndarray,
        n_trials: int = 10,
        test_data: Optional[np.ndarray] = None
    ) -> List[FaultInjectionResult]:
        """
        Run comprehensive fault injection campaign.
        
        Args:
            fault_types: List of fault types to inject
            fault_rates: Array of fault rates to test
            n_trials: Number of Monte Carlo trials per configuration
            test_data: Test dataset for evaluation
            
        Returns:
            List of fault injection results
        """
        # Establish baseline performance
        if self.baseline_performance is None:
            self.baseline_performance = self._measure_baseline_performance(test_data)
        
        results = []
        
        for fault_type_str in fault_types:
            fault_type = FaultType(fault_type_str)
            print(f"Injecting {fault_type.value} faults...")
            
            for fault_rate in fault_rates:
                print(f"  Fault rate: {fault_rate:.1e}")
                
                for trial in range(n_trials):
                    result = self._run_fault_injection_trial(
                        fault_type, fault_rate, trial, test_data
                    )
                    results.append(result)
        
        self.fault_results = results
        return results
    
    def _measure_baseline_performance(self, test_data: Optional[np.ndarray]) -> Dict[str, float]:
        """Measure baseline performance without faults.""" 
        # Create clean copy of model
        clean_model = self._create_clean_model_copy()
        
        if test_data is not None:
            sim_results = simulate(clean_model, test_data, max_batches=5)
        else:
            # Use synthetic data
            input_shape = (100, 784)  # Default MNIST-like
            synthetic_data = np.random.randn(*input_shape)
            sim_results = simulate(clean_model, synthetic_data, max_batches=5)
        
        return {
            "accuracy": sim_results.accuracy,
            "power_mw": sim_results.power_mw,
            "latency_us": sim_results.latency_us
        }
    
    def _create_clean_model_copy(self) -> MappedModel:
        """Create a clean copy of the mapped model."""
        # For simplicity, return the original model
        # In practice, would create deep copy
        return self.mapped_model
    
    def _run_fault_injection_trial(
        self,
        fault_type: FaultType,
        fault_rate: float,
        trial_id: int,
        test_data: Optional[np.ndarray]
    ) -> FaultInjectionResult:
        """Run a single fault injection trial."""
        # Create model copy for fault injection
        faulty_model = self._create_clean_model_copy()
        
        # Inject faults
        total_devices, functional_devices = self._inject_fault_type(
            faulty_model, fault_type, fault_rate
        )
        
        # Measure performance with faults
        try:
            if test_data is not None:
                sim_results = simulate(faulty_model, test_data, max_batches=5)
            else:
                input_shape = (100, 784)
                synthetic_data = np.random.randn(*input_shape)
                sim_results = simulate(faulty_model, synthetic_data, max_batches=5)
            
            # Calculate degradation
            accuracy_degradation = (
                self.baseline_performance["accuracy"] - sim_results.accuracy
            ) / self.baseline_performance["accuracy"]
            
            power_increase = (
                sim_results.power_mw - self.baseline_performance["power_mw"]
            ) / self.baseline_performance["power_mw"]
            
            latency_increase = (
                sim_results.latency_us - self.baseline_performance["latency_us"]
            ) / self.baseline_performance["latency_us"]
            
        except Exception as e:
            # If simulation fails, assume complete failure
            accuracy_degradation = 1.0  # 100% degradation
            power_increase = 0.0
            latency_increase = float('inf')
        
        return FaultInjectionResult(
            fault_type=fault_type,
            fault_rate=fault_rate,
            accuracy_degradation=accuracy_degradation,
            power_increase=power_increase,
            latency_increase=latency_increase,
            functional_devices=functional_devices,
            total_devices=total_devices,
            trial_id=trial_id
        )
    
    def _inject_fault_type(
        self,
        model: MappedModel,
        fault_type: FaultType,
        fault_rate: float
    ) -> Tuple[int, int]:
        """Inject specific fault type into model."""
        total_devices = 0
        functional_devices = 0
        
        for layer in model.mapped_layers:
            for crossbar in layer.crossbars:
                layer_total = crossbar.rows * crossbar.cols
                total_devices += layer_total
                
                if fault_type == FaultType.STUCK_AT_ON:
                    crossbar.inject_stuck_faults(fault_rate)
                    fault_mask = np.random.random((crossbar.rows, crossbar.cols)) < fault_rate
                    functional_devices += layer_total - np.sum(fault_mask)
                    
                elif fault_type == FaultType.STUCK_AT_OFF:
                    crossbar.inject_stuck_faults(fault_rate)
                    fault_mask = np.random.random((crossbar.rows, crossbar.cols)) < fault_rate  
                    functional_devices += layer_total - np.sum(fault_mask)
                    
                elif fault_type == FaultType.DRIFT:
                    # Apply temporal drift
                    crossbar.apply_drift(time_hours=fault_rate * 8760)  # Scale by fault rate
                    functional_devices += layer_total  # All devices still functional
                    
                elif fault_type == FaultType.READ_NOISE:
                    # Increase read noise
                    crossbar.config.read_noise_sigma *= (1 + fault_rate)
                    functional_devices += layer_total
                    
                elif fault_type == FaultType.WRITE_FAILURE:
                    # Simulate write failures as stuck devices
                    crossbar.inject_stuck_faults(fault_rate)
                    fault_mask = np.random.random((crossbar.rows, crossbar.cols)) < fault_rate
                    functional_devices += layer_total - np.sum(fault_mask)
        
        return total_devices, functional_devices
    
    def plot_reliability_curves(
        self,
        fault_results: Optional[List[FaultInjectionResult]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot reliability curves for different fault types."""
        results = fault_results or self.fault_results
        
        if not results:
            raise ValueError("No fault injection results available.")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([
            {
                'fault_type': r.fault_type.value,
                'fault_rate': r.fault_rate,
                'accuracy_degradation': r.accuracy_degradation,
                'trial_id': r.trial_id
            }
            for r in results
        ])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        fault_types = df['fault_type'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(fault_types)))
        
        # Plot 1: Accuracy vs Fault Rate
        ax = axes[0]
        for fault_type, color in zip(fault_types, colors):
            fault_data = df[df['fault_type'] == fault_type]
            
            # Calculate mean and std for each fault rate
            grouped = fault_data.groupby('fault_rate')['accuracy_degradation']
            means = grouped.mean()
            stds = grouped.std()
            
            ax.errorbar(means.index, means.values, yerr=stds.values, 
                       label=fault_type.replace('_', ' ').title(), 
                       color=color, marker='o')
        
        ax.set_xlabel('Fault Rate')
        ax.set_ylabel('Accuracy Degradation')
        ax.set_title('Accuracy Degradation vs Fault Rate')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Survival Function
        ax = axes[1]
        for fault_type, color in zip(fault_types, colors):
            fault_data = df[df['fault_type'] == fault_type]
            
            # Calculate survival probability (accuracy degradation < 10%)
            grouped = fault_data.groupby('fault_rate')
            survival_prob = grouped.apply(lambda x: np.mean(x['accuracy_degradation'] < 0.1))
            
            ax.plot(survival_prob.index, survival_prob.values, 
                   label=fault_type.replace('_', ' ').title(), 
                   color=color, marker='s')
        
        ax.set_xlabel('Fault Rate')
        ax.set_ylabel('Survival Probability')
        ax.set_title('System Survival Function')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Fault Distribution
        ax = axes[2]
        accuracy_bins = np.linspace(0, 1, 20)
        
        for fault_type, color in zip(fault_types, colors):
            fault_data = df[df['fault_type'] == fault_type]
            ax.hist(fault_data['accuracy_degradation'], bins=accuracy_bins, 
                   alpha=0.6, label=fault_type.replace('_', ' ').title(), 
                   color=color, density=True)
        
        ax.set_xlabel('Accuracy Degradation')
        ax.set_ylabel('Probability Density')
        ax.set_title('Distribution of Accuracy Degradation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Critical Fault Rate Analysis
        ax = axes[3]
        critical_rates = []
        
        for fault_type in fault_types:
            fault_data = df[df['fault_type'] == fault_type]
            grouped = fault_data.groupby('fault_rate')['accuracy_degradation']
            means = grouped.mean()
            
            # Find fault rate where accuracy degradation exceeds 10%
            critical_idx = np.where(means.values > 0.1)[0]
            if len(critical_idx) > 0:
                critical_rate = means.index[critical_idx[0]]
                critical_rates.append(critical_rate)
            else:
                critical_rates.append(float('inf'))
        
        bars = ax.bar(range(len(fault_types)), critical_rates, color=colors)
        ax.set_xticks(range(len(fault_types)))
        ax.set_xticklabels([ft.replace('_', ' ').title() for ft in fault_types], rotation=45)
        ax.set_ylabel('Critical Fault Rate')
        ax.set_title('Critical Fault Rates (10% Accuracy Loss)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_mtbf(
        self,
        operating_conditions: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate Mean Time Between Failures (MTBF) for different fault types.
        
        Args:
            operating_conditions: Environmental conditions (temperature, voltage, etc.)
            
        Returns:
            MTBF estimates for each fault type in hours
        """
        conditions = operating_conditions or {
            "temperature_k": 300.0,
            "voltage_v": 1.0,
            "stress_factor": 1.0
        }
        
        # Empirical failure rate models (simplified)
        base_failure_rates = {
            FaultType.STUCK_AT_ON: 1e-9,    # failures per device per hour
            FaultType.STUCK_AT_OFF: 1e-9,
            FaultType.DRIFT: 1e-8,
            FaultType.READ_NOISE: 1e-10,
            FaultType.WRITE_FAILURE: 1e-9
        }
        
        # Arrhenius model for temperature acceleration
        activation_energy = 0.7  # eV
        boltzmann = 8.617e-5     # eV/K
        temp_factor = np.exp(activation_energy / boltzmann * (1/300 - 1/conditions["temperature_k"]))
        
        # Voltage acceleration (power law)
        voltage_factor = (conditions["voltage_v"] / 1.0) ** 2
        
        mtbf_estimates = {}
        hw_stats = self.mapped_model.get_hardware_stats()
        total_devices = hw_stats["total_devices"]
        
        for fault_type, base_rate in base_failure_rates.items():
            # Accelerated failure rate
            accelerated_rate = base_rate * temp_factor * voltage_factor * conditions["stress_factor"]
            
            # System MTBF (considering all devices)
            system_failure_rate = total_devices * accelerated_rate
            mtbf_hours = 1.0 / system_failure_rate if system_failure_rate > 0 else float('inf')
            
            mtbf_estimates[fault_type.value] = mtbf_hours
        
        return mtbf_estimates
    
    def generate_reliability_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive reliability analysis report."""
        if not self.fault_results:
            return "No fault injection results available. Run inject_faults() first."
        
        report = []
        report.append("# Reliability Analysis Report\n")
        
        # Summary statistics
        df = pd.DataFrame([
            {
                'fault_type': r.fault_type.value,
                'fault_rate': r.fault_rate,
                'accuracy_degradation': r.accuracy_degradation
            }
            for r in self.fault_results
        ])
        
        report.append("## Fault Tolerance Summary\n")
        
        for fault_type in df['fault_type'].unique():
            fault_data = df[df['fault_type'] == fault_type]
            
            # Find critical fault rate (10% accuracy loss)
            grouped = fault_data.groupby('fault_rate')['accuracy_degradation']
            means = grouped.mean()
            critical_idx = np.where(means.values > 0.1)[0]
            
            if len(critical_idx) > 0:
                critical_rate = means.index[critical_idx[0]]
                report.append(f"**{fault_type.replace('_', ' ').title()}:**")
                report.append(f"- Critical fault rate: {critical_rate:.2e}")
                report.append(f"- Fault tolerance: {'High' if critical_rate > 1e-3 else 'Medium' if critical_rate > 1e-5 else 'Low'}")
                report.append("")
        
        # MTBF Analysis
        report.append("## Mean Time Between Failures (MTBF)\n")
        mtbf_estimates = self.calculate_mtbf()
        
        for fault_type, mtbf in mtbf_estimates.items():
            if mtbf != float('inf'):
                years = mtbf / 8760
                report.append(f"**{fault_type.replace('_', ' ').title()}:** {years:.1f} years")
            else:
                report.append(f"**{fault_type.replace('_', ' ').title()}:** >1000 years")
        
        report.append("")
        
        # Recommendations
        report.append("## Reliability Recommendations\n")
        
        most_critical = min(mtbf_estimates.items(), key=lambda x: x[1])
        report.append(f"**Most Critical:** {most_critical[0].replace('_', ' ').title()}")
        
        if most_critical[1] < 8760:  # Less than 1 year
            report.append("- **Action Required:** Implement error correction or redundancy")
        elif most_critical[1] < 87600:  # Less than 10 years
            report.append("- **Recommendation:** Consider error detection mechanisms")
        else:
            report.append("- **Status:** Acceptable reliability for most applications")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text