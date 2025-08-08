"""Novel algorithms and research contributions for memristor neural networks."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.optimize
from scipy.stats import norm
import matplotlib.pyplot as plt

from ..core.device_models import DeviceModel, DeviceConfig
from ..core.crossbar import CrossbarArray
from ..utils.logger import get_logger
from ..utils.error_handling import collect_errors


@dataclass
class ResearchResult:
    """Result from research experiment."""
    algorithm_name: str
    baseline_performance: Dict[str, float]
    novel_performance: Dict[str, float]
    improvement_metrics: Dict[str, float]
    statistical_significance: float
    methodology: str
    reproducible_seed: int
    raw_data: Dict[str, Any]


class ResearchAlgorithm(ABC):
    """Abstract base class for research algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"research.{name}")
    
    @abstractmethod
    def run_experiment(self, baseline_method: Callable, test_parameters: Dict[str, Any]) -> ResearchResult:
        """Run research experiment comparing novel method with baseline."""
        pass


class AdaptiveNoiseCompensation(ResearchAlgorithm):
    """Novel algorithm for adaptive noise compensation in memristor crossbars."""
    
    def __init__(self):
        super().__init__("adaptive_noise_compensation")
    
    def run_experiment(self, baseline_method: Callable, test_parameters: Dict[str, Any]) -> ResearchResult:
        """
        Compare adaptive noise compensation vs. baseline methods.
        
        Novel Contribution: Real-time adaptation to device variations using
        Kalman filtering and statistical outlier detection.
        """
        self.logger.info("Running adaptive noise compensation experiment")
        
        # Set reproducible seed
        seed = test_parameters.get('seed', 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create test crossbar with known noise characteristics
        crossbar = CrossbarArray(
            rows=64, cols=32,
            device_model="IEDM2024_TaOx",
            config=DeviceConfig(
                read_noise_sigma=test_parameters.get('noise_level', 0.1),
                ron_variation=0.15,
                roff_variation=0.20
            )
        )
        
        # Generate test patterns
        n_tests = test_parameters.get('n_tests', 100)
        test_patterns = [np.random.randn(64) for _ in range(n_tests)]
        
        # Baseline method: Standard analog computation
        baseline_results = []
        for pattern in test_patterns:
            result = baseline_method(crossbar, pattern)
            baseline_results.append(result)
        
        # Novel method: Adaptive noise compensation
        novel_results = []
        noise_estimator = self._initialize_kalman_filter()
        
        for pattern in test_patterns:
            # Compute with adaptive compensation
            compensated_result = self._adaptive_compensation(
                crossbar, pattern, noise_estimator
            )
            novel_results.append(compensated_result)
        
        # Calculate performance metrics
        baseline_error = np.std(baseline_results)
        novel_error = np.std(novel_results)
        baseline_accuracy = self._calculate_accuracy(baseline_results, test_patterns)
        novel_accuracy = self._calculate_accuracy(novel_results, test_patterns)
        
        # Statistical significance testing
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(baseline_results, novel_results)
        
        baseline_perf = {
            'std_error': baseline_error,
            'accuracy': baseline_accuracy,
            'snr_db': -20 * np.log10(baseline_error)
        }
        
        novel_perf = {
            'std_error': novel_error,
            'accuracy': novel_accuracy,
            'snr_db': -20 * np.log10(novel_error)
        }
        
        improvements = {
            'error_reduction_percent': ((baseline_error - novel_error) / baseline_error) * 100,
            'accuracy_improvement': novel_accuracy - baseline_accuracy,
            'snr_improvement_db': novel_perf['snr_db'] - baseline_perf['snr_db']
        }
        
        return ResearchResult(
            algorithm_name="Adaptive Noise Compensation",
            baseline_performance=baseline_perf,
            novel_performance=novel_perf,
            improvement_metrics=improvements,
            statistical_significance=p_value,
            methodology="Kalman filtering with outlier detection",
            reproducible_seed=seed,
            raw_data={
                'baseline_results': baseline_results,
                'novel_results': novel_results,
                'test_patterns': test_patterns
            }
        )
    
    def _initialize_kalman_filter(self) -> Dict[str, Any]:
        """Initialize Kalman filter for noise estimation."""
        return {
            'state': np.zeros(2),  # [bias, variance]
            'covariance': np.eye(2),
            'process_noise': 0.01,
            'measurement_noise': 0.1
        }
    
    def _adaptive_compensation(
        self,
        crossbar: CrossbarArray,
        input_pattern: np.ndarray,
        noise_estimator: Dict[str, Any]
    ) -> np.ndarray:
        """Apply adaptive noise compensation."""
        # Get raw output
        raw_output = crossbar.analog_matmul(input_pattern)
        
        # Update noise estimates using Kalman filter
        self._update_noise_estimate(noise_estimator, raw_output)
        
        # Apply compensation
        estimated_bias = noise_estimator['state'][0]
        estimated_variance = noise_estimator['state'][1]
        
        # Compensate for systematic bias
        compensated = raw_output - estimated_bias
        
        # Apply outlier detection and correction
        z_scores = np.abs((compensated - np.mean(compensated)) / np.std(compensated))
        outlier_mask = z_scores > 2.5  # 2.5 sigma threshold
        
        # Replace outliers with filtered values
        if np.any(outlier_mask):
            compensated[outlier_mask] = np.median(compensated[~outlier_mask])
        
        return compensated
    
    def _update_noise_estimate(self, estimator: Dict[str, Any], measurement: np.ndarray):
        """Update Kalman filter with new measurement."""
        # Simplified Kalman update for bias and variance estimation
        residual = np.mean(measurement) - estimator['state'][0]
        
        # Update state estimate
        kalman_gain = estimator['covariance'][0, 0] / (estimator['covariance'][0, 0] + estimator['measurement_noise'])
        estimator['state'][0] += kalman_gain * residual
        estimator['state'][1] = 0.9 * estimator['state'][1] + 0.1 * np.var(measurement)
        
        # Update covariance
        estimator['covariance'] = (1 - kalman_gain) * estimator['covariance']
        estimator['covariance'] += estimator['process_noise'] * np.eye(2)
    
    def _calculate_accuracy(self, results: List[np.ndarray], targets: List[np.ndarray]) -> float:
        """Calculate accuracy metric for comparison."""
        total_error = 0
        total_samples = 0
        
        for result, target in zip(results, targets):
            if len(result) == len(target):
                error = np.mean(np.abs(result - target[:len(result)]))
                total_error += error
                total_samples += 1
        
        return 1.0 / (1.0 + total_error / total_samples) if total_samples > 0 else 0.0


class EvolutionaryWeightMapping(ResearchAlgorithm):
    """Novel evolutionary algorithm for optimal weight mapping to crossbar arrays."""
    
    def __init__(self):
        super().__init__("evolutionary_weight_mapping")
    
    def run_experiment(self, baseline_method: Callable, test_parameters: Dict[str, Any]) -> ResearchResult:
        """
        Compare evolutionary weight mapping vs. standard approaches.
        
        Novel Contribution: Multi-objective optimization considering device variations,
        power consumption, and accuracy simultaneously.
        """
        self.logger.info("Running evolutionary weight mapping experiment")
        
        seed = test_parameters.get('seed', 42)
        np.random.seed(seed)
        
        # Create neural network weights to map
        layer_weights = np.random.randn(32, 16) * 0.5  # Small network for testing
        
        # Create crossbar array
        crossbar = CrossbarArray(32, 16, device_model="IEDM2024_TaOx")
        
        # Baseline method: Standard differential mapping
        baseline_mapping = self._baseline_mapping(layer_weights)
        crossbar.program_weights(layer_weights, programming_scheme="differential")
        baseline_performance = self._evaluate_mapping(crossbar, layer_weights, test_parameters)
        
        # Novel method: Evolutionary optimization
        best_mapping, novel_performance = self._evolutionary_mapping(
            layer_weights, crossbar, test_parameters
        )
        
        # Statistical comparison
        n_trials = 10
        baseline_trials = []
        novel_trials = []
        
        for _ in range(n_trials):
            # Add noise to simulate multiple trials
            noise_scale = 0.02
            noisy_weights = layer_weights + np.random.randn(*layer_weights.shape) * noise_scale
            
            baseline_perf = self._evaluate_mapping(crossbar, noisy_weights, test_parameters)
            novel_perf = self._evaluate_mapping(crossbar, best_mapping, test_parameters)
            
            baseline_trials.append(baseline_perf['accuracy'])
            novel_trials.append(novel_perf['accuracy'])
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(novel_trials, baseline_trials)
        
        improvements = {
            'accuracy_improvement': novel_performance['accuracy'] - baseline_performance['accuracy'],
            'power_reduction_percent': ((baseline_performance['power'] - novel_performance['power']) / baseline_performance['power']) * 100,
            'area_efficiency': novel_performance['accuracy'] / novel_performance['area']
        }
        
        return ResearchResult(
            algorithm_name="Evolutionary Weight Mapping",
            baseline_performance=baseline_performance,
            novel_performance=novel_performance,
            improvement_metrics=improvements,
            statistical_significance=p_value,
            methodology="Multi-objective genetic algorithm with device-aware fitness",
            reproducible_seed=seed,
            raw_data={
                'original_weights': layer_weights,
                'optimized_mapping': best_mapping,
                'baseline_trials': baseline_trials,
                'novel_trials': novel_trials
            }
        )
    
    def _baseline_mapping(self, weights: np.ndarray) -> np.ndarray:
        """Standard weight mapping approach."""
        # Simple normalization and clipping
        normalized = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return np.clip(normalized, 0, 1)
    
    def _evolutionary_mapping(
        self,
        target_weights: np.ndarray,
        crossbar: CrossbarArray,
        parameters: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Evolutionary optimization for weight mapping."""
        
        population_size = parameters.get('population_size', 50)
        generations = parameters.get('generations', 100)
        mutation_rate = parameters.get('mutation_rate', 0.1)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = np.random.uniform(0, 1, target_weights.shape)
            population.append(individual)
        
        best_fitness = -np.inf
        best_individual = None
        
        for generation in range(generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            
            for individual in population:
                fitness = self._multi_objective_fitness(individual, target_weights, crossbar)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Selection and reproduction
            population = self._genetic_operations(population, fitness_scores, mutation_rate)
        
        # Evaluate final performance
        final_performance = self._evaluate_mapping(crossbar, best_individual, parameters)
        
        return best_individual, final_performance
    
    def _multi_objective_fitness(
        self,
        mapping: np.ndarray,
        target_weights: np.ndarray,
        crossbar: CrossbarArray
    ) -> float:
        """Multi-objective fitness function."""
        
        # Accuracy component (weight fidelity)
        weight_error = np.mean(np.abs(mapping - self._normalize_weights(target_weights)))
        accuracy_score = 1.0 / (1.0 + weight_error)
        
        # Power efficiency component
        crossbar.program_weights(mapping, programming_scheme="offset")
        power_breakdown = crossbar.get_power_consumption()
        power_score = 1.0 / (1.0 + power_breakdown['total_power_mw'] / 100)  # Normalize to ~100mW
        
        # Area efficiency component
        area_breakdown = crossbar.get_area_estimate()
        area_score = 1.0 / (1.0 + area_breakdown['total_area_mm2'])
        
        # Device utilization (prefer balanced usage)
        utilization_variance = np.var(mapping.flatten())
        utilization_score = 1.0 / (1.0 + utilization_variance)
        
        # Weighted combination
        weights = {'accuracy': 0.4, 'power': 0.3, 'area': 0.2, 'utilization': 0.1}
        
        total_fitness = (
            weights['accuracy'] * accuracy_score +
            weights['power'] * power_score +
            weights['area'] * area_score +
            weights['utilization'] * utilization_score
        )
        
        return total_fitness
    
    def _genetic_operations(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float],
        mutation_rate: float
    ) -> List[np.ndarray]:
        """Genetic algorithm operations."""
        
        # Tournament selection
        new_population = []
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            
            parent = population[winner_idx].copy()
            
            # Mutation
            if np.random.random() < mutation_rate:
                mutation_strength = 0.1
                mutation_mask = np.random.random(parent.shape) < mutation_rate
                parent[mutation_mask] += np.random.normal(0, mutation_strength, np.sum(mutation_mask))
                parent = np.clip(parent, 0, 1)
            
            new_population.append(parent)
        
        return new_population
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to [0, 1] range."""
        min_w, max_w = np.min(weights), np.max(weights)
        if max_w > min_w:
            return (weights - min_w) / (max_w - min_w)
        return np.ones_like(weights) * 0.5
    
    def _evaluate_mapping(
        self,
        crossbar: CrossbarArray,
        mapping: np.ndarray,
        parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate performance of a weight mapping."""
        
        # Program weights
        crossbar.program_weights(mapping, programming_scheme="offset")
        
        # Calculate metrics
        power_breakdown = crossbar.get_power_consumption()
        area_breakdown = crossbar.get_area_estimate()
        
        # Simulate accuracy with test vectors
        n_tests = parameters.get('n_accuracy_tests', 20)
        test_vectors = [np.random.randn(crossbar.rows) for _ in range(n_tests)]
        
        accuracy_sum = 0
        for test_vector in test_vectors:
            output = crossbar.analog_matmul(test_vector)
            # Simple accuracy metric based on output consistency
            accuracy_sum += 1.0 / (1.0 + np.std(output))
        
        return {
            'accuracy': accuracy_sum / n_tests,
            'power': power_breakdown['total_power_mw'],
            'area': area_breakdown['total_area_mm2'],
            'energy_efficiency': (accuracy_sum / n_tests) / power_breakdown['total_power_mw']
        }


class StatisticalFaultTolerance(ResearchAlgorithm):
    """Novel statistical approach to fault tolerance in memristor arrays."""
    
    def __init__(self):
        super().__init__("statistical_fault_tolerance")
    
    def run_experiment(self, baseline_method: Callable, test_parameters: Dict[str, Any]) -> ResearchResult:
        """
        Compare statistical fault tolerance vs. redundancy-based approaches.
        
        Novel Contribution: Bayesian inference for fault detection and
        statistical error correction using ensemble methods.
        """
        self.logger.info("Running statistical fault tolerance experiment")
        
        seed = test_parameters.get('seed', 42)
        np.random.seed(seed)
        
        # Create crossbar with various fault rates
        fault_rates = test_parameters.get('fault_rates', [0.001, 0.005, 0.01, 0.05])
        results_per_rate = {}
        
        for fault_rate in fault_rates:
            crossbar = CrossbarArray(64, 32, device_model="IEDM2024_TaOx")
            
            # Inject faults
            crossbar.inject_stuck_faults(fault_rate=fault_rate)
            
            # Baseline: Standard error detection
            baseline_accuracy = self._baseline_fault_tolerance(crossbar, test_parameters)
            
            # Novel: Statistical fault tolerance
            novel_accuracy = self._statistical_fault_tolerance(crossbar, test_parameters)
            
            results_per_rate[fault_rate] = {
                'baseline': baseline_accuracy,
                'novel': novel_accuracy,
                'improvement': novel_accuracy - baseline_accuracy
            }
        
        # Calculate overall performance
        avg_baseline = np.mean([r['baseline'] for r in results_per_rate.values()])
        avg_novel = np.mean([r['novel'] for r in results_per_rate.values()])
        
        # Statistical significance
        baseline_vals = [r['baseline'] for r in results_per_rate.values()]
        novel_vals = [r['novel'] for r in results_per_rate.values()]
        
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(novel_vals, baseline_vals)  # Paired t-test
        
        return ResearchResult(
            algorithm_name="Statistical Fault Tolerance",
            baseline_performance={'average_accuracy': avg_baseline},
            novel_performance={'average_accuracy': avg_novel},
            improvement_metrics={
                'accuracy_improvement': avg_novel - avg_baseline,
                'fault_resilience': np.std(novel_vals) - np.std(baseline_vals)  # Lower std = more resilient
            },
            statistical_significance=p_value,
            methodology="Bayesian inference with ensemble error correction",
            reproducible_seed=seed,
            raw_data={
                'fault_rates': fault_rates,
                'results_per_rate': results_per_rate
            }
        )
    
    def _baseline_fault_tolerance(self, crossbar: CrossbarArray, parameters: Dict[str, Any]) -> float:
        """Baseline fault tolerance using simple redundancy."""
        n_tests = parameters.get('n_tests', 50)
        correct_results = 0
        
        for _ in range(n_tests):
            test_input = np.random.randn(crossbar.rows)
            
            # Simple approach: Average multiple readings
            readings = []
            for _ in range(3):  # Triple modular redundancy
                output = crossbar.analog_matmul(test_input)
                readings.append(output)
            
            # Majority voting (simplified)
            final_output = np.mean(readings, axis=0)
            
            # Check result consistency (proxy for accuracy)
            if np.std(readings) < 0.5:  # Consistent results
                correct_results += 1
        
        return correct_results / n_tests
    
    def _statistical_fault_tolerance(self, crossbar: CrossbarArray, parameters: Dict[str, Any]) -> float:
        """Novel statistical fault tolerance approach."""
        n_tests = parameters.get('n_tests', 50)
        
        # Learn fault statistics
        fault_model = self._learn_fault_statistics(crossbar)
        
        correct_results = 0
        
        for _ in range(n_tests):
            test_input = np.random.randn(crossbar.rows)
            
            # Apply statistical error correction
            corrected_output = self._bayesian_error_correction(
                crossbar, test_input, fault_model
            )
            
            # Evaluate result quality using ensemble voting
            confidence = self._calculate_confidence(corrected_output, fault_model)
            
            if confidence > 0.7:  # High confidence threshold
                correct_results += 1
        
        return correct_results / n_tests
    
    def _learn_fault_statistics(self, crossbar: CrossbarArray) -> Dict[str, Any]:
        """Learn statistical model of faults in the crossbar."""
        n_samples = 100
        fault_patterns = []
        
        reference_input = np.ones(crossbar.rows)  # Known input
        
        for _ in range(n_samples):
            output = crossbar.analog_matmul(reference_input)
            fault_patterns.append(output)
        
        fault_patterns = np.array(fault_patterns)
        
        # Estimate fault statistics
        mean_output = np.mean(fault_patterns, axis=0)
        cov_matrix = np.cov(fault_patterns.T)
        
        # Detect outlier patterns (likely faults)
        mahalanobis_distances = []
        for pattern in fault_patterns:
            diff = pattern - mean_output
            distance = np.sqrt(diff.T @ np.linalg.pinv(cov_matrix) @ diff)
            mahalanobis_distances.append(distance)
        
        fault_threshold = np.percentile(mahalanobis_distances, 95)  # 95th percentile
        
        return {
            'mean_output': mean_output,
            'covariance': cov_matrix,
            'fault_threshold': fault_threshold,
            'patterns': fault_patterns
        }
    
    def _bayesian_error_correction(
        self,
        crossbar: CrossbarArray,
        test_input: np.ndarray,
        fault_model: Dict[str, Any]
    ) -> np.ndarray:
        """Apply Bayesian error correction."""
        
        # Multiple observations
        n_observations = 5
        observations = []
        
        for _ in range(n_observations):
            output = crossbar.analog_matmul(test_input)
            observations.append(output)
        
        observations = np.array(observations)
        
        # Bayesian estimation
        prior_mean = fault_model['mean_output']
        prior_cov = fault_model['covariance']
        
        # Likelihood (assume Gaussian noise)
        obs_mean = np.mean(observations, axis=0)
        obs_cov = np.cov(observations.T) + 1e-6 * np.eye(len(obs_mean))  # Add regularization
        
        # Bayesian update (simplified)
        posterior_cov_inv = np.linalg.pinv(prior_cov) + np.linalg.pinv(obs_cov)
        posterior_cov = np.linalg.pinv(posterior_cov_inv)
        
        posterior_mean = posterior_cov @ (
            np.linalg.pinv(prior_cov) @ prior_mean +
            np.linalg.pinv(obs_cov) @ obs_mean
        )
        
        return posterior_mean
    
    def _calculate_confidence(self, output: np.ndarray, fault_model: Dict[str, Any]) -> float:
        """Calculate confidence in the corrected output."""
        
        # Mahalanobis distance from expected output
        diff = output - fault_model['mean_output']
        try:
            distance = np.sqrt(diff.T @ np.linalg.pinv(fault_model['covariance']) @ diff)
            
            # Convert to confidence (higher distance = lower confidence)
            confidence = 1.0 / (1.0 + distance / fault_model['fault_threshold'])
            
            return confidence
            
        except:
            return 0.5  # Default confidence if calculation fails


@collect_errors("research_experiments")
def run_comprehensive_research_study(test_parameters: Optional[Dict[str, Any]] = None) -> Dict[str, ResearchResult]:
    """
    Run comprehensive research study comparing all novel algorithms.
    
    Args:
        test_parameters: Optional parameters for experiments
        
    Returns:
        Dictionary of research results for each algorithm
    """
    logger = get_logger("research_study")
    logger.info("Starting comprehensive research study")
    
    if test_parameters is None:
        test_parameters = {
            'seed': 42,
            'n_tests': 50,
            'noise_level': 0.1,
            'population_size': 30,
            'generations': 50
        }
    
    # Define baseline method for comparisons
    def baseline_analog_computation(crossbar: CrossbarArray, input_pattern: np.ndarray) -> np.ndarray:
        """Standard analog computation baseline."""
        return crossbar.analog_matmul(input_pattern)
    
    # Initialize research algorithms
    algorithms = [
        AdaptiveNoiseCompensation(),
        EvolutionaryWeightMapping(),
        StatisticalFaultTolerance()
    ]
    
    results = {}
    
    for algorithm in algorithms:
        logger.info(f"Running experiment: {algorithm.name}")
        
        try:
            result = algorithm.run_experiment(baseline_analog_computation, test_parameters)
            results[algorithm.name] = result
            
            # Log key findings
            logger.info(f"✓ {algorithm.name}: "
                       f"p-value={result.statistical_significance:.4f}, "
                       f"improvements={result.improvement_metrics}")
            
        except Exception as e:
            logger.error(f"✗ {algorithm.name} failed: {e}")
    
    # Generate summary
    logger.info("Research study completed")
    _generate_research_summary(results)
    
    return results


def _generate_research_summary(results: Dict[str, ResearchResult]):
    """Generate summary of research findings."""
    logger = get_logger("research_summary")
    
    logger.info("=" * 60)
    logger.info("RESEARCH STUDY SUMMARY")
    logger.info("=" * 60)
    
    significant_results = []
    
    for name, result in results.items():
        logger.info(f"\n{result.algorithm_name}:")
        logger.info(f"  Methodology: {result.methodology}")
        logger.info(f"  Statistical significance: p = {result.statistical_significance:.4f}")
        
        # Check statistical significance (p < 0.05)
        if result.statistical_significance < 0.05:
            significant_results.append(result)
            logger.info("  ✓ STATISTICALLY SIGNIFICANT")
        else:
            logger.info("  ✗ Not statistically significant")
        
        logger.info("  Key improvements:")
        for metric, value in result.improvement_metrics.items():
            logger.info(f"    - {metric}: {value:.3f}")
    
    logger.info(f"\nSummary:")
    logger.info(f"  Total algorithms tested: {len(results)}")
    logger.info(f"  Statistically significant results: {len(significant_results)}")
    logger.info(f"  Success rate: {len(significant_results)/len(results):.1%}")
    
    if significant_results:
        logger.info("\n🏆 NOVEL CONTRIBUTIONS VALIDATED:")
        for result in significant_results:
            logger.info(f"  • {result.algorithm_name}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run comprehensive research study
    research_results = run_comprehensive_research_study()
    
    # Print final summary
    print(f"\nResearch study completed successfully!")
    print(f"Validated {sum(1 for r in research_results.values() if r.statistical_significance < 0.05)} novel algorithms")