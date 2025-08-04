"""Design space exploration for memristive neural accelerators."""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import torch
import torch.nn as nn

from ..core.crossbar import CrossbarArray
from ..core.device_models import DeviceConfig, create_device
from ..mapping.neural_mapper import map_to_crossbar
from ..simulator.simulator import simulate


class DesignSpaceExplorer:
    """Comprehensive design space exploration for memristive accelerators."""
    
    def __init__(
        self,
        model: nn.Module,
        dataset: Optional[torch.utils.data.DataLoader] = None,
        metrics: List[str] = None
    ):
        """
        Initialize design space explorer.
        
        Args:
            model: Neural network model to explore
            dataset: Dataset for accuracy evaluation
            metrics: Metrics to optimize ['power', 'latency', 'accuracy', 'area']
        """
        self.model = model
        self.dataset = dataset
        self.metrics = metrics or ['power', 'latency', 'accuracy', 'area']
        self.exploration_results = []
        
    def explore(
        self,
        param_space: Dict[str, List[Any]],
        n_samples: int = 100,
        parallel: bool = True
    ) -> pd.DataFrame:
        """
        Explore design parameter space.
        
        Args:
            param_space: Dictionary of parameters and their ranges
            n_samples: Number of design points to sample
            parallel: Whether to use parallel evaluation
            
        Returns:
            DataFrame with exploration results
        """
        # Generate design points
        design_points = self._generate_design_points(param_space, n_samples)
        
        # Evaluate design points
        if parallel and len(design_points) > 10:
            results = self._parallel_evaluation(design_points)
        else:
            results = self._sequential_evaluation(design_points)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.exploration_results = results_df
        
        return results_df
    
    def _generate_design_points(
        self,
        param_space: Dict[str, List[Any]],
        n_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate design points for exploration."""
        if n_samples <= len(list(product(*param_space.values()))):
            # Full factorial if feasible
            param_combinations = list(product(*param_space.values()))
            keys = list(param_space.keys())
            
            design_points = []
            for combo in param_combinations[:n_samples]:
                point = dict(zip(keys, combo))
                design_points.append(point)
        else:
            # Random sampling for large spaces
            design_points = []
            keys = list(param_space.keys())
            
            for _ in range(n_samples):
                point = {}
                for key in keys:
                    point[key] = np.random.choice(param_space[key])
                design_points.append(point)
        
        return design_points
    
    def _sequential_evaluation(self, design_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate design points sequentially."""
        results = []
        
        for i, point in enumerate(design_points):
            print(f"Evaluating design point {i+1}/{len(design_points)}")
            result = self._evaluate_design_point(point)
            results.append(result)
            
        return results
    
    def _parallel_evaluation(self, design_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate design points in parallel."""
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._evaluate_design_point, point) 
                      for point in design_points]
            results = [future.result() for future in futures]
            
        return results
    
    def _evaluate_design_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single design point."""
        try:
            # Create crossbar with specified parameters
            crossbar = self._create_crossbar_from_params(params)
            
            # Map model to crossbar
            mapped_model = map_to_crossbar(self.model, crossbar)
            
            # Run simulation
            if self.dataset:
                sim_results = simulate(mapped_model, self.dataset, max_batches=10)
            else:
                # Use synthetic data for quick evaluation
                input_shape = self._infer_input_shape()
                test_data = torch.randn(100, input_shape[1])
                sim_results = simulate(mapped_model, test_data, max_batches=5)
            
            # Collect metrics
            result = params.copy()
            result.update({
                'power_mw': sim_results.power_mw,
                'latency_us': sim_results.latency_us,
                'accuracy': sim_results.accuracy,
                'area_mm2': sim_results.area_mm2,
                'energy_pj': sim_results.energy_pj,
                'throughput_gops': sim_results.throughput_gops
            })
            
            return result
            
        except Exception as e:
            # Return invalid result for failed evaluations
            result = params.copy()
            result.update({
                'power_mw': float('inf'),
                'latency_us': float('inf'),
                'accuracy': 0.0,
                'area_mm2': float('inf'),
                'energy_pj': float('inf'),
                'throughput_gops': 0.0,
                'error': str(e)
            })
            return result
    
    def _create_crossbar_from_params(self, params: Dict[str, Any]) -> CrossbarArray:
        """Create crossbar array from design parameters."""
        # Extract parameters with defaults
        tile_size = params.get('tile_size', 128)
        device_tech = params.get('device_technology', 'IEDM2024_TaOx')
        adc_precision = params.get('adc_precision', 8)
        
        # Create device config
        config = DeviceConfig()
        
        # Apply peripheral optimization
        peripheral_opt = params.get('peripheral_optimization', 'baseline')
        if peripheral_opt == 'low_power':
            config.read_noise_sigma *= 0.5  # Better ADCs
        elif peripheral_opt == 'high_perf':
            config.temp_coefficient *= 0.5  # Better temperature control
        
        # Create crossbar
        crossbar = CrossbarArray(
            rows=tile_size,
            cols=tile_size,
            device_model=device_tech,
            tile_size=tile_size,
            config=config
        )
        
        # Set ADC precision
        crossbar.adc_bits = adc_precision
        
        return crossbar
    
    def _infer_input_shape(self) -> Tuple[int, int]:
        """Infer input shape from model."""
        # Simple heuristic - look for first linear layer
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                return (1, module.in_features)
        
        # Default fallback
        return (1, 784)  # MNIST-like
    
    def find_pareto_frontier(
        self,
        objectives: List[str] = None,
        minimize: List[bool] = None
    ) -> pd.DataFrame:
        """
        Find Pareto frontier for multi-objective optimization.
        
        Args:
            objectives: List of objective metrics
            minimize: Whether to minimize each objective (True) or maximize (False)
            
        Returns:
            DataFrame with Pareto optimal designs
        """
        if self.exploration_results.empty:
            raise ValueError("No exploration results available. Run explore() first.")
        
        objectives = objectives or ['power_mw', 'latency_us', 'accuracy']
        minimize = minimize or [True, True, False]  # minimize power/latency, maximize accuracy
        
        # Filter valid results
        valid_results = self.exploration_results[
            (self.exploration_results['power_mw'] != float('inf')) &
            (self.exploration_results['latency_us'] != float('inf')) &
            (self.exploration_results['accuracy'] > 0)
        ].copy()
        
        if valid_results.empty:
            return pd.DataFrame()
        
        # Apply minimization/maximization
        for obj, minimize_obj in zip(objectives, minimize):
            if not minimize_obj:
                valid_results[obj] = -valid_results[obj]  # Convert to minimization
        
        # Find Pareto frontier
        pareto_mask = np.ones(len(valid_results), dtype=bool)
        
        for i, row_i in valid_results.iterrows():
            for j, row_j in valid_results.iterrows():
                if i != j:
                    # Check if j dominates i
                    dominates = all(
                        row_j[obj] <= row_i[obj] for obj in objectives
                    ) and any(
                        row_j[obj] < row_i[obj] for obj in objectives
                    )
                    
                    if dominates:
                        pareto_mask[valid_results.index.get_loc(i)] = False
                        break
        
        pareto_frontier = valid_results[pareto_mask].copy()
        
        # Restore original values for maximization objectives
        for obj, minimize_obj in zip(objectives, minimize):
            if not minimize_obj:
                pareto_frontier[obj] = -pareto_frontier[obj]
        
        return pareto_frontier.sort_values(objectives[0])
    
    def plot_pareto_2d(
        self,
        x_metric: str = 'power_mw',
        y_metric: str = 'latency_us',
        color_metric: str = 'accuracy',
        save_path: Optional[str] = None
    ) -> None:
        """Plot 2D Pareto frontier."""
        if self.exploration_results.empty:
            raise ValueError("No exploration results available.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all points
        scatter = ax.scatter(
            self.exploration_results[x_metric],
            self.exploration_results[y_metric],
            c=self.exploration_results[color_metric],
            alpha=0.6,
            cmap='viridis'
        )
        
        # Plot Pareto frontier
        pareto_df = self.find_pareto_frontier([x_metric, y_metric, color_metric])
        if not pareto_df.empty:
            ax.scatter(
                pareto_df[x_metric],
                pareto_df[y_metric],
                c='red',
                marker='x',
                s=100,
                label='Pareto Frontier'
            )
        
        ax.set_xlabel(x_metric.replace('_', ' ').title())
        ax.set_ylabel(y_metric.replace('_', ' ').title())
        ax.set_title(f'Design Space Exploration: {x_metric} vs {y_metric}')
        ax.legend()
        
        plt.colorbar(scatter, label=color_metric.replace('_', ' ').title())
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pareto_3d(
        self,
        x: str = 'power_mw',
        y: str = 'latency_us',
        z: str = 'accuracy',
        color: str = 'area_mm2',
        save_path: Optional[str] = None
    ) -> None:
        """Plot 3D Pareto frontier visualization."""
        if self.exploration_results.empty:
            raise ValueError("No exploration results available.")
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot all points
        scatter = ax.scatter(
            self.exploration_results[x],
            self.exploration_results[y],
            self.exploration_results[z],
            c=self.exploration_results[color],
            alpha=0.6,
            cmap='plasma'
        )
        
        # Plot Pareto frontier
        pareto_df = self.find_pareto_frontier([x, y, z])
        if not pareto_df.empty:
            ax.scatter(
                pareto_df[x],
                pareto_df[y],
                pareto_df[z],
                c='red',
                marker='x',
                s=100,
                label='Pareto Frontier'
            )
        
        ax.set_xlabel(x.replace('_', ' ').title())
        ax.set_ylabel(y.replace('_', ' ').title())
        ax.set_zlabel(z.replace('_', ' ').title())
        ax.set_title(f'3D Design Space: {x} vs {y} vs {z}')
        ax.legend()
        
        plt.colorbar(scatter, label=color.replace('_', ' ').title())
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive exploration report."""
        if self.exploration_results.empty:
            return "No exploration results available."
        
        report = []
        report.append("# Design Space Exploration Report\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        valid_results = self.exploration_results[
            self.exploration_results['power_mw'] != float('inf')
        ]
        
        if not valid_results.empty:
            for metric in ['power_mw', 'latency_us', 'accuracy', 'area_mm2']:
                if metric in valid_results.columns:
                    stats = valid_results[metric].describe()
                    report.append(f"### {metric.replace('_', ' ').title()}")
                    report.append(f"- Mean: {stats['mean']:.3f}")
                    report.append(f"- Std: {stats['std']:.3f}")
                    report.append(f"- Min: {stats['min']:.3f}")
                    report.append(f"- Max: {stats['max']:.3f}\n")
        
        # Pareto frontier analysis
        report.append("## Pareto Frontier Analysis\n")
        pareto_df = self.find_pareto_frontier()
        
        if not pareto_df.empty:
            report.append(f"Found {len(pareto_df)} Pareto optimal designs:\n")
            for idx, row in pareto_df.iterrows():
                report.append(f"**Design {idx}:**")
                report.append(f"- Power: {row['power_mw']:.2f} mW")
                report.append(f"- Latency: {row['latency_us']:.2f} μs")
                report.append(f"- Accuracy: {row['accuracy']:.3f}")
                report.append(f"- Area: {row['area_mm2']:.3f} mm²\n")
        
        # Best designs per metric
        report.append("## Best Designs Per Metric\n")
        if not valid_results.empty:
            best_power = valid_results.loc[valid_results['power_mw'].idxmin()]
            best_latency = valid_results.loc[valid_results['latency_us'].idxmin()]
            best_accuracy = valid_results.loc[valid_results['accuracy'].idxmax()]
            
            report.append(f"**Lowest Power:** {best_power['power_mw']:.2f} mW")
            report.append(f"**Lowest Latency:** {best_latency['latency_us']:.2f} μs")
            report.append(f"**Highest Accuracy:** {best_accuracy['accuracy']:.3f}\n")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text