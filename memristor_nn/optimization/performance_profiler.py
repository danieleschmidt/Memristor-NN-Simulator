"""Performance profiling and optimization utilities."""

import time
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import functools
import numpy as np

from ..utils.logger import get_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    
    operation_name: str
    execution_time: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    call_count: int = 1
    timestamps: List[float] = field(default_factory=list)
    
    def add_measurement(self, exec_time: float, timestamp: Optional[float] = None) -> None:
        """Add a new measurement."""
        self.execution_time = (self.execution_time * self.call_count + exec_time) / (self.call_count + 1)
        self.call_count += 1
        
        if timestamp is None:
            timestamp = time.time()
        self.timestamps.append(timestamp)


class PerformanceProfiler:
    """Comprehensive performance profiler for memristor simulations."""
    
    def __init__(self, enable_detailed_profiling: bool = False):
        """
        Initialize performance profiler.
        
        Args:
            enable_detailed_profiling: Enable detailed cProfile profiling
        """
        self.enable_detailed = enable_detailed_profiling
        self.logger = get_logger("performance_profiler")
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.RLock()
        
        # Profiling state
        self._profiling_active = False
        self._profiler: Optional[cProfile.Profile] = None
        
        self.logger.info(f"Performance profiler initialized (detailed: {enable_detailed_profiling})")
    
    def start_profiling(self) -> None:
        """Start detailed profiling."""
        if not self.enable_detailed:
            self.logger.warning("Detailed profiling not enabled")
            return
        
        with self._lock:
            if self._profiling_active:
                return
            
            self._profiler = cProfile.Profile()
            self._profiler.enable()
            self._profiling_active = True
            
            self.logger.info("Started detailed profiling")
    
    def stop_profiling(self) -> Optional[pstats.Stats]:
        """Stop detailed profiling and return stats."""
        if not self.enable_detailed or not self._profiling_active:
            return None
        
        with self._lock:
            if self._profiler:
                self._profiler.disable()
                
                # Create stats object
                s = io.StringIO()
                stats = pstats.Stats(self._profiler, stream=s)
                
                self._profiling_active = False
                self._profiler = None
                
                self.logger.info("Stopped detailed profiling")
                return stats
        
        return None
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_cpu = self._get_cpu_usage()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            end_cpu = self._get_cpu_usage()
            end_memory = self._get_memory_usage()
            
            # Calculate resource usage
            cpu_usage = end_cpu - start_cpu
            memory_usage = end_memory - start_memory
            
            # Record metrics
            self.record_operation(
                operation_name,
                execution_time,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage
            )
    
    def record_operation(
        self,
        operation_name: str,
        execution_time: float,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
        gpu_usage: float = 0.0
    ) -> None:
        """Record performance metrics for an operation."""
        with self._lock:
            if operation_name in self.metrics:
                # Update existing metrics
                self.metrics[operation_name].add_measurement(execution_time)
                # Update resource usage (simple average)
                metrics = self.metrics[operation_name]
                metrics.cpu_usage = (metrics.cpu_usage + cpu_usage) / 2
                metrics.memory_usage = (metrics.memory_usage + memory_usage) / 2
                metrics.gpu_usage = (metrics.gpu_usage + gpu_usage) / 2
            else:
                # Create new metrics
                self.metrics[operation_name] = PerformanceMetrics(
                    operation_name=operation_name,
                    execution_time=execution_time,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    gpu_usage=gpu_usage,
                    timestamps=[time.time()]
                )
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific operation."""
        with self._lock:
            if operation_name not in self.metrics:
                return None
            
            metrics = self.metrics[operation_name]
            
            # Calculate statistics
            if len(metrics.timestamps) > 1:
                time_deltas = np.diff(metrics.timestamps)
                frequency = 1.0 / np.mean(time_deltas) if len(time_deltas) > 0 else 0.0
            else:
                frequency = 0.0
            
            return {
                "operation_name": metrics.operation_name,
                "average_execution_time": metrics.execution_time,
                "call_count": metrics.call_count,
                "total_time": metrics.execution_time * metrics.call_count,
                "average_cpu_usage": metrics.cpu_usage,
                "average_memory_usage": metrics.memory_usage,
                "average_gpu_usage": metrics.gpu_usage,
                "call_frequency": frequency
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all operations."""
        with self._lock:
            return {name: self.get_operation_stats(name) 
                   for name in self.metrics.keys()}
    
    def get_slowest_operations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest operations."""
        all_stats = self.get_all_stats()
        
        # Sort by total time
        sorted_ops = sorted(
            all_stats.values(),
            key=lambda x: x["total_time"],
            reverse=True
        )
        
        return sorted_ops[:top_n]
    
    def get_most_frequent_operations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently called operations."""
        all_stats = self.get_all_stats()
        
        # Sort by call count
        sorted_ops = sorted(
            all_stats.values(),
            key=lambda x: x["call_count"],
            reverse=True
        )
        
        return sorted_ops[:top_n]
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks."""
        all_stats = self.get_all_stats()
        
        if not all_stats:
            return {"error": "No performance data available"}
        
        # Calculate totals
        total_time = sum(stats["total_time"] for stats in all_stats.values())
        total_calls = sum(stats["call_count"] for stats in all_stats.values())
        
        # Find bottlenecks
        slowest_ops = self.get_slowest_operations(5)
        most_frequent_ops = self.get_most_frequent_operations(5)
        
        # Operations with highest time per call
        highest_avg_time = sorted(
            all_stats.values(),
            key=lambda x: x["average_execution_time"],
            reverse=True
        )[:5]
        
        return {
            "summary": {
                "total_operations": len(all_stats),
                "total_execution_time": total_time,
                "total_calls": total_calls,
                "average_time_per_call": total_time / total_calls if total_calls > 0 else 0
            },
            "bottlenecks": {
                "slowest_total": slowest_ops,
                "most_frequent": most_frequent_ops,
                "highest_average": highest_avg_time
            }
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive performance report."""
        analysis = self.analyze_bottlenecks()
        
        if "error" in analysis:
            return analysis["error"]
        
        report = []
        report.append("# Performance Profiling Report\n")
        
        # Summary
        summary = analysis["summary"]
        report.append("## Summary")
        report.append(f"- Total Operations: {summary['total_operations']}")
        report.append(f"- Total Execution Time: {summary['total_execution_time']:.3f}s")
        report.append(f"- Total Calls: {summary['total_calls']}")
        report.append(f"- Average Time per Call: {summary['average_time_per_call']:.3f}s\n")
        
        # Bottlenecks
        bottlenecks = analysis["bottlenecks"]
        
        report.append("## Slowest Operations (by total time)")
        for i, op in enumerate(bottlenecks["slowest_total"]):
            report.append(f"{i+1}. **{op['operation_name']}**: {op['total_time']:.3f}s "
                         f"({op['call_count']} calls, {op['average_execution_time']:.3f}s avg)")
        
        report.append("\n## Most Frequent Operations")
        for i, op in enumerate(bottlenecks["most_frequent"]):
            report.append(f"{i+1}. **{op['operation_name']}**: {op['call_count']} calls "
                         f"({op['total_time']:.3f}s total, {op['average_execution_time']:.3f}s avg)")
        
        report.append("\n## Highest Average Time Operations")
        for i, op in enumerate(bottlenecks["highest_average"]):
            report.append(f"{i+1}. **{op['operation_name']}**: {op['average_execution_time']:.3f}s avg "
                         f"({op['call_count']} calls, {op['total_time']:.3f}s total)")
        
        # Recommendations
        report.append("\n## Optimization Recommendations")
        
        slowest = bottlenecks["slowest_total"][0] if bottlenecks["slowest_total"] else None
        if slowest:
            report.append(f"- **Focus on '{slowest['operation_name']}'**: Consumes "
                         f"{slowest['total_time']:.1f}s ({slowest['total_time']/summary['total_execution_time']*100:.1f}% of total time)")
        
        most_frequent = bottlenecks["most_frequent"][0] if bottlenecks["most_frequent"] else None
        if most_frequent and most_frequent['call_count'] > 100:
            report.append(f"- **Optimize '{most_frequent['operation_name']}'**: Called "
                         f"{most_frequent['call_count']} times - consider caching or vectorization")
        
        highest_avg = bottlenecks["highest_average"][0] if bottlenecks["highest_average"] else None
        if highest_avg and highest_avg['average_execution_time'] > 1.0:
            report.append(f"- **Profile '{highest_avg['operation_name']}'**: High average time "
                         f"({highest_avg['average_execution_time']:.3f}s) suggests algorithmic improvements needed")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self.metrics.clear()
            self.logger.info("Cleared performance metrics")


def profile_function(profiler: Optional[PerformanceProfiler] = None):
    """Decorator to profile function execution."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _profiler = profiler or _get_global_profiler()
            
            with _profiler.profile_operation(func.__name__):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def _get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler."""
    return _get_global_profiler()


class BenchmarkSuite:
    """Benchmark suite for memristor operations."""
    
    def __init__(self):
        self.logger = get_logger("benchmark_suite")
        self.results: Dict[str, Dict[str, float]] = {}
    
    def benchmark_crossbar_operations(
        self,
        crossbar_sizes: List[tuple],
        num_iterations: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark crossbar operations across different sizes."""
        import memristor_nn as mn
        
        results = {}
        
        for rows, cols in crossbar_sizes:
            size_key = f"{rows}x{cols}"
            results[size_key] = {}
            
            self.logger.info(f"Benchmarking {size_key} crossbar")
            
            # Creation time
            create_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                crossbar = mn.CrossbarArray(rows=rows, cols=cols)
                create_times.append(time.time() - start_time)
            
            results[size_key]["creation_time"] = np.mean(create_times)
            
            # Weight programming time
            crossbar = mn.CrossbarArray(rows=rows, cols=cols)
            weights = np.random.randn(rows, cols)
            
            program_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                crossbar.program_weights(weights)
                program_times.append(time.time() - start_time)
            
            results[size_key]["programming_time"] = np.mean(program_times)
            
            # Matrix multiplication time
            input_vector = np.random.randn(rows)
            
            matmul_times = []
            for _ in range(num_iterations):
                start_time = time.time()
                output = crossbar.analog_matmul(input_vector)
                matmul_times.append(time.time() - start_time)
            
            results[size_key]["matmul_time"] = np.mean(matmul_times)
        
        self.results.update(results)
        return results
    
    def benchmark_simulation_scaling(
        self,
        model_sizes: List[int],
        batch_sizes: List[int],
        num_iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark simulation scaling with model and batch sizes."""
        import torch
        import torch.nn as nn
        import memristor_nn as mn
        
        results = {}
        
        for model_size in model_sizes:
            for batch_size in batch_sizes:
                key = f"model_{model_size}_batch_{batch_size}"
                
                self.logger.info(f"Benchmarking {key}")
                
                # Create model and crossbar
                model = nn.Linear(model_size, model_size // 2)
                crossbar = mn.CrossbarArray(rows=model_size, cols=model_size // 2)
                mapped_model = mn.map_to_crossbar(model, crossbar)
                
                # Create test data
                test_data = torch.randn(batch_size, model_size)
                
                # Benchmark simulation
                sim_times = []
                for _ in range(num_iterations):
                    start_time = time.time()
                    result = mn.simulate(mapped_model, test_data, max_batches=2)
                    sim_times.append(time.time() - start_time)
                
                results[key] = {
                    "simulation_time": np.mean(sim_times),
                    "throughput": batch_size / np.mean(sim_times),  # samples/second
                    "model_size": model_size,
                    "batch_size": batch_size
                }
        
        return results
    
    def generate_benchmark_report(self, save_path: Optional[str] = None) -> str:
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available"
        
        report = []
        report.append("# Benchmark Results\n")
        
        # Crossbar operations
        crossbar_results = {k: v for k, v in self.results.items() if 'x' in k}
        if crossbar_results:
            report.append("## Crossbar Operations\n")
            report.append("| Size | Creation (s) | Programming (s) | MatMul (s) |")
            report.append("|------|--------------|-----------------|------------|")
            
            for size, metrics in crossbar_results.items():
                report.append(f"| {size} | {metrics.get('creation_time', 0):.6f} | "
                             f"{metrics.get('programming_time', 0):.6f} | "
                             f"{metrics.get('matmul_time', 0):.6f} |")
        
        # Simulation scaling
        sim_results = {k: v for k, v in self.results.items() if 'model_' in k}
        if sim_results:
            report.append("\n## Simulation Scaling\n")
            report.append("| Configuration | Simulation Time (s) | Throughput (samples/s) |")
            report.append("|---------------|--------------------|-----------------------|")
            
            for config, metrics in sim_results.items():
                report.append(f"| {config} | {metrics['simulation_time']:.3f} | "
                             f"{metrics['throughput']:.1f} |")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text