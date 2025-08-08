"""Performance benchmarking suite for memristor neural networks."""

import time
import statistics
import psutil
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np

from ..core.crossbar import CrossbarArray
from ..mapping.neural_mapper import map_to_crossbar
from ..simulator.simulator import simulate
from ..optimization.parallel_simulator import get_parallel_simulator, ParallelConfig
from ..optimization.scaling_manager import get_scaling_manager
from ..utils.logger import get_logger


@dataclass
class BenchmarkResult:
    """Result from a performance benchmark."""
    name: str
    duration_s: float
    throughput: float  # operations per second
    memory_peak_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: str = None
    metadata: Dict[str, Any] = None


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"benchmark.{name}")
    
    def run(self) -> BenchmarkResult:
        """Run the benchmark."""
        self.logger.info(f"Starting benchmark: {self.name}")
        
        # Monitor system resources
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        cpu_monitor = []
        
        def monitor_cpu():
            cpu_monitor.append(psutil.cpu_percent(interval=0.1))
        
        try:
            start_time = time.time()
            
            # Run benchmark-specific code
            throughput, metadata = self.execute()
            
            duration = time.time() - start_time
            
            # Calculate resource usage
            final_memory = process.memory_info().rss / (1024 * 1024)
            memory_peak = max(initial_memory, final_memory)
            avg_cpu = statistics.mean(cpu_monitor) if cpu_monitor else 0.0
            
            return BenchmarkResult(
                name=self.name,
                duration_s=duration,
                throughput=throughput,
                memory_peak_mb=memory_peak,
                cpu_usage_percent=avg_cpu,
                success=True,
                metadata=metadata
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Benchmark {self.name} failed: {e}")
            
            return BenchmarkResult(
                name=self.name,
                duration_s=duration,
                throughput=0.0,
                memory_peak_mb=0.0,
                cpu_usage_percent=0.0,
                success=False,
                error_message=str(e)
            )
    
    def execute(self) -> tuple[float, Dict[str, Any]]:
        """Execute benchmark-specific logic. Override in subclasses."""
        raise NotImplementedError


class SimulationSpeedBenchmark(PerformanceBenchmark):
    """Benchmark simulation speed for different model sizes."""
    
    def __init__(self, model_sizes: List[tuple[int, int]]):
        super().__init__("simulation_speed")
        self.model_sizes = model_sizes
    
    def execute(self) -> tuple[float, Dict[str, Any]]:
        """Execute simulation speed benchmark."""
        results = {}
        total_inferences = 0
        total_time = 0
        
        for input_size, output_size in self.model_sizes:
            # Create model
            model = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, 10)
            )
            
            # Map to crossbar
            crossbar = CrossbarArray(input_size, output_size, device_model="IEDM2024_TaOx")
            mapped_model = map_to_crossbar(model, crossbar)
            
            # Test data
            test_data = torch.randn(100, input_size)
            
            # Run simulation
            start_time = time.time()
            sim_results = simulate(
                mapped_model,
                test_data,
                batch_size=20,
                max_batches=5
            )
            benchmark_time = time.time() - start_time
            
            # Record results
            size_key = f"{input_size}x{output_size}"
            results[size_key] = {
                "time_s": benchmark_time,
                "inferences": sim_results.inference_count,
                "throughput": sim_results.inference_count / benchmark_time
            }
            
            total_inferences += sim_results.inference_count
            total_time += benchmark_time
        
        overall_throughput = total_inferences / total_time
        return overall_throughput, results


class ParallelismBenchmark(PerformanceBenchmark):
    """Benchmark parallel simulation performance."""
    
    def __init__(self, worker_counts: List[int]):
        super().__init__("parallelism")
        self.worker_counts = worker_counts
    
    def execute(self) -> tuple[float, Dict[str, Any]]:
        """Execute parallelism benchmark."""
        results = {}
        
        # Create test workload
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        crossbar = CrossbarArray(64, 32, device_model="IEDM2024_TaOx")
        mapped_model = map_to_crossbar(model, crossbar)
        
        test_datasets = [torch.randn(50, 64) for _ in range(20)]  # 20 tasks
        configs = [{"include_noise": True, "batch_size": 10} for _ in range(20)]
        
        best_throughput = 0
        
        for worker_count in self.worker_counts:
            config = ParallelConfig(max_workers=worker_count, use_processes=False)
            parallel_sim = get_parallel_simulator(config)
            
            start_time = time.time()
            parallel_results = parallel_sim.simulate_batch_parallel(
                [mapped_model] * 20,
                test_datasets,
                configs
            )
            parallel_time = time.time() - start_time
            
            successful_sims = sum(1 for r in parallel_results if hasattr(r, 'accuracy'))
            throughput = successful_sims / parallel_time
            
            results[f"workers_{worker_count}"] = {
                "time_s": parallel_time,
                "successful_sims": successful_sims,
                "throughput": throughput
            }
            
            best_throughput = max(best_throughput, throughput)
        
        return best_throughput, results


class MemoryEfficiencyBenchmark(PerformanceBenchmark):
    """Benchmark memory usage efficiency."""
    
    def __init__(self, data_sizes: List[int]):
        super().__init__("memory_efficiency")
        self.data_sizes = data_sizes
    
    def execute(self) -> tuple[float, Dict[str, Any]]:
        """Execute memory efficiency benchmark."""
        results = {}
        process = psutil.Process()
        
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
        crossbar = CrossbarArray(128, 64, device_model="IEDM2024_TaOx")
        mapped_model = map_to_crossbar(model, crossbar)
        
        for data_size in self.data_sizes:
            # Measure memory before
            initial_memory = process.memory_info().rss / (1024 * 1024)
            
            # Create large dataset
            test_data = torch.randn(data_size, 128)
            
            # Run simulation
            start_time = time.time()
            sim_results = simulate(
                mapped_model,
                test_data,
                batch_size=min(64, data_size // 10),
                max_batches=10
            )
            sim_time = time.time() - start_time
            
            # Measure memory after
            peak_memory = process.memory_info().rss / (1024 * 1024)
            memory_used = peak_memory - initial_memory
            
            # Memory efficiency = inferences per MB
            efficiency = sim_results.inference_count / max(memory_used, 1)
            
            results[f"data_size_{data_size}"] = {
                "memory_used_mb": memory_used,
                "inferences": sim_results.inference_count,
                "efficiency": efficiency,
                "time_s": sim_time
            }
            
            # Clean up
            del test_data
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Return average efficiency
        avg_efficiency = statistics.mean([r["efficiency"] for r in results.values()])
        return avg_efficiency, results


class ScalingBenchmark(PerformanceBenchmark):
    """Benchmark adaptive scaling performance."""
    
    def __init__(self):
        super().__init__("adaptive_scaling")
    
    def execute(self) -> tuple[float, Dict[str, Any]]:
        """Execute adaptive scaling benchmark."""
        scaling_manager = get_scaling_manager(min_workers=1, max_workers=8)
        
        # Simulate varying workloads
        results = {}
        workload_phases = [
            ("low", 2),
            ("medium", 5), 
            ("high", 15),
            ("peak", 25),
            ("cooldown", 3)
        ]
        
        scaling_actions = 0
        total_efficiency = 0
        
        for phase_name, pending_tasks in workload_phases:
            # Optimize worker count
            start_time = time.time()
            optimal_workers = scaling_manager.optimize_worker_count(pending_tasks)
            optimization_time = time.time() - start_time
            
            # Calculate efficiency metrics
            utilization = min(pending_tasks / optimal_workers, 1.0)
            efficiency = utilization / (optimal_workers / 8)  # Normalized efficiency
            
            results[phase_name] = {
                "pending_tasks": pending_tasks,
                "optimal_workers": optimal_workers,
                "utilization": utilization,
                "efficiency": efficiency,
                "optimization_time_ms": optimization_time * 1000
            }
            
            total_efficiency += efficiency
            if phase_name != "low":  # First phase won't scale
                scaling_actions += 1
            
            time.sleep(0.1)  # Simulate time between phases
        
        # Get scaling stats
        scaling_stats = scaling_manager.get_comprehensive_stats()
        results["scaling_stats"] = scaling_stats
        
        avg_efficiency = total_efficiency / len(workload_phases)
        return avg_efficiency, results


class ThroughputBenchmark(PerformanceBenchmark):
    """Benchmark maximum sustainable throughput."""
    
    def __init__(self, duration_seconds: int = 30):
        super().__init__("throughput")
        self.duration_seconds = duration_seconds
    
    def execute(self) -> tuple[float, Dict[str, Any]]:
        """Execute throughput benchmark."""
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10))
        crossbar = CrossbarArray(32, 16, device_model="IEDM2024_TaOx")
        mapped_model = map_to_crossbar(model, crossbar)
        
        total_inferences = 0
        batch_count = 0
        start_time = time.time()
        end_time = start_time + self.duration_seconds
        
        while time.time() < end_time:
            # Generate batch
            batch_data = torch.randn(32, 32)
            
            # Run simulation
            results = simulate(
                mapped_model,
                batch_data,
                batch_size=16,
                max_batches=2
            )
            
            total_inferences += results.inference_count
            batch_count += 1
        
        actual_duration = time.time() - start_time
        sustained_throughput = total_inferences / actual_duration
        
        results = {
            "total_inferences": total_inferences,
            "batch_count": batch_count,
            "actual_duration_s": actual_duration,
            "sustained_throughput": sustained_throughput,
            "avg_batch_time_ms": (actual_duration / batch_count) * 1000
        }
        
        return sustained_throughput, results


class BenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.logger = get_logger("benchmark_suite")
        self.benchmarks: List[PerformanceBenchmark] = []
    
    def add_benchmark(self, benchmark: PerformanceBenchmark):
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)
    
    def add_standard_benchmarks(self):
        """Add standard benchmark suite."""
        # Simulation speed with different model sizes
        model_sizes = [(32, 16), (64, 32), (128, 64), (256, 128)]
        self.add_benchmark(SimulationSpeedBenchmark(model_sizes))
        
        # Parallel performance
        worker_counts = [1, 2, 4, 8]
        self.add_benchmark(ParallelismBenchmark(worker_counts))
        
        # Memory efficiency
        data_sizes = [100, 500, 1000, 2000]
        self.add_benchmark(MemoryEfficiencyBenchmark(data_sizes))
        
        # Adaptive scaling
        self.add_benchmark(ScalingBenchmark())
        
        # Sustained throughput
        self.add_benchmark(ThroughputBenchmark(duration_seconds=15))
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        self.logger.info(f"Running {len(self.benchmarks)} benchmarks...")
        
        results = []
        
        for benchmark in self.benchmarks:
            result = benchmark.run()
            results.append(result)
            
            if result.success:
                self.logger.info(f"✓ {result.name}: {result.throughput:.2f} ops/s "
                               f"({result.duration_s:.2f}s, {result.memory_peak_mb:.1f}MB)")
            else:
                self.logger.error(f"✗ {result.name}: {result.error_message}")
        
        return results
    
    def generate_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if successful_results:
            avg_throughput = statistics.mean([r.throughput for r in successful_results])
            avg_duration = statistics.mean([r.duration_s for r in successful_results])
            avg_memory = statistics.mean([r.memory_peak_mb for r in successful_results])
            avg_cpu = statistics.mean([r.cpu_usage_percent for r in successful_results])
        else:
            avg_throughput = avg_duration = avg_memory = avg_cpu = 0.0
        
        report = {
            "summary": {
                "total_benchmarks": len(results),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "success_rate": len(successful_results) / len(results) if results else 0.0,
                "avg_throughput": avg_throughput,
                "avg_duration_s": avg_duration,
                "avg_memory_peak_mb": avg_memory,
                "avg_cpu_usage": avg_cpu
            },
            "detailed_results": [
                {
                    "name": r.name,
                    "success": r.success,
                    "throughput": r.throughput,
                    "duration_s": r.duration_s,
                    "memory_peak_mb": r.memory_peak_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "error_message": r.error_message,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "performance_grade": self._calculate_performance_grade(successful_results)
        }
        
        return report
    
    def _calculate_performance_grade(self, results: List[BenchmarkResult]) -> str:
        """Calculate overall performance grade."""
        if not results:
            return "F"
        
        # Performance criteria (adjust thresholds as needed)
        criteria = [
            ("throughput", lambda r: r.throughput > 100, 25),  # > 100 ops/s
            ("efficiency", lambda r: r.memory_peak_mb < 500, 25),  # < 500MB memory
            ("speed", lambda r: r.duration_s < 60, 25),  # < 60s per benchmark
            ("reliability", lambda r: r.success, 25)  # Successful execution
        ]
        
        total_score = 0
        for criterion_name, test_func, weight in criteria:
            passed_tests = sum(1 for r in results if test_func(r))
            score = (passed_tests / len(results)) * weight
            total_score += score
        
        # Grade mapping
        if total_score >= 90:
            return "A"
        elif total_score >= 80:
            return "B"  
        elif total_score >= 70:
            return "C"
        elif total_score >= 60:
            return "D"
        else:
            return "F"


def run_performance_benchmarks() -> Dict[str, Any]:
    """Run comprehensive performance benchmark suite."""
    suite = BenchmarkSuite()
    suite.add_standard_benchmarks()
    
    # Run benchmarks
    results = suite.run_all_benchmarks()
    
    # Generate report
    report = suite.generate_report(results)
    
    return report


if __name__ == "__main__":
    # Run performance benchmarks
    print("Starting comprehensive performance benchmarks...")
    print("=" * 60)
    
    report = run_performance_benchmarks()
    
    # Print summary
    summary = report["summary"]
    print(f"\nPERFORMANCE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Benchmarks: {summary['successful']}/{summary['total_benchmarks']} successful")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Avg Throughput: {summary['avg_throughput']:.2f} ops/s")
    print(f"Avg Duration: {summary['avg_duration_s']:.2f}s")
    print(f"Avg Memory: {summary['avg_memory_peak_mb']:.1f}MB")
    print(f"Performance Grade: {report['performance_grade']}")
    print("=" * 60)