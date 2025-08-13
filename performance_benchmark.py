#!/usr/bin/env python3
"""
Performance benchmarking suite for memristor neural network simulator.
Tests scaling, optimization, and performance under various conditions.
"""

import sys
sys.path.insert(0, '/root/repo')

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any

from memristor_nn.core.device_models import DeviceModel, DeviceConfig
from memristor_nn.optimization.cache_manager import CacheManager
from memristor_nn.optimization.memory_optimizer import MemoryOptimizer
from memristor_nn.optimization.performance_profiler import PerformanceProfiler
from memristor_nn.utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)

class BenchmarkSuite:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.results = {}
        self.cache_manager = CacheManager(max_size=1000, ttl_seconds=3600)
        self.memory_optimizer = MemoryOptimizer()
        self.profiler = PerformanceProfiler()
    
    def benchmark_matrix_operations(self):
        """Benchmark large-scale matrix operations."""
        print("\n=== Matrix Operations Benchmark ===")
        
        sizes = [128, 256, 512, 1024, 2048]
        results = {}
        
        for size in sizes:
            with PerformanceLogger(f"matrix_ops_{size}x{size}") as perf:
                # Generate random matrices
                A = np.random.randn(size, size).astype(np.float32)
                B = np.random.randn(size, size).astype(np.float32)
                
                # Matrix multiplication
                start_time = time.time()
                C = np.dot(A, B)
                end_time = time.time()
                
                # Calculate GFLOPS
                flops = 2 * size**3  # Approximate FLOPs for matrix multiplication
                gflops = flops / (end_time - start_time) / 1e9
                
                results[size] = {
                    'time_s': end_time - start_time,
                    'gflops': gflops,
                    'memory_mb': (A.nbytes + B.nbytes + C.nbytes) / 1024 / 1024
                }
                
                print(f"  {size}x{size}: {end_time - start_time:.3f}s, {gflops:.2f} GFLOPS")
        
        self.results['matrix_operations'] = results
        return results
    
    def benchmark_parallel_execution(self):
        """Benchmark parallel processing capabilities."""
        print("\n=== Parallel Execution Benchmark ===")
        
        def compute_intensive_task(n: int) -> float:
            """CPU-intensive task for benchmarking."""
            result = 0.0
            for i in range(n):
                result += np.sin(i) * np.cos(i)
            return result
        
        task_sizes = [100000, 200000, 500000]
        num_workers_list = [1, 2, 4, mp.cpu_count()]
        
        results = {}
        
        for task_size in task_sizes:
            task_results = {}
            
            # Sequential execution baseline
            start_time = time.time()
            sequential_result = sum(compute_intensive_task(task_size) for _ in range(8))
            sequential_time = time.time() - start_time
            
            task_results['sequential'] = {
                'time_s': sequential_time,
                'speedup': 1.0,
                'efficiency': 1.0
            }
            
            print(f"  Task size {task_size}:")
            print(f"    Sequential: {sequential_time:.3f}s")
            
            # Parallel execution with different worker counts
            for num_workers in num_workers_list:
                if num_workers == 1:
                    continue  # Skip single worker as it's essentially sequential
                
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(compute_intensive_task, task_size) for _ in range(8)]
                    parallel_result = sum(f.result() for f in futures)
                parallel_time = time.time() - start_time
                
                speedup = sequential_time / parallel_time
                efficiency = speedup / num_workers
                
                task_results[f'workers_{num_workers}'] = {
                    'time_s': parallel_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
                
                print(f"    {num_workers} workers: {parallel_time:.3f}s, {speedup:.2f}x speedup, {efficiency:.2f} efficiency")
            
            results[task_size] = task_results
        
        self.results['parallel_execution'] = results
        return results
    
    def benchmark_memory_optimization(self):
        """Benchmark memory optimization strategies."""
        print("\n=== Memory Optimization Benchmark ===")
        
        import psutil
        process = psutil.Process()
        
        results = {}
        
        # Test different array sizes and optimization strategies
        array_sizes = [1024, 2048, 4096, 8192]
        
        for size in array_sizes:
            test_results = {}
            
            # Baseline: No optimization
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            arrays = []
            start_time = time.time()
            for _ in range(10):
                arr = np.random.randn(size, size).astype(np.float64)
                arrays.append(arr)
            creation_time = time.time() - start_time
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            # Clean up
            del arrays
            
            test_results['unoptimized'] = {
                'creation_time_s': creation_time,
                'memory_used_mb': memory_used,
                'memory_per_array_mb': memory_used / 10
            }
            
            # Optimized: Use float32 instead of float64
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            arrays = []
            start_time = time.time()
            for _ in range(10):
                arr = np.random.randn(size, size).astype(np.float32)
                arrays.append(arr)
            creation_time = time.time() - start_time
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_used = peak_memory - initial_memory
            
            del arrays
            
            test_results['float32_optimized'] = {
                'creation_time_s': creation_time,
                'memory_used_mb': memory_used,
                'memory_per_array_mb': memory_used / 10
            }
            
            # Calculate optimization benefits
            memory_reduction = (test_results['unoptimized']['memory_used_mb'] - 
                              test_results['float32_optimized']['memory_used_mb'])
            reduction_percent = (memory_reduction / test_results['unoptimized']['memory_used_mb']) * 100
            
            test_results['optimization'] = {
                'memory_reduction_mb': memory_reduction,
                'reduction_percent': reduction_percent
            }
            
            results[size] = test_results
            
            print(f"  Array size {size}x{size}:")
            print(f"    Unoptimized: {test_results['unoptimized']['memory_used_mb']:.1f} MB")
            print(f"    Optimized:   {test_results['float32_optimized']['memory_used_mb']:.1f} MB")
            print(f"    Reduction:   {memory_reduction:.1f} MB ({reduction_percent:.1f}%)")
        
        self.results['memory_optimization'] = results
        return results
    
    def benchmark_cache_performance(self):
        """Benchmark caching system performance."""
        print("\n=== Cache Performance Benchmark ===")
        
        def expensive_computation(x: float, y: float) -> float:
            """Simulate expensive computation."""
            time.sleep(0.001)  # Simulate 1ms computation
            return np.sin(x) * np.cos(y) + np.exp(-x*y)
        
        # Test cache with different hit ratios
        cache = CacheManager(max_size=100, ttl_seconds=60)
        
        # Generate test data
        test_params = [(np.random.random(), np.random.random()) for _ in range(50)]
        
        results = {}
        
        # Benchmark without cache
        start_time = time.time()
        uncached_results = []
        for x, y in test_params * 4:  # Repeat 4 times to simulate cache hits
            result = expensive_computation(x, y)
            uncached_results.append(result)
        uncached_time = time.time() - start_time
        
        # Benchmark with cache
        @cache.cached
        def cached_computation(x: float, y: float) -> float:
            return expensive_computation(x, y)
        
        start_time = time.time()
        cached_results = []
        for x, y in test_params * 4:  # Repeat 4 times to get cache hits
            result = cached_computation(x, y)
            cached_results.append(result)
        cached_time = time.time() - start_time
        
        # Calculate performance metrics
        speedup = uncached_time / cached_time
        cache_hit_ratio = 0.75  # 75% hit ratio (3 out of 4 repeats hit cache)
        
        results = {
            'uncached_time_s': uncached_time,
            'cached_time_s': cached_time,
            'speedup': speedup,
            'cache_hit_ratio': cache_hit_ratio,
            'cache_size': len(cache._cache)
        }
        
        print(f"  Uncached: {uncached_time:.3f}s")
        print(f"  Cached:   {cached_time:.3f}s")
        print(f"  Speedup:  {speedup:.2f}x")
        print(f"  Hit ratio: {cache_hit_ratio:.1%}")
        
        self.results['cache_performance'] = results
        return results
    
    def benchmark_scaling_limits(self):
        """Test scaling limits and performance degradation."""
        print("\n=== Scaling Limits Benchmark ===")
        
        results = {}
        
        # Test crossbar size limits
        crossbar_sizes = [64, 128, 256, 512, 1024]
        
        for size in crossbar_sizes:
            try:
                with PerformanceLogger(f"crossbar_{size}x{size}") as perf:
                    # Simulate crossbar operations
                    conductance_matrix = np.random.uniform(1e-6, 1e-4, (size, size))
                    input_vector = np.random.randn(size)
                    
                    start_time = time.time()
                    # Matrix-vector multiplication (analog computation)
                    output = np.dot(conductance_matrix.T, input_vector)
                    # Add noise
                    noise = np.random.normal(0, 0.01 * np.abs(output))
                    noisy_output = output + noise
                    computation_time = time.time() - start_time
                    
                    # Memory usage estimation
                    memory_mb = (conductance_matrix.nbytes + input_vector.nbytes + 
                               output.nbytes) / 1024 / 1024
                    
                    results[size] = {
                        'computation_time_s': computation_time,
                        'memory_mb': memory_mb,
                        'throughput_ops_per_sec': size**2 / computation_time,
                        'success': True
                    }
                    
                    print(f"  {size}x{size}: {computation_time:.4f}s, {memory_mb:.1f} MB")
                    
            except Exception as e:
                results[size] = {
                    'computation_time_s': float('inf'),
                    'memory_mb': float('inf'),
                    'throughput_ops_per_sec': 0,
                    'success': False,
                    'error': str(e)
                }
                print(f"  {size}x{size}: FAILED - {e}")
        
        self.results['scaling_limits'] = results
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 MEMRISTOR-NN PERFORMANCE BENCHMARK SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.benchmark_matrix_operations()
        self.benchmark_parallel_execution()
        self.benchmark_memory_optimization()
        self.benchmark_cache_performance()
        self.benchmark_scaling_limits()
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'total_benchmark_time_s': total_time,
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'python_version': sys.version,
            },
            'benchmarks': self.results
        }
        
        print(f"\n🎯 BENCHMARK COMPLETE: {total_time:.2f}s total")
        
        # Performance analysis
        self._analyze_performance()
        
        return summary
    
    def _analyze_performance(self):
        """Analyze benchmark results and provide recommendations."""
        print("\n📊 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Matrix operations analysis
        if 'matrix_operations' in self.results:
            matrix_results = self.results['matrix_operations']
            best_gflops = max(r['gflops'] for r in matrix_results.values())
            print(f"🔥 Peak performance: {best_gflops:.2f} GFLOPS")
        
        # Parallel execution analysis
        if 'parallel_execution' in self.results:
            parallel_results = self.results['parallel_execution']
            max_speedup = 0
            for task_results in parallel_results.values():
                for config, metrics in task_results.items():
                    if 'speedup' in metrics and metrics['speedup'] > max_speedup:
                        max_speedup = metrics['speedup']
            print(f"⚡ Maximum parallel speedup: {max_speedup:.2f}x")
        
        # Memory optimization analysis
        if 'memory_optimization' in self.results:
            memory_results = self.results['memory_optimization']
            total_reduction = sum(r['optimization']['reduction_percent'] 
                                for r in memory_results.values()) / len(memory_results)
            print(f"💾 Average memory reduction: {total_reduction:.1f}%")
        
        # Cache performance analysis
        if 'cache_performance' in self.results:
            cache_results = self.results['cache_performance']
            cache_speedup = cache_results['speedup']
            print(f"🎯 Cache speedup: {cache_speedup:.2f}x")
        
        # Scaling analysis
        if 'scaling_limits' in self.results:
            scaling_results = self.results['scaling_limits']
            max_successful_size = max(size for size, result in scaling_results.items() 
                                    if result['success'])
            print(f"📈 Maximum crossbar size: {max_successful_size}x{max_successful_size}")

def main():
    """Run performance benchmark suite."""
    benchmark = BenchmarkSuite()
    results = benchmark.run_full_benchmark()
    return 0

if __name__ == "__main__":
    sys.exit(main())