#!/usr/bin/env python3
"""
Scaling and performance test without PyTorch dependencies.
Focus on core computational kernels and optimization strategies.
"""

import sys
sys.path.insert(0, '/root/repo')

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import psutil

from memristor_nn.core.device_models import DeviceModel, DeviceConfig
from memristor_nn.utils.logger import get_logger, PerformanceLogger

logger = get_logger(__name__)

class ScalingTest:
    """Test scaling and performance optimizations."""
    
    def __init__(self):
        self.results = {}
    
    def test_crossbar_scaling(self):
        """Test crossbar computation scaling."""
        print("=== Crossbar Scaling Test ===")
        
        sizes = [64, 128, 256, 512, 1024]
        results = {}
        
        for size in sizes:
            print(f"Testing {size}x{size} crossbar...", end=" ")
            
            try:
                with PerformanceLogger(f"crossbar_{size}") as perf:
                    # Generate memristor conductance matrix
                    ron = 1e4  # Low resistance
                    roff = 1e6  # High resistance
                    states = np.random.random((size, size))  # Random device states
                    conductances = 1/ron + states * (1/roff - 1/ron)
                    
                    # Input voltage vector
                    voltages = np.random.randn(size)
                    
                    # Analog matrix-vector multiplication
                    start_time = time.time()
                    currents = np.dot(conductances.T, voltages)
                    
                    # Add device variations and noise
                    variation = np.random.normal(1.0, 0.1, currents.shape)
                    noise = np.random.normal(0, 0.01 * np.abs(currents))
                    noisy_currents = currents * variation + noise
                    
                    computation_time = time.time() - start_time
                
                # Calculate metrics
                memory_mb = (conductances.nbytes + voltages.nbytes + currents.nbytes) / 1024 / 1024
                ops = size * size  # MAC operations
                throughput = ops / computation_time
                
                results[size] = {
                    'time_s': computation_time,
                    'memory_mb': memory_mb,
                    'throughput_mops': throughput / 1e6,
                    'success': True
                }
                
                print(f"✓ {computation_time:.4f}s, {throughput/1e6:.1f} MOPS")
                
            except Exception as e:
                results[size] = {'success': False, 'error': str(e)}
                print(f"✗ Failed: {e}")
        
        self.results['crossbar_scaling'] = results
        return results
    
    def test_parallel_performance(self):
        """Test parallel execution performance.""" 
        print("\n=== Parallel Performance Test ===")
        
        def simulate_device(params):
            """Simulate a single memristor device."""
            device_id, voltage, duration = params
            
            # Simulate device physics (simplified)
            ron, roff = 1e4, 1e6
            state = 0.5  # Initial state
            
            for _ in range(int(duration * 1000)):  # Simulate time steps
                if abs(voltage) > 0.5:
                    state += 0.001 * np.sign(voltage)
                    state = max(0, min(1, state))
            
            resistance = ron + state * (roff - ron)
            return device_id, 1/resistance
        
        # Test different numbers of devices and workers
        device_counts = [100, 500, 1000, 2000]
        worker_counts = [1, 2, 4, mp.cpu_count()]
        
        results = {}
        
        for num_devices in device_counts:
            device_results = {}
            
            # Generate test parameters
            params = [(i, np.random.uniform(-2, 2), np.random.uniform(0.1, 1.0)) 
                     for i in range(num_devices)]
            
            print(f"Testing {num_devices} devices:")
            
            # Sequential baseline
            start_time = time.time()
            sequential_results = [simulate_device(p) for p in params]
            sequential_time = time.time() - start_time
            
            device_results['sequential'] = {
                'time_s': sequential_time,
                'speedup': 1.0
            }
            print(f"  Sequential: {sequential_time:.3f}s")
            
            # Parallel execution
            for num_workers in worker_counts:
                if num_workers == 1:
                    continue
                
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    parallel_results = list(executor.map(simulate_device, params))
                parallel_time = time.time() - start_time
                
                speedup = sequential_time / parallel_time
                efficiency = speedup / num_workers
                
                device_results[f'workers_{num_workers}'] = {
                    'time_s': parallel_time,
                    'speedup': speedup,
                    'efficiency': efficiency
                }
                
                print(f"  {num_workers} workers: {parallel_time:.3f}s, {speedup:.2f}x speedup")
            
            results[num_devices] = device_results
        
        self.results['parallel_performance'] = results
        return results
    
    def test_memory_optimization(self):
        """Test memory usage optimization."""
        print("\n=== Memory Optimization Test ===")
        
        process = psutil.Process()
        results = {}
        
        array_sizes = [512, 1024, 2048, 4096]
        
        for size in array_sizes:
            print(f"Testing {size}x{size} arrays...")
            
            test_results = {}
            
            # Test 1: Standard float64
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            arrays_f64 = []
            start_time = time.time()
            for _ in range(5):
                arr = np.random.randn(size, size).astype(np.float64)
                arrays_f64.append(arr)
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_f64 = peak_memory - initial_memory
            
            del arrays_f64
            
            # Test 2: Optimized float32
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            arrays_f32 = []
            start_time = time.time()
            for _ in range(5):
                arr = np.random.randn(size, size).astype(np.float32)
                arrays_f32.append(arr)
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_f32 = peak_memory - initial_memory
            
            del arrays_f32
            
            # Test 3: Block-wise processing (memory-efficient)
            def process_blocks(size, block_size=256):
                """Process large arrays in blocks to save memory."""
                result = 0
                for i in range(0, size, block_size):
                    for j in range(0, size, block_size):
                        end_i = min(i + block_size, size)
                        end_j = min(j + block_size, size)
                        
                        block = np.random.randn(end_i - i, end_j - j).astype(np.float32)
                        result += np.sum(block ** 2)
                        del block
                
                return result
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.time()
            block_result = process_blocks(size)
            block_time = time.time() - start_time
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_blocks = peak_memory - initial_memory
            
            # Calculate savings
            memory_reduction_f32 = ((memory_f64 - memory_f32) / memory_f64) * 100
            memory_reduction_blocks = ((memory_f64 - memory_blocks) / memory_f64) * 100
            
            test_results = {
                'float64_memory_mb': memory_f64,
                'float32_memory_mb': memory_f32,
                'blocks_memory_mb': memory_blocks,
                'f32_reduction_percent': memory_reduction_f32,
                'blocks_reduction_percent': memory_reduction_blocks
            }
            
            results[size] = test_results
            
            print(f"  Float64: {memory_f64:.1f} MB")
            print(f"  Float32: {memory_f32:.1f} MB ({memory_reduction_f32:.1f}% reduction)")
            print(f"  Blocks:  {memory_blocks:.1f} MB ({memory_reduction_blocks:.1f}% reduction)")
        
        self.results['memory_optimization'] = results
        return results
    
    def test_algorithmic_optimization(self):
        """Test algorithmic optimization strategies."""
        print("\n=== Algorithmic Optimization Test ===")
        
        results = {}
        
        # Test sparse vs dense matrix operations
        densities = [0.1, 0.3, 0.5, 0.7, 1.0]
        matrix_size = 1024
        
        for density in densities:
            print(f"Testing {density:.1%} matrix density...")
            
            # Generate sparse matrix
            matrix = np.random.random((matrix_size, matrix_size))
            mask = np.random.random((matrix_size, matrix_size)) < density
            sparse_matrix = matrix * mask
            
            vector = np.random.randn(matrix_size)
            
            # Dense operation
            start_time = time.time()
            dense_result = np.dot(sparse_matrix, vector)
            dense_time = time.time() - start_time
            
            # Sparse operation (manual implementation)
            start_time = time.time()
            sparse_result = np.zeros(matrix_size)
            rows, cols = np.nonzero(sparse_matrix)
            for r, c in zip(rows, cols):
                sparse_result[r] += sparse_matrix[r, c] * vector[c]
            sparse_time = time.time() - start_time
            
            # Calculate speedup
            speedup = dense_time / sparse_time if sparse_time > 0 else 1.0
            sparsity_benefit = density < 0.5 and speedup > 1.0
            
            results[density] = {
                'dense_time_s': dense_time,
                'sparse_time_s': sparse_time,
                'speedup': speedup,
                'benefits_from_sparsity': sparsity_benefit,
                'memory_savings_percent': (1 - density) * 100
            }
            
            print(f"  Dense: {dense_time:.4f}s, Sparse: {sparse_time:.4f}s, Speedup: {speedup:.2f}x")
        
        self.results['algorithmic_optimization'] = results
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all scaling and performance tests."""
        print("🚀 MEMRISTOR-NN SCALING & PERFORMANCE TEST SUITE")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all tests
        self.test_crossbar_scaling()
        self.test_parallel_performance()
        self.test_memory_optimization()
        self.test_algorithmic_optimization()
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = {
            'total_test_time_s': total_time,
            'system_info': {
                'cpu_count': mp.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1024**3
            },
            'test_results': self.results
        }
        
        print(f"\n🎯 TESTS COMPLETE: {total_time:.2f}s total")
        self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self):
        """Generate performance optimization recommendations."""
        print("\n📊 PERFORMANCE RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = []
        
        # Crossbar scaling analysis
        if 'crossbar_scaling' in self.results:
            max_size = max(size for size, result in self.results['crossbar_scaling'].items() 
                          if result['success'])
            recommendations.append(f"✅ Maximum efficient crossbar size: {max_size}x{max_size}")
        
        # Parallel performance analysis
        if 'parallel_performance' in self.results:
            best_speedup = 0
            best_config = None
            for device_count, configs in self.results['parallel_performance'].items():
                for config, metrics in configs.items():
                    if 'speedup' in metrics and metrics['speedup'] > best_speedup:
                        best_speedup = metrics['speedup']
                        best_config = (device_count, config)
            
            if best_config:
                recommendations.append(f"⚡ Best parallel config: {best_config[1]} for {best_config[0]} devices ({best_speedup:.2f}x speedup)")
        
        # Memory optimization analysis
        if 'memory_optimization' in self.results:
            avg_f32_savings = np.mean([r['f32_reduction_percent'] for r in self.results['memory_optimization'].values()])
            avg_block_savings = np.mean([r['blocks_reduction_percent'] for r in self.results['memory_optimization'].values()])
            
            recommendations.append(f"💾 Use float32: Average {avg_f32_savings:.1f}% memory savings")
            recommendations.append(f"🔄 Use block processing: Average {avg_block_savings:.1f}% memory savings")
        
        # Algorithmic optimization analysis
        if 'algorithmic_optimization' in self.results:
            best_sparse_density = None
            best_sparse_speedup = 0
            for density, result in self.results['algorithmic_optimization'].items():
                if result['benefits_from_sparsity'] and result['speedup'] > best_sparse_speedup:
                    best_sparse_speedup = result['speedup']
                    best_sparse_density = density
            
            if best_sparse_density:
                recommendations.append(f"🎯 Use sparse algorithms for matrices <{best_sparse_density:.0%} density")
        
        # General recommendations
        recommendations.extend([
            "🔧 Enable compiler optimizations (e.g., -O3, vectorization)",
            "🎪 Consider GPU acceleration for large crossbars (>512x512)",
            "📈 Monitor memory usage and implement caching for repeated computations",
            "🔄 Use lazy evaluation for large computation graphs"
        ])
        
        for rec in recommendations:
            print(f"  {rec}")

def main():
    """Run scaling and performance test suite."""
    test_suite = ScalingTest()
    results = test_suite.run_all_tests()
    return 0

if __name__ == "__main__":
    sys.exit(main())