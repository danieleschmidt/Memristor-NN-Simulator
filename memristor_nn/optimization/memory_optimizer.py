"""Memory optimization utilities for memristor simulations."""

import gc
import psutil
import sys
from typing import Dict, Any, Optional, List, Generator
import numpy as np
import torch
from contextlib import contextmanager
import threading
import time

from ..utils.logger import get_logger


class MemoryOptimizer:
    """Memory optimization and monitoring utilities."""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        """
        Initialize memory optimizer.
        
        Args:
            warning_threshold: Memory usage fraction to trigger warnings
            critical_threshold: Memory usage fraction to trigger aggressive cleanup
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = get_logger("memory_optimizer")
        
        # Memory tracking
        self._peak_memory = 0.0
        self._memory_history: List[float] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        virtual_memory = psutil.virtual_memory()
        
        # Process memory in MB
        rss_mb = memory_info.rss / (1024 * 1024)
        vms_mb = memory_info.vms / (1024 * 1024)
        
        # System memory in MB
        total_mb = virtual_memory.total / (1024 * 1024)
        available_mb = virtual_memory.available / (1024 * 1024)
        used_mb = virtual_memory.used / (1024 * 1024)
        
        return {
            "process_rss_mb": rss_mb,
            "process_vms_mb": vms_mb,
            "system_total_mb": total_mb,
            "system_used_mb": used_mb,
            "system_available_mb": available_mb,
            "system_usage_fraction": used_mb / total_mb,
            "process_usage_fraction": rss_mb / total_mb
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        stats = self.get_memory_usage()
        usage_fraction = stats["system_usage_fraction"]
        
        if usage_fraction > self.critical_threshold:
            self.logger.critical(f"Critical memory usage: {usage_fraction:.2%}")
            return True
        elif usage_fraction > self.warning_threshold:
            self.logger.warning(f"High memory usage: {usage_fraction:.2%}")
            return True
        
        return False
    
    def optimize_memory(self, aggressive: bool = False) -> None:
        """Perform memory optimization."""
        self.logger.info("Performing memory optimization")
        
        # Python garbage collection
        collected = gc.collect()
        self.logger.debug(f"Garbage collected {collected} objects")
        
        if aggressive:
            # Force garbage collection for all generations
            for generation in range(3):
                gc.collect(generation)
            
            # Clear PyTorch cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("Cleared CUDA cache")
        
        # Update peak memory tracking
        current_memory = self.get_memory_usage()["process_rss_mb"]
        self._peak_memory = max(self._peak_memory, current_memory)
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start memory monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, args=(interval,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        self.logger.info(f"Started memory monitoring (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        self.logger.info("Stopped memory monitoring")
    
    def _monitor_memory(self, interval: float) -> None:
        """Background memory monitoring loop."""
        while self._monitoring:
            stats = self.get_memory_usage()
            current_usage = stats["system_usage_fraction"]
            
            # Track history
            self._memory_history.append(current_usage)
            if len(self._memory_history) > 100:  # Keep last 100 readings
                self._memory_history.pop(0)
            
            # Check for pressure and optimize if needed
            if current_usage > self.critical_threshold:
                self.optimize_memory(aggressive=True)
            elif current_usage > self.warning_threshold:
                self.optimize_memory(aggressive=False)
            
            time.sleep(interval)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        current_stats = self.get_memory_usage()
        
        # Calculate statistics from history
        if self._memory_history:
            avg_usage = np.mean(self._memory_history)
            max_usage = np.max(self._memory_history)
            min_usage = np.min(self._memory_history)
        else:
            avg_usage = max_usage = min_usage = 0.0
        
        return {
            "current": current_stats,
            "peak_process_mb": self._peak_memory,
            "history_stats": {
                "average_usage": avg_usage,
                "max_usage": max_usage,
                "min_usage": min_usage,
                "samples": len(self._memory_history)
            },
            "thresholds": {
                "warning": self.warning_threshold,
                "critical": self.critical_threshold
            }
        }


@contextmanager
def memory_managed_execution(
    memory_limit_mb: float = 1000.0,
    cleanup_on_exit: bool = True
):
    """
    Context manager for memory-managed execution.
    
    Args:
        memory_limit_mb: Memory limit in MB
        cleanup_on_exit: Whether to cleanup on exit
    """
    optimizer = MemoryOptimizer()
    initial_memory = optimizer.get_memory_usage()["process_rss_mb"]
    
    try:
        yield optimizer
        
        # Check memory usage during execution
        current_memory = optimizer.get_memory_usage()["process_rss_mb"]
        if current_memory > memory_limit_mb:
            optimizer.logger.warning(f"Memory limit exceeded: {current_memory:.1f}MB > {memory_limit_mb:.1f}MB")
            optimizer.optimize_memory(aggressive=True)
        
    finally:
        if cleanup_on_exit:
            optimizer.optimize_memory(aggressive=True)
        
        final_memory = optimizer.get_memory_usage()["process_rss_mb"]
        memory_delta = final_memory - initial_memory
        
        optimizer.logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")


class ChunkedDataProcessor:
    """Process large datasets in memory-efficient chunks."""
    
    def __init__(self, chunk_size: int = 1000, memory_limit_mb: float = 500.0):
        """
        Initialize chunked processor.
        
        Args:
            chunk_size: Number of samples per chunk
            memory_limit_mb: Memory limit for processing
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.logger = get_logger("chunked_processor")
        
    def process_in_chunks(
        self,
        data: torch.Tensor,
        processor_func: callable,
        **kwargs
    ) -> Generator[Any, None, None]:
        """
        Process data in memory-efficient chunks.
        
        Args:
            data: Input data tensor
            processor_func: Function to process each chunk
            **kwargs: Additional arguments for processor_func
            
        Yields:
            Results from processing each chunk
        """
        total_samples = data.shape[0]
        num_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        self.logger.info(f"Processing {total_samples} samples in {num_chunks} chunks")
        
        with memory_managed_execution(self.memory_limit_mb) as optimizer:
            for i in range(num_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, total_samples)
                
                # Extract chunk
                chunk = data[start_idx:end_idx]
                
                # Process chunk
                try:
                    result = processor_func(chunk, **kwargs)
                    yield result
                    
                    # Memory check after each chunk
                    if optimizer.check_memory_pressure():
                        optimizer.optimize_memory(aggressive=True)
                        
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i}: {e}")
                    continue
                
                # Optional: Clear chunk from memory
                del chunk
                
                if (i + 1) % 10 == 0:  # Every 10 chunks
                    gc.collect()
    
    def estimate_memory_usage(self, data_shape: tuple, dtype: torch.dtype = torch.float32) -> float:
        """Estimate memory usage for data processing."""
        # Size in bytes
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = np.prod(data_shape)
        size_bytes = total_elements * element_size
        
        # Convert to MB and add overhead
        size_mb = size_bytes / (1024 * 1024)
        overhead_factor = 2.0  # Account for intermediate computations
        
        return size_mb * overhead_factor


class GPUMemoryManager:
    """Manage GPU memory for CUDA operations."""
    
    def __init__(self):
        self.logger = get_logger("gpu_memory")
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.device_count = torch.cuda.device_count()
            self.logger.info(f"CUDA available with {self.device_count} devices")
        else:
            self.logger.info("CUDA not available")
    
    def get_gpu_memory_stats(self, device: int = 0) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if not self.cuda_available:
            return {}
        
        # Memory in bytes
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        
        # Convert to MB
        return {
            "allocated_mb": allocated / (1024 * 1024),
            "reserved_mb": reserved / (1024 * 1024),
            "max_allocated_mb": max_allocated / (1024 * 1024),
            "max_reserved_mb": max_reserved / (1024 * 1024)
        }
    
    def optimize_gpu_memory(self, device: int = 0) -> None:
        """Optimize GPU memory usage."""
        if not self.cuda_available:
            return
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(device)
        
        self.logger.info(f"Optimized GPU memory for device {device}")
    
    @contextmanager
    def gpu_memory_context(self, device: int = 0):
        """Context manager for GPU memory management."""
        if not self.cuda_available:
            yield
            return
        
        initial_stats = self.get_gpu_memory_stats(device)
        
        try:
            yield
        finally:
            final_stats = self.get_gpu_memory_stats(device)
            
            # Log memory usage
            allocated_delta = final_stats["allocated_mb"] - initial_stats["allocated_mb"]
            self.logger.info(f"GPU memory change: {allocated_delta:+.1f}MB")
            
            # Cleanup
            self.optimize_gpu_memory(device)


class MemoryProfiler:
    """Memory profiler for detailed analysis."""
    
    def __init__(self):
        self.logger = get_logger("memory_profiler")
        self.snapshots: List[Dict[str, Any]] = []
        
    def take_snapshot(self, label: str = "") -> None:
        """Take a memory snapshot."""
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_usage()
        
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "memory_stats": stats
        }
        
        self.snapshots.append(snapshot)
        self.logger.debug(f"Memory snapshot '{label}': {stats['process_rss_mb']:.1f}MB")
    
    def analyze_snapshots(self) -> Dict[str, Any]:
        """Analyze memory snapshots."""
        if len(self.snapshots) < 2:
            return {"error": "Need at least 2 snapshots for analysis"}
        
        # Calculate memory deltas
        deltas = []
        for i in range(1, len(self.snapshots)):
            prev = self.snapshots[i-1]["memory_stats"]["process_rss_mb"]
            curr = self.snapshots[i]["memory_stats"]["process_rss_mb"]
            delta = curr - prev
            
            deltas.append({
                "from_label": self.snapshots[i-1]["label"],
                "to_label": self.snapshots[i]["label"],
                "delta_mb": delta
            })
        
        # Find largest memory increase
        max_increase = max(deltas, key=lambda x: x["delta_mb"])
        
        # Calculate total change
        total_change = (self.snapshots[-1]["memory_stats"]["process_rss_mb"] - 
                       self.snapshots[0]["memory_stats"]["process_rss_mb"])
        
        return {
            "total_snapshots": len(self.snapshots),
            "total_memory_change_mb": total_change,
            "largest_increase": max_increase,
            "deltas": deltas
        }
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate memory profiling report."""
        analysis = self.analyze_snapshots()
        
        if "error" in analysis:
            return analysis["error"]
        
        report = []
        report.append("# Memory Profiling Report\n")
        
        report.append(f"**Total Snapshots:** {analysis['total_snapshots']}")
        report.append(f"**Total Memory Change:** {analysis['total_memory_change_mb']:+.1f} MB")
        report.append(f"**Largest Increase:** {analysis['largest_increase']['delta_mb']:.1f} MB "
                     f"({analysis['largest_increase']['from_label']} → {analysis['largest_increase']['to_label']})\n")
        
        report.append("## Memory Deltas\n")
        for delta in analysis['deltas']:
            report.append(f"- {delta['from_label']} → {delta['to_label']}: {delta['delta_mb']:+.1f} MB")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text