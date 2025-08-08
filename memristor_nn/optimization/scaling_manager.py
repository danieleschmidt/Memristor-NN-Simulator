"""Advanced scaling and resource management for memristor simulations."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import psutil
from collections import deque, defaultdict

from ..utils.logger import get_logger, PerformanceLogger
from ..utils.error_handling import collect_errors, error_context
from .cache_manager import CacheManager


class ResourceType(Enum):
    """Types of resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    active_processes: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingEvent:
    """Record of scaling decisions."""
    timestamp: float
    event_type: str  # "scale_up", "scale_down", "throttle"
    reason: str
    metrics: ResourceMetrics
    action_taken: str


class AutoScaler:
    """Automatic scaling based on system resources and workload."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = None,
        cpu_threshold_up: float = 80.0,
        cpu_threshold_down: float = 30.0,
        memory_threshold: float = 85.0,
        scale_cooldown: float = 60.0  # seconds
    ):
        """
        Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (None = CPU count)
            cpu_threshold_up: CPU usage % to scale up
            cpu_threshold_down: CPU usage % to scale down
            memory_threshold: Memory usage % limit
            scale_cooldown: Minimum time between scaling actions
        """
        self.min_workers = min_workers
        self.max_workers = max_workers or psutil.cpu_count()
        self.cpu_threshold_up = cpu_threshold_up
        self.cpu_threshold_down = cpu_threshold_down
        self.memory_threshold = memory_threshold
        self.scale_cooldown = scale_cooldown
        
        self.current_workers = min_workers
        self.last_scale_time = 0.0
        self.scaling_history = deque(maxlen=100)
        self.logger = get_logger("auto_scaler")
        
        self.logger.info(f"AutoScaler initialized: {min_workers}-{max_workers} workers")
    
    def should_scale(self, metrics: ResourceMetrics, pending_tasks: int) -> Optional[str]:
        """
        Determine if scaling action is needed.
        
        Args:
            metrics: Current system metrics
            pending_tasks: Number of pending tasks
            
        Returns:
            Scaling action: "scale_up", "scale_down", or None
        """
        # Check cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return None
        
        # Memory protection - always scale down if memory is critical
        if metrics.memory_percent > self.memory_threshold:
            if self.current_workers > self.min_workers:
                return "scale_down"
            return "throttle"
        
        # Scale up conditions
        if (metrics.cpu_percent > self.cpu_threshold_up and 
            pending_tasks > self.current_workers and
            self.current_workers < self.max_workers):
            return "scale_up"
        
        # Scale down conditions
        if (metrics.cpu_percent < self.cpu_threshold_down and
            pending_tasks < self.current_workers * 0.5 and
            self.current_workers > self.min_workers):
            return "scale_down"
        
        return None
    
    def execute_scaling(self, action: str, metrics: ResourceMetrics) -> int:
        """
        Execute scaling action.
        
        Args:
            action: Scaling action to execute
            metrics: Current metrics
            
        Returns:
            New worker count
        """
        old_workers = self.current_workers
        
        if action == "scale_up":
            self.current_workers = min(self.current_workers * 2, self.max_workers)
        elif action == "scale_down":
            self.current_workers = max(self.current_workers // 2, self.min_workers)
        elif action == "throttle":
            # Reduce workers to free memory
            self.current_workers = max(self.current_workers - 1, 1)
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            event_type=action,
            reason=f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%",
            metrics=metrics,
            action_taken=f"{old_workers} -> {self.current_workers} workers"
        )
        
        self.scaling_history.append(event)
        self.last_scale_time = time.time()
        
        self.logger.info(f"Scaling {action}: {old_workers} -> {self.current_workers} workers")
        
        return self.current_workers
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        if not self.scaling_history:
            return {"total_events": 0}
        
        events_by_type = defaultdict(int)
        for event in self.scaling_history:
            events_by_type[event.event_type] += 1
        
        return {
            "total_events": len(self.scaling_history),
            "events_by_type": dict(events_by_type),
            "current_workers": self.current_workers,
            "last_event": self.scaling_history[-1].__dict__ if self.scaling_history else None
        }


class ResourceMonitor:
    """Continuous monitoring of system resources."""
    
    def __init__(self, sampling_interval: float = 5.0):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: Time between samples (seconds)
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics_history = deque(maxlen=1000)
        self.monitor_thread = None
        self.logger = get_logger("resource_monitor")
        
        # Callbacks for resource events
        self.callbacks = defaultdict(list)
    
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for resource alerts
                self._check_alerts(metrics)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            active_processes=len(psutil.pids())
        )
    
    def _check_alerts(self, metrics: ResourceMetrics):
        """Check for resource alert conditions."""
        alerts = []
        
        if metrics.cpu_percent > 90:
            alerts.append("HIGH_CPU")
        
        if metrics.memory_percent > 95:
            alerts.append("CRITICAL_MEMORY")
        elif metrics.memory_percent > 85:
            alerts.append("HIGH_MEMORY")
        
        if metrics.disk_usage_percent > 90:
            alerts.append("HIGH_DISK")
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self.callbacks[alert]:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, alert_type: str, callback: Callable[[ResourceMetrics], None]):
        """Add callback for resource alerts."""
        self.callbacks[alert_type].append(callback)
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 10) -> List[ResourceMetrics]:
        """Get metrics history for specified time period."""
        cutoff_time = time.time() - (minutes * 60)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]


class LoadBalancer:
    """Intelligent load balancing for distributed simulations."""
    
    def __init__(self):
        self.worker_performance = {}  # worker_id -> performance score
        self.worker_load = {}  # worker_id -> current load
        self.task_completion_times = deque(maxlen=1000)
        self.logger = get_logger("load_balancer")
    
    def assign_task(self, workers: List[str], task_weight: float = 1.0) -> str:
        """
        Assign task to optimal worker.
        
        Args:
            workers: Available worker IDs
            task_weight: Relative weight of task
            
        Returns:
            Selected worker ID
        """
        if not workers:
            raise ValueError("No workers available")
        
        # Initialize new workers
        for worker_id in workers:
            if worker_id not in self.worker_performance:
                self.worker_performance[worker_id] = 1.0
                self.worker_load[worker_id] = 0.0
        
        # Calculate efficiency scores
        best_worker = None
        best_score = -1
        
        for worker_id in workers:
            performance = self.worker_performance[worker_id]
            load = max(self.worker_load[worker_id], 0.1)  # Avoid division by zero
            efficiency = performance / load
            
            if efficiency > best_score:
                best_score = efficiency
                best_worker = worker_id
        
        # Update load
        self.worker_load[best_worker] += task_weight
        
        return best_worker
    
    def update_worker_performance(self, worker_id: str, task_duration: float, task_weight: float = 1.0):
        """Update worker performance based on task completion."""
        if task_duration <= 0:
            return
        
        # Performance is inverse of duration (faster = better)
        new_performance = task_weight / task_duration
        
        # Exponential moving average
        if worker_id in self.worker_performance:
            alpha = 0.1
            self.worker_performance[worker_id] = (
                alpha * new_performance + (1 - alpha) * self.worker_performance[worker_id]
            )
        else:
            self.worker_performance[worker_id] = new_performance
        
        # Update load
        self.worker_load[worker_id] = max(0, self.worker_load[worker_id] - task_weight)
        
        # Record completion time
        self.task_completion_times.append(task_duration)
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        if not self.task_completion_times:
            return {"no_data": True}
        
        completion_times = list(self.task_completion_times)
        
        return {
            "worker_count": len(self.worker_performance),
            "avg_completion_time": np.mean(completion_times),
            "completion_time_std": np.std(completion_times),
            "total_load": sum(self.worker_load.values()),
            "worker_performance": dict(self.worker_performance),
            "load_distribution": dict(self.worker_load)
        }


@collect_errors("adaptive_scaling")
class AdaptiveScalingManager:
    """Comprehensive adaptive scaling and resource management."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize adaptive scaling manager.
        
        Args:
            min_workers: Minimum worker count
            max_workers: Maximum worker count
            enable_monitoring: Enable resource monitoring
        """
        self.auto_scaler = AutoScaler(min_workers=min_workers, max_workers=max_workers)
        self.load_balancer = LoadBalancer()
        self.logger = get_logger("adaptive_scaling")
        
        # Resource monitoring
        if enable_monitoring:
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.add_alert_callback("CRITICAL_MEMORY", self._handle_memory_alert)
            self.resource_monitor.add_alert_callback("HIGH_CPU", self._handle_cpu_alert)
            self.resource_monitor.start_monitoring()
        else:
            self.resource_monitor = None
        
        # Performance tracking
        self.scaling_decisions = []
        self.performance_history = deque(maxlen=100)
        
        self.logger.info("Adaptive scaling manager initialized")
    
    def optimize_worker_count(self, pending_tasks: int) -> int:
        """
        Optimize worker count based on current conditions.
        
        Args:
            pending_tasks: Number of pending tasks
            
        Returns:
            Optimal worker count
        """
        if not self.resource_monitor:
            return self.auto_scaler.current_workers
        
        with error_context("worker_optimization", self.logger):
            current_metrics = self.resource_monitor.get_current_metrics()
            if not current_metrics:
                return self.auto_scaler.current_workers
            
            # Determine scaling action
            action = self.auto_scaler.should_scale(current_metrics, pending_tasks)
            
            if action:
                new_count = self.auto_scaler.execute_scaling(action, current_metrics)
                self.scaling_decisions.append({
                    'timestamp': time.time(),
                    'action': action,
                    'old_count': self.auto_scaler.current_workers,
                    'new_count': new_count,
                    'metrics': current_metrics.__dict__
                })
                return new_count
            
            return self.auto_scaler.current_workers
    
    def _handle_memory_alert(self, metrics: ResourceMetrics):
        """Handle critical memory alert."""
        self.logger.warning(f"Critical memory usage: {metrics.memory_percent:.1f}%")
        
        # Force scale down
        if self.auto_scaler.current_workers > 1:
            self.auto_scaler.execute_scaling("scale_down", metrics)
    
    def _handle_cpu_alert(self, metrics: ResourceMetrics):
        """Handle high CPU usage alert."""
        self.logger.info(f"High CPU usage detected: {metrics.cpu_percent:.1f}%")
        # Could implement CPU-specific optimizations here
    
    def record_performance(self, task_count: int, total_time: float, success_rate: float):
        """Record performance metrics for analysis."""
        performance = {
            'timestamp': time.time(),
            'task_count': task_count,
            'total_time': total_time,
            'success_rate': success_rate,
            'throughput': task_count / total_time if total_time > 0 else 0,
            'workers': self.auto_scaler.current_workers
        }
        
        self.performance_history.append(performance)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance statistics."""
        stats = {
            'auto_scaler': self.auto_scaler.get_scaling_stats(),
            'load_balancer': self.load_balancer.get_load_stats(),
            'scaling_decisions': len(self.scaling_decisions),
            'performance_samples': len(self.performance_history)
        }
        
        if self.resource_monitor:
            current_metrics = self.resource_monitor.get_current_metrics()
            stats['current_metrics'] = current_metrics.__dict__ if current_metrics else None
        
        if self.performance_history:
            recent_perf = list(self.performance_history)[-10:]  # Last 10 samples
            stats['recent_performance'] = {
                'avg_throughput': np.mean([p['throughput'] for p in recent_perf]),
                'avg_success_rate': np.mean([p['success_rate'] for p in recent_perf]),
                'avg_workers': np.mean([p['workers'] for p in recent_perf])
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown scaling manager and monitoring."""
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        self.logger.info("Adaptive scaling manager shut down")


# Global scaling manager instance
_global_scaling_manager = None

def get_scaling_manager(
    min_workers: int = 1,
    max_workers: int = None,
    enable_monitoring: bool = True
) -> AdaptiveScalingManager:
    """Get or create global scaling manager instance."""
    global _global_scaling_manager
    if _global_scaling_manager is None:
        _global_scaling_manager = AdaptiveScalingManager(
            min_workers=min_workers,
            max_workers=max_workers,
            enable_monitoring=enable_monitoring
        )
    return _global_scaling_manager