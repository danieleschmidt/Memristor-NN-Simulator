"""
Advanced metrics collection and monitoring system.

Implements:
- Real-time metrics collection
- Multi-dimensional metrics
- Custom metric definitions
- Performance analytics
- Resource monitoring
"""

import time
import threading
import json
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
from pathlib import Path

from ..utils.logger import LoggingMixin


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    aggregation_window: float = 60.0  # seconds
    retention_period: float = 3600.0  # 1 hour


@dataclass
class MetricSample:
    """Individual metric sample."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector(LoggingMixin):
    """Advanced metrics collection and monitoring system."""
    
    def __init__(
        self,
        collection_interval: float = 5.0,
        max_samples_per_metric: int = 1000,
        enable_auto_collection: bool = True
    ):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: How often to collect metrics (seconds)
            max_samples_per_metric: Maximum samples to retain per metric
            enable_auto_collection: Enable automatic collection of system metrics
        """
        super().__init__()
        
        self.collection_interval = collection_interval
        self.max_samples = max_samples_per_metric
        
        # Metric storage
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_samples))
        self.metric_aggregates: Dict[str, Dict[str, float]] = {}
        
        # Collection state
        self.collection_active = False
        self.collection_thread = None
        self.custom_collectors: Dict[str, Callable] = {}
        
        # Performance tracking
        self.collection_stats = {
            "total_collections": 0,
            "collection_errors": 0,
            "avg_collection_time": 0.0,
            "last_collection": 0.0
        }
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        if enable_auto_collection:
            self.start_collection()
        
        self.logger.info(f"Metrics collector initialized with {len(self.metric_definitions)} metrics")
    
    def _initialize_default_metrics(self) -> None:
        """Initialize default system and application metrics."""
        default_metrics = [
            MetricDefinition(
                name="system_cpu_usage",
                metric_type=MetricType.GAUGE,
                description="System CPU utilization percentage",
                unit="percent"
            ),
            MetricDefinition(
                name="system_memory_usage",
                metric_type=MetricType.GAUGE,
                description="System memory utilization",
                unit="bytes"
            ),
            MetricDefinition(
                name="crossbar_operations_total",
                metric_type=MetricType.COUNTER,
                description="Total number of crossbar operations",
                unit="count"
            ),
            MetricDefinition(
                name="simulation_duration",
                metric_type=MetricType.TIMER,
                description="Time taken for simulation operations",
                unit="seconds"
            ),
            MetricDefinition(
                name="device_conductance",
                metric_type=MetricType.HISTOGRAM,
                description="Distribution of device conductances",
                unit="siemens"
            ),
            MetricDefinition(
                name="error_rate",
                metric_type=MetricType.RATE,
                description="Rate of errors per second",
                unit="errors/second"
            ),
            MetricDefinition(
                name="throughput",
                metric_type=MetricType.GAUGE,
                description="Operations per second throughput",
                unit="ops/second"
            ),
            MetricDefinition(
                name="temperature",
                metric_type=MetricType.GAUGE,
                description="Device temperature",
                unit="kelvin",
                labels={"device_type": "memristor"}
            ),
            MetricDefinition(
                name="power_consumption",
                metric_type=MetricType.GAUGE,
                description="System power consumption",
                unit="watts"
            ),
            MetricDefinition(
                name="fault_count",
                metric_type=MetricType.COUNTER,
                description="Number of detected faults",
                unit="count"
            )
        ]
        
        for metric in default_metrics:
            self.metric_definitions[metric.name] = metric
    
    def define_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        unit: str = "",
        labels: Optional[Dict[str, str]] = None,
        aggregation_window: float = 60.0
    ) -> None:
        """
        Define a new metric for collection.
        
        Args:
            name: Metric name (must be unique)
            metric_type: Type of metric
            description: Human-readable description
            unit: Unit of measurement
            labels: Default labels for metric
            aggregation_window: Time window for aggregation
        """
        if name in self.metric_definitions:
            self.logger.warning(f"Metric '{name}' already exists, updating definition")
        
        metric_def = MetricDefinition(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit,
            labels=labels or {},
            aggregation_window=aggregation_window
        )
        
        self.metric_definitions[name] = metric_def
        self.logger.info(f"Defined metric: {name} ({metric_type.value})")
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        """
        Record a metric sample.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Additional labels for this sample
            timestamp: Sample timestamp (default: current time)
        """
        if name not in self.metric_definitions:
            self.logger.warning(f"Recording undefined metric: {name}")
            return
        
        sample = MetricSample(
            timestamp=timestamp or time.time(),
            value=value,
            labels=labels or {}
        )
        
        self.metric_samples[name].append(sample)
        
        # Update aggregates
        self._update_aggregates(name)
    
    def _update_aggregates(self, metric_name: str) -> None:
        """Update aggregated statistics for a metric."""
        if metric_name not in self.metric_samples:
            return
        
        samples = list(self.metric_samples[metric_name])
        if not samples:
            return
        
        metric_def = self.metric_definitions[metric_name]
        current_time = time.time()
        
        # Filter samples within aggregation window
        window_samples = [
            s for s in samples
            if current_time - s.timestamp <= metric_def.aggregation_window
        ]
        
        if not window_samples:
            return
        
        values = [s.value for s in window_samples]
        
        # Calculate aggregates based on metric type
        aggregates = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "latest": values[-1],
            "window_start": min(s.timestamp for s in window_samples),
            "window_end": max(s.timestamp for s in window_samples)
        }
        
        if len(values) > 1:
            aggregates["stddev"] = statistics.stdev(values)
            aggregates["median"] = statistics.median(values)
        else:
            aggregates["stddev"] = 0.0
            aggregates["median"] = values[0]
        
        # Type-specific aggregates
        if metric_def.metric_type == MetricType.COUNTER:
            # Rate calculation for counters
            time_span = aggregates["window_end"] - aggregates["window_start"]
            if time_span > 0:
                aggregates["rate"] = (aggregates["max"] - aggregates["min"]) / time_span
            else:
                aggregates["rate"] = 0.0
        
        elif metric_def.metric_type == MetricType.HISTOGRAM:
            # Percentiles for histograms
            sorted_values = sorted(values)
            aggregates["p50"] = self._percentile(sorted_values, 0.5)
            aggregates["p90"] = self._percentile(sorted_values, 0.9)
            aggregates["p95"] = self._percentile(sorted_values, 0.95)
            aggregates["p99"] = self._percentile(sorted_values, 0.99)
        
        self.metric_aggregates[metric_name] = aggregates
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0.0
        
        k = (len(sorted_values) - 1) * percentile
        f = int(k)
        c = k - f
        
        if f + 1 < len(sorted_values):
            return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
        else:
            return sorted_values[f]
    
    def register_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]) -> None:
        """
        Register a custom metric collector function.
        
        Args:
            name: Collector name
            collector_func: Function that returns dict of metric_name -> value
        """
        self.custom_collectors[name] = collector_func
        self.logger.info(f"Registered custom collector: {name}")
    
    def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self.collection_active:
            self.logger.warning("Metrics collection already active")
            return
        
        self.collection_active = True
        
        def collection_loop():
            while self.collection_active:
                try:
                    collection_start = time.time()
                    
                    # Collect system metrics
                    self._collect_system_metrics()
                    
                    # Collect custom metrics
                    self._collect_custom_metrics()
                    
                    # Update collection stats
                    collection_time = time.time() - collection_start
                    self.collection_stats["total_collections"] += 1
                    self.collection_stats["last_collection"] = time.time()
                    
                    # Update average collection time
                    current_avg = self.collection_stats["avg_collection_time"]
                    count = self.collection_stats["total_collections"]
                    self.collection_stats["avg_collection_time"] = (
                        (current_avg * (count - 1) + collection_time) / count
                    )
                    
                    # Sleep until next collection
                    time.sleep(max(0, self.collection_interval - collection_time))
                    
                except Exception as e:
                    self.logger.error(f"Metrics collection error: {e}")
                    self.collection_stats["collection_errors"] += 1
                    time.sleep(self.collection_interval)
        
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("Metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10.0)
        self.logger.info("Metrics collection stopped")
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            current_time = time.time()
            
            # CPU usage (simulated)
            import random
            cpu_usage = random.uniform(20, 80)  # Simulate CPU usage
            self.record_metric("system_cpu_usage", cpu_usage, timestamp=current_time)
            
            # Memory usage (simulated)
            memory_usage = random.uniform(1e9, 4e9)  # 1-4 GB
            self.record_metric("system_memory_usage", memory_usage, timestamp=current_time)
            
            # Power consumption (simulated)
            power = random.uniform(50, 200)  # 50-200 watts
            self.record_metric("power_consumption", power, timestamp=current_time)
            
            # Temperature (simulated)
            temperature = random.uniform(300, 350)  # 300-350 K
            self.record_metric("temperature", temperature, 
                             labels={"device_type": "memristor"}, timestamp=current_time)
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
    
    def _collect_custom_metrics(self) -> None:
        """Collect metrics from registered custom collectors."""
        current_time = time.time()
        
        for collector_name, collector_func in self.custom_collectors.items():
            try:
                metrics = collector_func()
                for metric_name, value in metrics.items():
                    self.record_metric(
                        metric_name, 
                        value, 
                        labels={"collector": collector_name},
                        timestamp=current_time
                    )
                    
            except Exception as e:
                self.logger.error(f"Custom collector '{collector_name}' failed: {e}")
    
    def get_metric_value(
        self,
        name: str,
        aggregation: str = "latest"
    ) -> Optional[float]:
        """
        Get current value of a metric.
        
        Args:
            name: Metric name
            aggregation: Aggregation type (latest, mean, max, min, etc.)
            
        Returns:
            Metric value or None if not found
        """
        if name not in self.metric_aggregates:
            return None
        
        return self.metric_aggregates[name].get(aggregation)
    
    def get_metric_history(
        self,
        name: str,
        time_range: Optional[Tuple[float, float]] = None
    ) -> List[MetricSample]:
        """
        Get historical samples for a metric.
        
        Args:
            name: Metric name
            time_range: Optional time range (start_time, end_time)
            
        Returns:
            List of metric samples
        """
        if name not in self.metric_samples:
            return []
        
        samples = list(self.metric_samples[name])
        
        if time_range:
            start_time, end_time = time_range
            samples = [
                s for s in samples
                if start_time <= s.timestamp <= end_time
            ]
        
        return samples
    
    def query_metrics(
        self,
        metric_pattern: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, List[MetricSample]]:
        """
        Query metrics by pattern, labels, and time range.
        
        Args:
            metric_pattern: Metric name pattern (simple wildcard supported)
            labels: Label filters
            time_range: Time range filter
            
        Returns:
            Dictionary of metric_name -> samples
        """
        results = {}
        
        for metric_name in self.metric_samples:
            # Pattern matching
            if metric_pattern and not self._matches_pattern(metric_name, metric_pattern):
                continue
            
            samples = self.get_metric_history(metric_name, time_range)
            
            # Label filtering
            if labels:
                filtered_samples = []
                for sample in samples:
                    if all(sample.labels.get(k) == v for k, v in labels.items()):
                        filtered_samples.append(sample)
                samples = filtered_samples
            
            if samples:
                results[metric_name] = samples
        
        return results
    
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Simple wildcard pattern matching."""
        if "*" not in pattern:
            return text == pattern
        
        # Split pattern by asterisks
        parts = pattern.split("*")
        
        # Check if text starts with first part
        if not text.startswith(parts[0]):
            return False
        
        # Check if text ends with last part
        if not text.endswith(parts[-1]):
            return False
        
        # Simple implementation - could be enhanced
        return True
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        current_time = time.time()
        
        return {
            "collection_info": {
                "active": self.collection_active,
                "interval_seconds": self.collection_interval,
                "total_metrics": len(self.metric_definitions),
                "active_metrics": len([
                    name for name, samples in self.metric_samples.items()
                    if samples and current_time - samples[-1].timestamp < 300  # Active in last 5 minutes
                ])
            },
            "collection_stats": self.collection_stats,
            "metric_definitions": {
                name: {
                    "type": defn.metric_type.value,
                    "description": defn.description,
                    "unit": defn.unit,
                    "labels": defn.labels,
                    "sample_count": len(self.metric_samples.get(name, []))
                }
                for name, defn in self.metric_definitions.items()
            },
            "recent_aggregates": {
                name: {k: v for k, v in agg.items() if k not in ["window_start", "window_end"]}
                for name, agg in self.metric_aggregates.items()
            }
        }
    
    def export_metrics(
        self,
        format_type: str = "json",
        output_path: Optional[Path] = None,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Export metrics in various formats.
        
        Args:
            format_type: Export format (json, prometheus, csv)
            output_path: Optional file path to save export
            time_range: Optional time range to export
            
        Returns:
            Exported metrics data
        """
        try:
            if format_type == "json":
                data = self._export_json(time_range)
            elif format_type == "prometheus":
                data = self._export_prometheus(time_range)
            elif format_type == "csv":
                data = self._export_csv(time_range)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if format_type == "json":
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    with open(output_path, 'w') as f:
                        f.write(data)
                
                self.logger.info(f"Metrics exported to {output_path}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Metrics export failed: {e}")
            raise
    
    def _export_json(self, time_range: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """Export metrics in JSON format."""
        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "collection_interval": self.collection_interval,
                "time_range": time_range
            },
            "definitions": {
                name: {
                    "type": defn.metric_type.value,
                    "description": defn.description,
                    "unit": defn.unit,
                    "labels": defn.labels
                }
                for name, defn in self.metric_definitions.items()
            },
            "samples": {}
        }
        
        for metric_name, samples in self.metric_samples.items():
            sample_data = []
            for sample in samples:
                if time_range:
                    if not (time_range[0] <= sample.timestamp <= time_range[1]):
                        continue
                
                sample_data.append({
                    "timestamp": sample.timestamp,
                    "value": sample.value,
                    "labels": sample.labels
                })
            
            if sample_data:
                export_data["samples"][metric_name] = sample_data
        
        return export_data
    
    def _export_prometheus(self, time_range: Optional[Tuple[float, float]]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, definition in self.metric_definitions.items():
            # Add metric metadata
            lines.append(f"# HELP {metric_name} {definition.description}")
            lines.append(f"# TYPE {metric_name} {definition.metric_type.value}")
            
            # Add samples
            if metric_name in self.metric_samples:
                for sample in self.metric_samples[metric_name]:
                    if time_range:
                        if not (time_range[0] <= sample.timestamp <= time_range[1]):
                            continue
                    
                    # Format labels
                    label_str = ""
                    if sample.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in sample.labels.items()]
                        label_str = "{" + ",".join(label_pairs) + "}"
                    
                    # Add sample line
                    timestamp_ms = int(sample.timestamp * 1000)
                    lines.append(f"{metric_name}{label_str} {sample.value} {timestamp_ms}")
        
        return "\\n".join(lines)
    
    def _export_csv(self, time_range: Optional[Tuple[float, float]]) -> str:
        """Export metrics in CSV format."""
        lines = ["metric_name,timestamp,value,labels"]
        
        for metric_name, samples in self.metric_samples.items():
            for sample in samples:
                if time_range:
                    if not (time_range[0] <= sample.timestamp <= time_range[1]):
                        continue
                
                labels_str = json.dumps(sample.labels) if sample.labels else ""
                lines.append(f"{metric_name},{sample.timestamp},{sample.value},\"{labels_str}\"")
        
        return "\\n".join(lines)
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data structure suitable for dashboard visualization."""
        dashboard_data = {
            "timestamp": time.time(),
            "metrics": {},
            "charts": []
        }
        
        # Group metrics by type for different chart types
        metric_groups = {
            MetricType.GAUGE: [],
            MetricType.COUNTER: [],
            MetricType.HISTOGRAM: [],
            MetricType.TIMER: []
        }
        
        for name, definition in self.metric_definitions.items():
            if name in self.metric_aggregates:
                metric_groups[definition.metric_type].append(name)
                
                # Add current values
                dashboard_data["metrics"][name] = {
                    "current_value": self.metric_aggregates[name].get("latest", 0),
                    "unit": definition.unit,
                    "description": definition.description
                }
        
        # Create chart configurations
        if metric_groups[MetricType.GAUGE]:
            dashboard_data["charts"].append({
                "type": "line_chart",
                "title": "System Gauges",
                "metrics": metric_groups[MetricType.GAUGE],
                "time_series": True
            })
        
        if metric_groups[MetricType.COUNTER]:
            dashboard_data["charts"].append({
                "type": "bar_chart",
                "title": "Counters",
                "metrics": metric_groups[MetricType.COUNTER],
                "show_rate": True
            })
        
        if metric_groups[MetricType.HISTOGRAM]:
            dashboard_data["charts"].append({
                "type": "distribution_chart",
                "title": "Distributions",
                "metrics": metric_groups[MetricType.HISTOGRAM],
                "show_percentiles": True
            })
        
        return dashboard_data