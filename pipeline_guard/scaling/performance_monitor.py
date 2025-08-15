"""
Performance monitoring and metrics collection for pipeline guard
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import logging


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Individual metric value"""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Get age of metric in seconds"""
        return time.time() - self.timestamp


@dataclass
class Alert:
    """Performance alert"""
    id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class Metric:
    """Base metric class"""
    
    def __init__(self, name: str, metric_type: MetricType, 
                 description: str = "", unit: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.unit = unit
        self.values = deque(maxlen=1000)  # Keep last 1000 values
        self.lock = threading.Lock()
        
    def record(self, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        with self.lock:
            self.values.append(MetricValue(
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            ))
            
    def get_current_value(self) -> Optional[float]:
        """Get current (latest) metric value"""
        with self.lock:
            return self.values[-1].value if self.values else None
            
    def get_values(self, duration_seconds: int = None) -> List[MetricValue]:
        """Get metric values within time window"""
        with self.lock:
            if duration_seconds is None:
                return list(self.values)
                
            cutoff_time = time.time() - duration_seconds
            return [v for v in self.values if v.timestamp >= cutoff_time]
            
    def get_statistics(self, duration_seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary of metric values"""
        values = [v.value for v in self.get_values(duration_seconds)]
        
        if not values:
            return {}
            
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "average": sum(values) / len(values),
            "latest": values[-1] if values else 0
        }


class MetricsCollector:
    """
    Collects and manages performance metrics
    """
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.alerts: List[Alert] = []
        self.alert_thresholds: Dict[str, Dict[str, Any]] = {}
        self.collection_enabled = True
        self.collection_thread = None
        self.logger = logging.getLogger(__name__)
        
        # System metrics collection
        self.collect_system_metrics = True
        self._start_collection()
        
    def register_metric(self, metric: Metric):
        """Register a new metric"""
        self.metrics[metric.name] = metric
        self.logger.info(f"Registered metric: {metric.name} ({metric.type.value})")
        
    def record_counter(self, name: str, value: float = 1, 
                      labels: Dict[str, str] = None):
        """Record counter metric"""
        if name not in self.metrics:
            self.register_metric(Metric(name, MetricType.COUNTER))
        self.metrics[name].record(value, labels)
        
    def record_gauge(self, name: str, value: float, 
                    labels: Dict[str, str] = None):
        """Record gauge metric"""
        if name not in self.metrics:
            self.register_metric(Metric(name, MetricType.GAUGE))
        self.metrics[name].record(value, labels)
        
    def record_timer(self, name: str, duration: float,
                    labels: Dict[str, str] = None):
        """Record timer metric (in seconds)"""
        if name not in self.metrics:
            self.register_metric(Metric(name, MetricType.TIMER, unit="seconds"))
        self.metrics[name].record(duration, labels)
        
    def set_alert_threshold(self, metric_name: str, 
                          warning_threshold: float = None,
                          critical_threshold: float = None,
                          condition: str = "greater"):
        """Set alert thresholds for a metric"""
        self.alert_thresholds[metric_name] = {
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "condition": condition  # "greater", "less", "equal"
        }
        
    def _start_collection(self):
        """Start metrics collection thread"""
        if self.collection_thread and self.collection_thread.is_alive():
            return
            
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
    def _collection_loop(self):
        """Main collection loop"""
        while self.collection_enabled:
            try:
                if self.collect_system_metrics:
                    self._collect_system_metrics()
                    
                self._check_alerts()
                time.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(30)
                
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.record_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_gauge("system_memory_percent", memory.percent)
            self.record_gauge("system_memory_available_bytes", memory.available)
            self.record_gauge("system_memory_used_bytes", memory.used)
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            self.record_gauge("system_disk_percent", disk_usage.percent)
            self.record_gauge("system_disk_free_bytes", disk_usage.free)
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                self.record_counter("system_network_bytes_sent", network.bytes_sent)
                self.record_counter("system_network_bytes_recv", network.bytes_recv)
            except:
                pass  # Network metrics not available on all systems
                
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
            
    def _check_alerts(self):
        """Check for alert conditions"""
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name not in self.metrics:
                continue
                
            current_value = self.metrics[metric_name].get_current_value()
            if current_value is None:
                continue
                
            self._evaluate_alert(metric_name, current_value, thresholds)
            
    def _evaluate_alert(self, metric_name: str, current_value: float, 
                       thresholds: Dict[str, Any]):
        """Evaluate alert condition for a metric"""
        condition = thresholds.get("condition", "greater")
        warning_threshold = thresholds.get("warning_threshold")
        critical_threshold = thresholds.get("critical_threshold")
        
        # Check critical threshold first
        if critical_threshold is not None:
            triggered = False
            if condition == "greater" and current_value > critical_threshold:
                triggered = True
            elif condition == "less" and current_value < critical_threshold:
                triggered = True
            elif condition == "equal" and abs(current_value - critical_threshold) < 0.001:
                triggered = True
                
            if triggered:
                self._create_alert(
                    metric_name, AlertSeverity.CRITICAL, 
                    current_value, critical_threshold
                )
                return
                
        # Check warning threshold
        if warning_threshold is not None:
            triggered = False
            if condition == "greater" and current_value > warning_threshold:
                triggered = True
            elif condition == "less" and current_value < warning_threshold:
                triggered = True
            elif condition == "equal" and abs(current_value - warning_threshold) < 0.001:
                triggered = True
                
            if triggered:
                self._create_alert(
                    metric_name, AlertSeverity.WARNING,
                    current_value, warning_threshold
                )
                
    def _create_alert(self, metric_name: str, severity: AlertSeverity,
                     current_value: float, threshold: float):
        """Create new alert"""
        alert_id = f"{metric_name}_{severity.value}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = None
        for alert in self.alerts:
            if (alert.metric_name == metric_name and 
                alert.severity == severity and 
                not alert.resolved):
                existing_alert = alert
                break
                
        if existing_alert:
            return  # Don't create duplicate alerts
            
        alert = Alert(
            id=alert_id,
            metric_name=metric_name,
            severity=severity,
            message=f"Metric {metric_name} {severity.value}: {current_value:.2f} (threshold: {threshold:.2f})",
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {alert.message}")
        
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        
        for name, metric in self.metrics.items():
            stats = metric.get_statistics()
            summary[name] = {
                "type": metric.type.value,
                "description": metric.description,
                "unit": metric.unit,
                "statistics": stats,
                "current_value": metric.get_current_value()
            }
            
        return summary
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active (unresolved) alerts"""
        active_alerts = []
        
        for alert in self.alerts:
            if not alert.resolved:
                active_alerts.append({
                    "id": alert.id,
                    "metric_name": alert.metric_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "age_seconds": (datetime.now() - alert.timestamp).total_seconds()
                })
                
        return active_alerts
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                self.logger.info(f"Alert resolved: {alert.message}")
                return True
        return False
        
    def stop_collection(self):
        """Stop metrics collection"""
        self.collection_enabled = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)


class PerformanceMonitor:
    """
    High-level performance monitoring for pipeline guard
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.logger = logging.getLogger(__name__)
        
        # Setup default alerts
        self._setup_default_alerts()
        
    def _setup_default_alerts(self):
        """Setup default system alerts"""
        # CPU alerts
        self.collector.set_alert_threshold(
            "system_cpu_percent",
            warning_threshold=80.0,
            critical_threshold=95.0,
            condition="greater"
        )
        
        # Memory alerts
        self.collector.set_alert_threshold(
            "system_memory_percent",
            warning_threshold=85.0,
            critical_threshold=95.0,
            condition="greater"
        )
        
        # Disk alerts
        self.collector.set_alert_threshold(
            "system_disk_percent",
            warning_threshold=85.0,
            critical_threshold=95.0,
            condition="greater"
        )
        
    def track_pipeline_operation(self, operation_name: str):
        """Context manager for tracking pipeline operations"""
        return OperationTracker(self.collector, operation_name)
        
    def record_pipeline_event(self, event_type: str, pipeline_id: str = None,
                            duration: float = None, success: bool = True):
        """Record pipeline-specific event"""
        labels = {"event_type": event_type}
        if pipeline_id:
            labels["pipeline_id"] = pipeline_id
            
        # Record event counter
        self.collector.record_counter(f"pipeline_events_total", 1, labels)
        
        # Record success/failure
        status_labels = labels.copy()
        status_labels["status"] = "success" if success else "failure"
        self.collector.record_counter(f"pipeline_status_total", 1, status_labels)
        
        # Record duration if provided
        if duration is not None:
            self.collector.record_timer(f"pipeline_{event_type}_duration", duration, labels)
            
    def record_healing_attempt(self, strategy: str, success: bool, 
                             duration: float, pipeline_id: str = None):
        """Record healing attempt metrics"""
        labels = {"strategy": strategy}
        if pipeline_id:
            labels["pipeline_id"] = pipeline_id
            
        self.collector.record_counter("healing_attempts_total", 1, labels)
        self.collector.record_timer("healing_duration", duration, labels)
        
        status_labels = labels.copy()
        status_labels["status"] = "success" if success else "failure"
        self.collector.record_counter("healing_status_total", 1, status_labels)
        
    def record_api_request(self, endpoint: str, method: str, 
                          status_code: int, duration: float):
        """Record API request metrics"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        }
        
        self.collector.record_counter("api_requests_total", 1, labels)
        self.collector.record_timer("api_request_duration", duration, labels)
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.collector.get_metric_summary(),
            "active_alerts": self.collector.get_active_alerts(),
            "system_status": self._get_system_status(),
            "performance_trends": self._get_performance_trends()
        }
        
    def _get_system_status(self) -> Dict[str, str]:
        """Get overall system status"""
        active_alerts = self.collector.get_active_alerts()
        
        critical_alerts = [a for a in active_alerts if a["severity"] == "critical"]
        warning_alerts = [a for a in active_alerts if a["severity"] == "warning"]
        
        if critical_alerts:
            return {"status": "critical", "message": f"{len(critical_alerts)} critical alerts"}
        elif warning_alerts:
            return {"status": "warning", "message": f"{len(warning_alerts)} warning alerts"}
        else:
            return {"status": "healthy", "message": "All systems operational"}
            
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends over time"""
        trends = {}
        
        key_metrics = ["system_cpu_percent", "system_memory_percent", "system_disk_percent"]
        
        for metric_name in key_metrics:
            if metric_name in self.collector.metrics:
                metric = self.collector.metrics[metric_name]
                
                # Get values for last hour
                recent_values = [v.value for v in metric.get_values(3600)]
                
                if len(recent_values) >= 2:
                    trend_direction = "stable"
                    if recent_values[-1] > recent_values[0] * 1.1:
                        trend_direction = "increasing"
                    elif recent_values[-1] < recent_values[0] * 0.9:
                        trend_direction = "decreasing"
                        
                    trends[metric_name] = {
                        "direction": trend_direction,
                        "change_percent": ((recent_values[-1] - recent_values[0]) / recent_values[0]) * 100,
                        "sample_count": len(recent_values)
                    }
                    
        return trends
        
    def stop(self):
        """Stop performance monitoring"""
        self.collector.stop_collection()


class OperationTracker:
    """Context manager for tracking operation performance"""
    
    def __init__(self, collector: MetricsCollector, operation_name: str):
        self.collector = collector
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        labels = {"operation": self.operation_name}
        
        self.collector.record_timer(f"operation_duration", duration, labels)
        
        status_labels = labels.copy()
        status_labels["status"] = "success" if success else "failure"
        self.collector.record_counter("operation_total", 1, status_labels)