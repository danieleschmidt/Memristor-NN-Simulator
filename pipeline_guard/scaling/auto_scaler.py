"""
Auto-scaling system for pipeline guard components
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    """Types of scaling triggers"""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    REQUEST_RATE = "request_rate"
    QUEUE_SIZE = "queue_size"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingMetric:
    """Individual scaling metric"""
    name: str
    current_value: float
    threshold_up: float
    threshold_down: float
    weight: float = 1.0
    evaluation_window: int = 300  # 5 minutes
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_value(self, value: float):
        """Add new metric value"""
        self.current_value = value
        self.history.append({
            "value": value,
            "timestamp": time.time()
        })
        
    def get_average(self, window_seconds: int = None) -> float:
        """Get average value over time window"""
        if not self.history:
            return self.current_value
            
        window_seconds = window_seconds or self.evaluation_window
        cutoff_time = time.time() - window_seconds
        
        recent_values = [
            entry["value"] for entry in self.history
            if entry["timestamp"] > cutoff_time
        ]
        
        return sum(recent_values) / len(recent_values) if recent_values else self.current_value
        
    def should_scale_up(self) -> bool:
        """Check if metric indicates need to scale up"""
        avg_value = self.get_average()
        return avg_value > self.threshold_up
        
    def should_scale_down(self) -> bool:
        """Check if metric indicates need to scale down"""
        avg_value = self.get_average()
        return avg_value < self.threshold_down


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior"""
    min_instances: int = 1
    max_instances: int = 10
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 600  # 10 minutes
    scale_up_step: int = 1
    scale_down_step: int = 1
    evaluation_periods: int = 2  # Consecutive periods before scaling
    enabled: bool = True


class AutoScaler:
    """
    Intelligent auto-scaling system for pipeline guard components
    """
    
    def __init__(self, scaling_policy: ScalingPolicy = None):
        self.policy = scaling_policy or ScalingPolicy()
        self.metrics: Dict[str, ScalingMetric] = {}
        self.current_instances = self.policy.min_instances
        self.last_scale_action = None
        self.last_scale_time = 0
        self.scale_decisions = deque(maxlen=10)
        
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread = None
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable] = None
        self.scale_down_callback: Optional[Callable] = None
        
    def add_metric(self, metric: ScalingMetric):
        """Add a scaling metric to monitor"""
        self.metrics[metric.name] = metric
        self.logger.info(f"Added scaling metric: {metric.name}")
        
    def set_scale_callbacks(self, scale_up_fn: Callable, scale_down_fn: Callable):
        """Set callbacks for scaling actions"""
        self.scale_up_callback = scale_up_fn
        self.scale_down_callback = scale_down_fn
        
    def start_monitoring(self):
        """Start auto-scaling monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
        self.logger.info("Auto-scaler monitoring started")
        
    def stop_monitoring(self):
        """Stop auto-scaling monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Auto-scaler monitoring stopped")
        
    def update_metric(self, metric_name: str, value: float):
        """Update a metric value"""
        if metric_name in self.metrics:
            self.metrics[metric_name].add_value(value)
            
    def _monitoring_loop(self):
        """Main monitoring loop for auto-scaling"""
        while self.monitoring:
            try:
                if self.policy.enabled:
                    decision = self._evaluate_scaling_decision()
                    if decision != ScalingDirection.STABLE:
                        self._execute_scaling_decision(decision)
                        
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
                
    def _evaluate_scaling_decision(self) -> ScalingDirection:
        """Evaluate whether scaling action is needed"""
        if not self.metrics:
            return ScalingDirection.STABLE
            
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        
        for metric in self.metrics.values():
            total_weight += metric.weight
            
            if metric.should_scale_up():
                scale_up_votes += metric.weight
                
            if metric.should_scale_down():
                scale_down_votes += metric.weight
                
        # Require majority weighted vote for scaling
        scale_up_threshold = total_weight * 0.6
        scale_down_threshold = total_weight * 0.7  # Higher threshold for scale down
        
        if scale_up_votes >= scale_up_threshold:
            decision = ScalingDirection.UP
        elif scale_down_votes >= scale_down_threshold:
            decision = ScalingDirection.DOWN
        else:
            decision = ScalingDirection.STABLE
            
        # Record decision for evaluation period tracking
        self.scale_decisions.append({
            "decision": decision,
            "timestamp": time.time(),
            "scale_up_votes": scale_up_votes,
            "scale_down_votes": scale_down_votes,
            "total_weight": total_weight
        })
        
        # Check if we have consistent decisions over evaluation periods
        recent_decisions = [
            d["decision"] for d in self.scale_decisions
            if d["timestamp"] > time.time() - 120  # Last 2 minutes
        ]
        
        if len(recent_decisions) >= self.policy.evaluation_periods:
            # Check if all recent decisions are the same
            if all(d == decision for d in recent_decisions[-self.policy.evaluation_periods:]):
                return decision
                
        return ScalingDirection.STABLE
        
    def _execute_scaling_decision(self, decision: ScalingDirection):
        """Execute scaling decision if cooldown allows"""
        current_time = time.time()
        
        if decision == ScalingDirection.UP:
            if (current_time - self.last_scale_time) < self.policy.scale_up_cooldown:
                return  # Still in cooldown
                
            if self.current_instances >= self.policy.max_instances:
                return  # Already at max capacity
                
            new_instances = min(
                self.current_instances + self.policy.scale_up_step,
                self.policy.max_instances
            )
            
            if self._scale_up(new_instances):
                self.current_instances = new_instances
                self.last_scale_action = "scale_up"
                self.last_scale_time = current_time
                self.logger.info(f"Scaled up to {new_instances} instances")
                
        elif decision == ScalingDirection.DOWN:
            if (current_time - self.last_scale_time) < self.policy.scale_down_cooldown:
                return  # Still in cooldown
                
            if self.current_instances <= self.policy.min_instances:
                return  # Already at min capacity
                
            new_instances = max(
                self.current_instances - self.policy.scale_down_step,
                self.policy.min_instances
            )
            
            if self._scale_down(new_instances):
                self.current_instances = new_instances
                self.last_scale_action = "scale_down"
                self.last_scale_time = current_time
                self.logger.info(f"Scaled down to {new_instances} instances")
                
    def _scale_up(self, target_instances: int) -> bool:
        """Execute scale up action"""
        if self.scale_up_callback:
            try:
                return self.scale_up_callback(target_instances)
            except Exception as e:
                self.logger.error(f"Scale up callback failed: {e}")
                return False
        return True
        
    def _scale_down(self, target_instances: int) -> bool:
        """Execute scale down action"""
        if self.scale_down_callback:
            try:
                return self.scale_down_callback(target_instances)
            except Exception as e:
                self.logger.error(f"Scale down callback failed: {e}")
                return False
        return True
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        metric_status = {}
        for name, metric in self.metrics.items():
            metric_status[name] = {
                "current_value": metric.current_value,
                "average_value": metric.get_average(),
                "threshold_up": metric.threshold_up,
                "threshold_down": metric.threshold_down,
                "should_scale_up": metric.should_scale_up(),
                "should_scale_down": metric.should_scale_down()
            }
            
        return {
            "current_instances": self.current_instances,
            "min_instances": self.policy.min_instances,
            "max_instances": self.policy.max_instances,
            "last_scale_action": self.last_scale_action,
            "last_scale_time": self.last_scale_time,
            "time_since_last_scale": time.time() - self.last_scale_time,
            "scaling_enabled": self.policy.enabled,
            "metrics": metric_status,
            "recent_decisions": list(self.scale_decisions),
            "timestamp": datetime.now().isoformat()
        }
        
    def force_scale(self, target_instances: int) -> bool:
        """Force scaling to specific number of instances"""
        if target_instances < self.policy.min_instances or target_instances > self.policy.max_instances:
            self.logger.error(f"Target instances {target_instances} outside allowed range")
            return False
            
        if target_instances > self.current_instances:
            success = self._scale_up(target_instances)
        elif target_instances < self.current_instances:
            success = self._scale_down(target_instances)
        else:
            return True  # Already at target
            
        if success:
            self.current_instances = target_instances
            self.last_scale_action = "manual"
            self.last_scale_time = time.time()
            self.logger.info(f"Manually scaled to {target_instances} instances")
            
        return success
        
    def update_policy(self, new_policy: ScalingPolicy):
        """Update scaling policy"""
        old_policy = self.policy
        self.policy = new_policy
        
        # Adjust current instances if needed
        if self.current_instances < new_policy.min_instances:
            self.force_scale(new_policy.min_instances)
        elif self.current_instances > new_policy.max_instances:
            self.force_scale(new_policy.max_instances)
            
        self.logger.info(f"Updated scaling policy: {old_policy} -> {new_policy}")
        
    def get_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations based on current metrics"""
        recommendations = []
        
        for name, metric in self.metrics.items():
            avg_value = metric.get_average()
            
            if avg_value > metric.threshold_up * 1.5:
                recommendations.append({
                    "type": "urgent_scale_up",
                    "metric": name,
                    "reason": f"Metric {name} significantly above threshold ({avg_value:.2f} > {metric.threshold_up:.2f})"
                })
            elif avg_value < metric.threshold_down * 0.5:
                recommendations.append({
                    "type": "consider_scale_down",
                    "metric": name,
                    "reason": f"Metric {name} significantly below threshold ({avg_value:.2f} < {metric.threshold_down:.2f})"
                })
                
        # Check for oscillating behavior
        recent_actions = [d["decision"] for d in self.scale_decisions if d["timestamp"] > time.time() - 1800]  # Last 30 minutes
        if len(recent_actions) > 4:
            up_count = recent_actions.count(ScalingDirection.UP)
            down_count = recent_actions.count(ScalingDirection.DOWN)
            
            if up_count > 0 and down_count > 0:
                recommendations.append({
                    "type": "review_thresholds",
                    "reason": "Detected oscillating scaling behavior - consider adjusting thresholds"
                })
                
        return {
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }