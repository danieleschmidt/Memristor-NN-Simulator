"""
Intelligent auto-scaling system for memristor neural network simulations.

Implements:
- Predictive scaling based on workload analysis
- Cost-optimal resource allocation
- Performance-based scaling decisions
- Multi-metric scaling policies
"""

import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import LoggingMixin
from ..utils.validators import validate_positive_number


class ScalingDirection(Enum):
    """Directions for scaling operations."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    QUEUE_LENGTH = "queue_length"
    COST_OPTIMIZATION = "cost_optimization"
    PREDICTIVE = "predictive"


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies."""
    name: str
    trigger: ScalingTrigger
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_instances: int = 1
    max_instances: int = 100
    cooldown_period_seconds: int = 300
    evaluation_periods: int = 3
    scale_up_adjustment: int = 1
    scale_down_adjustment: int = -1
    enabled: bool = True


@dataclass
class ResourceMetrics:
    """Container for resource utilization metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    throughput_ops_per_second: float
    average_latency_ms: float
    queue_length: int
    active_instances: int
    cost_per_hour: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "throughput_ops_per_second": self.throughput_ops_per_second,
            "average_latency_ms": self.average_latency_ms,
            "queue_length": self.queue_length,
            "active_instances": self.active_instances,
            "cost_per_hour": self.cost_per_hour
        }


class WorkloadPredictor:
    """Predictive model for workload forecasting."""
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.metrics_history: List[ResourceMetrics] = []
        self.prediction_models = {
            "linear_trend": self._linear_trend_model,
            "moving_average": self._moving_average_model,
            "exponential_smoothing": self._exponential_smoothing_model
        }
        
    def add_metrics(self, metrics: ResourceMetrics) -> None:
        """Add new metrics to history."""
        self.metrics_history.append(metrics)
        
        # Maintain rolling window
        if len(self.metrics_history) > self.history_window:
            self.metrics_history.pop(0)
    
    def predict_future_load(
        self, 
        minutes_ahead: int = 30,
        model: str = "exponential_smoothing"
    ) -> Dict[str, float]:
        """
        Predict future resource utilization.
        
        Args:
            minutes_ahead: How many minutes to predict ahead
            model: Prediction model to use
            
        Returns:
            Predicted resource metrics
        """
        if len(self.metrics_history) < 5:
            # Not enough history, return current metrics
            if self.metrics_history:
                current = self.metrics_history[-1]
                return {
                    "predicted_cpu_utilization": current.cpu_utilization,
                    "predicted_memory_utilization": current.memory_utilization,
                    "predicted_throughput": current.throughput_ops_per_second,
                    "confidence": 0.3
                }
            else:
                return {
                    "predicted_cpu_utilization": 0.5,
                    "predicted_memory_utilization": 0.5,
                    "predicted_throughput": 100.0,
                    "confidence": 0.1
                }
        
        prediction_func = self.prediction_models.get(model, self._exponential_smoothing_model)
        return prediction_func(minutes_ahead)
    
    def _linear_trend_model(self, minutes_ahead: int) -> Dict[str, float]:
        """Simple linear trend extrapolation."""
        if len(self.metrics_history) < 2:
            return self._moving_average_model(minutes_ahead)
        
        # Use last 10 points for trend calculation
        recent_metrics = self.metrics_history[-10:]
        timestamps = [m.timestamp for m in recent_metrics]
        cpu_utils = [m.cpu_utilization for m in recent_metrics]
        memory_utils = [m.memory_utilization for m in recent_metrics]
        throughputs = [m.throughput_ops_per_second for m in recent_metrics]
        
        # Calculate linear trends
        cpu_trend = self._calculate_trend(timestamps, cpu_utils)
        memory_trend = self._calculate_trend(timestamps, memory_utils)
        throughput_trend = self._calculate_trend(timestamps, throughputs)
        
        # Project forward
        future_time = time.time() + (minutes_ahead * 60)
        current = recent_metrics[-1]
        time_delta = future_time - current.timestamp
        
        predicted_cpu = current.cpu_utilization + (cpu_trend * time_delta)
        predicted_memory = current.memory_utilization + (memory_trend * time_delta)
        predicted_throughput = current.throughput_ops_per_second + (throughput_trend * time_delta)
        
        return {
            "predicted_cpu_utilization": max(0.0, min(1.0, predicted_cpu)),
            "predicted_memory_utilization": max(0.0, min(1.0, predicted_memory)),
            "predicted_throughput": max(0.0, predicted_throughput),
            "confidence": 0.7 if len(recent_metrics) >= 5 else 0.4
        }
    
    def _calculate_trend(self, x: List[float], y: List[float]) -> float:
        """Calculate linear trend slope."""
        if len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Linear regression slope
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-12:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _moving_average_model(self, minutes_ahead: int) -> Dict[str, float]:
        """Simple moving average prediction."""
        window_size = min(5, len(self.metrics_history))
        recent_metrics = self.metrics_history[-window_size:]
        
        avg_cpu = np.mean([m.cpu_utilization for m in recent_metrics])
        avg_memory = np.mean([m.memory_utilization for m in recent_metrics])
        avg_throughput = np.mean([m.throughput_ops_per_second for m in recent_metrics])
        
        return {
            "predicted_cpu_utilization": avg_cpu,
            "predicted_memory_utilization": avg_memory,
            "predicted_throughput": avg_throughput,
            "confidence": 0.6
        }
    
    def _exponential_smoothing_model(self, minutes_ahead: int) -> Dict[str, float]:
        """Exponential smoothing prediction."""
        alpha = 0.3  # Smoothing parameter
        
        if len(self.metrics_history) < 2:
            return self._moving_average_model(minutes_ahead)
        
        # Initialize with first value
        cpu_smooth = self.metrics_history[0].cpu_utilization
        memory_smooth = self.metrics_history[0].memory_utilization
        throughput_smooth = self.metrics_history[0].throughput_ops_per_second
        
        # Apply exponential smoothing
        for metrics in self.metrics_history[1:]:
            cpu_smooth = alpha * metrics.cpu_utilization + (1 - alpha) * cpu_smooth
            memory_smooth = alpha * metrics.memory_utilization + (1 - alpha) * memory_smooth
            throughput_smooth = alpha * metrics.throughput_ops_per_second + (1 - alpha) * throughput_smooth
        
        return {
            "predicted_cpu_utilization": cpu_smooth,
            "predicted_memory_utilization": memory_smooth,
            "predicted_throughput": throughput_smooth,
            "confidence": 0.8
        }


class AutoScaler(LoggingMixin):
    """Intelligent auto-scaling engine for memristor simulations."""
    
    def __init__(
        self,
        target_system,  # DistributedSimulator or similar
        scaling_policies: Optional[List[ScalingPolicy]] = None,
        enable_predictive_scaling: bool = True,
        metrics_collection_interval: float = 60.0
    ):
        """
        Initialize auto-scaler.
        
        Args:
            target_system: System to scale (DistributedSimulator)
            scaling_policies: List of scaling policies
            enable_predictive_scaling: Enable predictive scaling
            metrics_collection_interval: Interval for metrics collection (seconds)
        """
        super().__init__()
        self.target_system = target_system
        self.enable_predictive = enable_predictive_scaling
        self.metrics_interval = metrics_collection_interval
        
        # Initialize scaling policies
        if scaling_policies is None:
            scaling_policies = self._create_default_policies()
        self.scaling_policies = {policy.name: policy for policy in scaling_policies}
        
        # Initialize workload predictor
        self.predictor = WorkloadPredictor()
        
        # Scaling state tracking
        self.scaling_history: List[Dict] = []
        self.last_scaling_action = {}
        self.policy_cooldowns = {}
        
        # Metrics collection
        self.current_metrics = None
        self.metrics_collection_active = False
        self.metrics_thread = None
        
        # Cost optimization
        self.cost_model = self._initialize_cost_model()
        
        self.logger.info(f"Auto-scaler initialized with {len(self.scaling_policies)} policies")
    
    def _create_default_policies(self) -> List[ScalingPolicy]:
        """Create default scaling policies."""
        return [
            ScalingPolicy(
                name="cpu_scaling",
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=0.75,
                scale_down_threshold=0.25,
                cooldown_period_seconds=300
            ),
            ScalingPolicy(
                name="memory_scaling", 
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=0.80,
                scale_down_threshold=0.30,
                cooldown_period_seconds=300
            ),
            ScalingPolicy(
                name="throughput_scaling",
                trigger=ScalingTrigger.THROUGHPUT,
                scale_up_threshold=1000.0,  # ops/second
                scale_down_threshold=200.0,
                cooldown_period_seconds=180
            ),
            ScalingPolicy(
                name="latency_scaling",
                trigger=ScalingTrigger.LATENCY,
                scale_up_threshold=500.0,  # ms
                scale_down_threshold=100.0,
                cooldown_period_seconds=240,
                scale_up_adjustment=2  # More aggressive for latency
            )
        ]
    
    def _initialize_cost_model(self) -> Dict[str, float]:
        """Initialize cost model for different instance types."""
        return {
            "base_instance_cost_per_hour": 0.10,
            "memory_cost_per_gb_hour": 0.02,
            "cpu_cost_per_core_hour": 0.05,
            "network_cost_per_gb": 0.01,
            "storage_cost_per_gb_hour": 0.001
        }
    
    def start_monitoring(self) -> None:
        """Start continuous metrics collection and scaling evaluation."""
        if self.metrics_collection_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.metrics_collection_active = True
        
        def monitoring_loop():
            while self.metrics_collection_active:
                try:
                    # Collect current metrics
                    metrics = self._collect_current_metrics()
                    if metrics:
                        self.current_metrics = metrics
                        self.predictor.add_metrics(metrics)
                        
                        # Evaluate scaling policies
                        scaling_decision = self._evaluate_scaling_policies(metrics)
                        
                        if scaling_decision["action"] != ScalingDirection.NO_CHANGE:
                            self._execute_scaling_action(scaling_decision)
                    
                    time.sleep(self.metrics_interval)
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}")
                    time.sleep(self.metrics_interval * 2)  # Back off on error
        
        self.metrics_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.metrics_thread.start()
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop metrics collection and scaling evaluation."""
        self.metrics_collection_active = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=10.0)
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _collect_current_metrics(self) -> Optional[ResourceMetrics]:
        """Collect current system metrics."""
        try:
            # Get metrics from target system
            if hasattr(self.target_system, 'get_load_balance_metrics'):
                load_metrics = self.target_system.get_load_balance_metrics()
                
                # Simulate additional metrics (in practice, these would come from monitoring system)
                cpu_util = min(1.0, load_metrics.get("average_utilization", 0.5))
                memory_util = np.random.uniform(0.3, 0.8)  # Simulated
                
                throughput = 0.0
                latency = 100.0
                if hasattr(self.target_system, 'performance_history') and self.target_system.performance_history:
                    recent_perf = self.target_system.performance_history[-1]
                    throughput = recent_perf.get("throughput_ops_per_second", 0.0)
                    latency = recent_perf.get("total_simulation_time", 0.1) * 1000  # Convert to ms
                
                queue_length = max(0, int(np.random.poisson(5)))  # Simulated queue
                active_instances = load_metrics.get("active_nodes", 1)
                
                # Calculate cost
                cost_per_hour = self._calculate_current_cost(active_instances)
                
                return ResourceMetrics(
                    timestamp=time.time(),
                    cpu_utilization=cpu_util,
                    memory_utilization=memory_util,
                    throughput_ops_per_second=throughput,
                    average_latency_ms=latency,
                    queue_length=queue_length,
                    active_instances=active_instances,
                    cost_per_hour=cost_per_hour
                )
                
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return None
    
    def _calculate_current_cost(self, instances: int) -> float:
        """Calculate current hourly cost."""
        base_cost = instances * self.cost_model["base_instance_cost_per_hour"]
        
        # Add memory and CPU costs (estimated)
        memory_cost = instances * 8 * self.cost_model["memory_cost_per_gb_hour"]  # 8GB per instance
        cpu_cost = instances * 4 * self.cost_model["cpu_cost_per_core_hour"]  # 4 cores per instance
        
        return base_cost + memory_cost + cpu_cost
    
    def _evaluate_scaling_policies(self, current_metrics: ResourceMetrics) -> Dict[str, Any]:
        """Evaluate all scaling policies and determine best action."""
        policy_recommendations = []
        
        for policy_name, policy in self.scaling_policies.items():
            if not policy.enabled:
                continue
            
            # Check cooldown
            if policy_name in self.policy_cooldowns:
                cooldown_remaining = time.time() - self.policy_cooldowns[policy_name]
                if cooldown_remaining < policy.cooldown_period_seconds:
                    continue
            
            recommendation = self._evaluate_single_policy(policy, current_metrics)
            if recommendation["action"] != ScalingDirection.NO_CHANGE:
                policy_recommendations.append(recommendation)
        
        # Add predictive scaling if enabled
        if self.enable_predictive:
            predictive_recommendation = self._evaluate_predictive_scaling(current_metrics)
            if predictive_recommendation["action"] != ScalingDirection.NO_CHANGE:
                policy_recommendations.append(predictive_recommendation)
        
        # Choose best recommendation
        if not policy_recommendations:
            return {"action": ScalingDirection.NO_CHANGE, "reason": "No policies triggered"}
        
        # Prioritize scale-up over scale-down for performance
        scale_up_recs = [r for r in policy_recommendations if r["action"] == ScalingDirection.SCALE_UP]
        if scale_up_recs:
            return max(scale_up_recs, key=lambda x: x.get("urgency", 1.0))
        
        # Return highest priority scale-down
        scale_down_recs = [r for r in policy_recommendations if r["action"] == ScalingDirection.SCALE_DOWN]
        if scale_down_recs:
            return max(scale_down_recs, key=lambda x: x.get("confidence", 1.0))
        
        return {"action": ScalingDirection.NO_CHANGE, "reason": "No clear recommendation"}
    
    def _evaluate_single_policy(
        self, 
        policy: ScalingPolicy, 
        metrics: ResourceMetrics
    ) -> Dict[str, Any]:
        """Evaluate a single scaling policy."""
        
        # Get relevant metric value
        metric_value = self._get_metric_value(policy.trigger, metrics)
        
        # Determine scaling action
        action = ScalingDirection.NO_CHANGE
        urgency = 1.0
        confidence = 0.8
        
        if policy.trigger in [ScalingTrigger.CPU_UTILIZATION, ScalingTrigger.MEMORY_UTILIZATION]:
            # Higher values trigger scale-up
            if metric_value > policy.scale_up_threshold:
                action = ScalingDirection.SCALE_UP
                urgency = (metric_value - policy.scale_up_threshold) / (1.0 - policy.scale_up_threshold)
            elif metric_value < policy.scale_down_threshold:
                action = ScalingDirection.SCALE_DOWN
                confidence = (policy.scale_down_threshold - metric_value) / policy.scale_down_threshold
        
        elif policy.trigger == ScalingTrigger.THROUGHPUT:
            # Lower throughput (under load) triggers scale-up
            if metric_value < policy.scale_up_threshold:
                action = ScalingDirection.SCALE_UP
                urgency = (policy.scale_up_threshold - metric_value) / policy.scale_up_threshold
            elif metric_value > policy.scale_down_threshold * 3:  # Scale down if much higher than needed
                action = ScalingDirection.SCALE_DOWN
        
        elif policy.trigger == ScalingTrigger.LATENCY:
            # Higher latency triggers scale-up
            if metric_value > policy.scale_up_threshold:
                action = ScalingDirection.SCALE_UP
                urgency = min(2.0, metric_value / policy.scale_up_threshold)
            elif metric_value < policy.scale_down_threshold:
                action = ScalingDirection.SCALE_DOWN
        
        # Check instance limits
        current_instances = metrics.active_instances
        if action == ScalingDirection.SCALE_UP and current_instances >= policy.max_instances:
            action = ScalingDirection.NO_CHANGE
        elif action == ScalingDirection.SCALE_DOWN and current_instances <= policy.min_instances:
            action = ScalingDirection.NO_CHANGE
        
        return {
            "action": action,
            "policy_name": policy.name,
            "trigger": policy.trigger.value,
            "metric_value": metric_value,
            "threshold_used": policy.scale_up_threshold if action == ScalingDirection.SCALE_UP else policy.scale_down_threshold,
            "adjustment": policy.scale_up_adjustment if action == ScalingDirection.SCALE_UP else policy.scale_down_adjustment,
            "urgency": urgency,
            "confidence": confidence,
            "current_instances": current_instances
        }
    
    def _get_metric_value(self, trigger: ScalingTrigger, metrics: ResourceMetrics) -> float:
        """Extract relevant metric value for trigger."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return metrics.cpu_utilization
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return metrics.memory_utilization
        elif trigger == ScalingTrigger.THROUGHPUT:
            return metrics.throughput_ops_per_second
        elif trigger == ScalingTrigger.LATENCY:
            return metrics.average_latency_ms
        elif trigger == ScalingTrigger.QUEUE_LENGTH:
            return metrics.queue_length
        else:
            return 0.0
    
    def _evaluate_predictive_scaling(self, current_metrics: ResourceMetrics) -> Dict[str, Any]:
        """Evaluate predictive scaling based on forecasted load."""
        try:
            prediction = self.predictor.predict_future_load(minutes_ahead=30)
            
            if prediction["confidence"] < 0.5:
                return {"action": ScalingDirection.NO_CHANGE, "reason": "Low prediction confidence"}
            
            predicted_cpu = prediction["predicted_cpu_utilization"]
            predicted_memory = prediction["predicted_memory_utilization"]
            
            # Conservative thresholds for predictive scaling
            if predicted_cpu > 0.85 or predicted_memory > 0.85:
                return {
                    "action": ScalingDirection.SCALE_UP,
                    "policy_name": "predictive",
                    "trigger": "predictive_load_increase",
                    "predicted_cpu": predicted_cpu,
                    "predicted_memory": predicted_memory,
                    "confidence": prediction["confidence"],
                    "urgency": max(predicted_cpu, predicted_memory),
                    "adjustment": 1,
                    "current_instances": current_metrics.active_instances
                }
            elif predicted_cpu < 0.20 and predicted_memory < 0.20 and current_metrics.active_instances > 1:
                return {
                    "action": ScalingDirection.SCALE_DOWN,
                    "policy_name": "predictive",
                    "trigger": "predictive_load_decrease", 
                    "predicted_cpu": predicted_cpu,
                    "predicted_memory": predicted_memory,
                    "confidence": prediction["confidence"],
                    "adjustment": -1,
                    "current_instances": current_metrics.active_instances
                }
            
            return {"action": ScalingDirection.NO_CHANGE, "reason": "Predicted load within acceptable range"}
            
        except Exception as e:
            self.logger.error(f"Predictive scaling evaluation failed: {e}")
            return {"action": ScalingDirection.NO_CHANGE, "reason": f"Prediction error: {e}"}
    
    def _execute_scaling_action(self, scaling_decision: Dict[str, Any]) -> None:
        """Execute the recommended scaling action."""
        try:
            action = scaling_decision["action"]
            current_instances = scaling_decision.get("current_instances", 1)
            adjustment = scaling_decision.get("adjustment", 1)
            
            if action == ScalingDirection.SCALE_UP:
                target_instances = current_instances + abs(adjustment)
            elif action == ScalingDirection.SCALE_DOWN:
                target_instances = max(1, current_instances - abs(adjustment))
            else:
                return  # No action needed
            
            self.logger.info(f"Executing {action.value}: {current_instances} -> {target_instances} instances")
            
            # Execute scaling on target system
            if hasattr(self.target_system, 'scale_nodes'):
                scaling_result = self.target_system.scale_nodes(target_instances)
                
                # Record scaling action
                scaling_entry = {
                    "timestamp": time.time(),
                    "action": action.value,
                    "trigger": scaling_decision.get("trigger", "unknown"),
                    "policy": scaling_decision.get("policy_name", "unknown"),
                    "previous_instances": current_instances,
                    "target_instances": target_instances,
                    "actual_instances": scaling_result.get("current_nodes", target_instances),
                    "success": scaling_result.get("action", "failed") != "failed",
                    "cost_impact": self._calculate_cost_impact(current_instances, target_instances),
                    "decision_details": scaling_decision
                }
                
                self.scaling_history.append(scaling_entry)
                self.last_scaling_action = scaling_entry
                
                # Set cooldown for triggering policy
                policy_name = scaling_decision.get("policy_name")
                if policy_name and policy_name in self.scaling_policies:
                    self.policy_cooldowns[policy_name] = time.time()
                
                self.logger.info(f"Scaling completed: {scaling_result}")
                
            else:
                self.logger.error("Target system does not support scaling")
                
        except Exception as e:
            self.logger.error(f"Scaling execution failed: {e}")
    
    def _calculate_cost_impact(self, old_instances: int, new_instances: int) -> Dict[str, float]:
        """Calculate cost impact of scaling action."""
        old_cost = self._calculate_current_cost(old_instances)
        new_cost = self._calculate_current_cost(new_instances)
        
        return {
            "old_cost_per_hour": old_cost,
            "new_cost_per_hour": new_cost,
            "cost_change_per_hour": new_cost - old_cost,
            "cost_change_percentage": ((new_cost - old_cost) / old_cost * 100) if old_cost > 0 else 0
        }
    
    def force_scaling_action(
        self,
        target_instances: int,
        reason: str = "manual_override"
    ) -> Dict[str, Any]:
        """Force a specific scaling action (manual override)."""
        try:
            current_instances = self.current_metrics.active_instances if self.current_metrics else 1
            
            self.logger.info(f"Manual scaling override: {current_instances} -> {target_instances}")
            
            if hasattr(self.target_system, 'scale_nodes'):
                scaling_result = self.target_system.scale_nodes(target_instances)
                
                # Record manual scaling
                scaling_entry = {
                    "timestamp": time.time(),
                    "action": "manual_override",
                    "trigger": reason,
                    "policy": "manual",
                    "previous_instances": current_instances,
                    "target_instances": target_instances,
                    "actual_instances": scaling_result.get("current_nodes", target_instances),
                    "success": scaling_result.get("action", "failed") != "failed",
                    "cost_impact": self._calculate_cost_impact(current_instances, target_instances)
                }
                
                self.scaling_history.append(scaling_entry)
                return scaling_entry
                
            else:
                return {"success": False, "error": "Target system does not support scaling"}
                
        except Exception as e:
            self.logger.error(f"Manual scaling failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get current scaling recommendations without executing them."""
        if not self.current_metrics:
            return {"error": "No current metrics available"}
        
        # Evaluate all policies
        scaling_decision = self._evaluate_scaling_policies(self.current_metrics)
        
        # Add cost analysis
        current_instances = self.current_metrics.active_instances
        recommendations = {
            "current_status": {
                "instances": current_instances,
                "cpu_utilization": self.current_metrics.cpu_utilization,
                "memory_utilization": self.current_metrics.memory_utilization,
                "throughput": self.current_metrics.throughput_ops_per_second,
                "cost_per_hour": self.current_metrics.cost_per_hour
            },
            "recommendation": scaling_decision
        }
        
        # Add predictive analysis
        if self.enable_predictive:
            prediction = self.predictor.predict_future_load()
            recommendations["predictive_analysis"] = prediction
        
        # Add cost optimization analysis
        cost_analysis = self._analyze_cost_optimization()
        recommendations["cost_optimization"] = cost_analysis
        
        return recommendations
    
    def _analyze_cost_optimization(self) -> Dict[str, Any]:
        """Analyze cost optimization opportunities."""
        if not self.current_metrics:
            return {"error": "No metrics available"}
        
        current_instances = self.current_metrics.active_instances
        current_cost = self.current_metrics.cost_per_hour
        
        # Calculate costs for different instance counts
        cost_scenarios = {}
        for instances in range(max(1, current_instances - 2), min(current_instances + 3, 20)):
            cost_scenarios[instances] = {
                "cost_per_hour": self._calculate_current_cost(instances),
                "performance_estimate": self._estimate_performance(instances),
                "cost_efficiency": self._calculate_cost_efficiency(instances)
            }
        
        # Find most cost-effective configuration
        best_efficiency = max(cost_scenarios.values(), key=lambda x: x["cost_efficiency"])
        best_instances = [k for k, v in cost_scenarios.items() if v == best_efficiency][0]
        
        return {
            "current_instances": current_instances,
            "current_cost_per_hour": current_cost,
            "cost_optimal_instances": best_instances,
            "potential_savings_per_hour": current_cost - best_efficiency["cost_per_hour"],
            "cost_scenarios": cost_scenarios,
            "recommendation": "scale_down" if best_instances < current_instances else "scale_up" if best_instances > current_instances else "no_change"
        }
    
    def _estimate_performance(self, instances: int) -> Dict[str, float]:
        """Estimate performance for given number of instances."""
        # Simplified performance model
        base_throughput = 100.0  # ops/second per instance
        efficiency_factor = min(1.0, 1.2 - (instances * 0.05))  # Diminishing returns
        
        estimated_throughput = instances * base_throughput * efficiency_factor
        estimated_latency = max(50.0, 200.0 / instances)  # Inverse relationship
        
        return {
            "estimated_throughput": estimated_throughput,
            "estimated_latency_ms": estimated_latency
        }
    
    def _calculate_cost_efficiency(self, instances: int) -> float:
        """Calculate cost efficiency score for given instance count."""
        cost = self._calculate_current_cost(instances)
        performance = self._estimate_performance(instances)
        
        if cost <= 0:
            return 0.0
        
        # Efficiency = Performance / Cost
        throughput_score = performance["estimated_throughput"] / 100.0  # Normalize
        latency_score = 200.0 / performance["estimated_latency_ms"]  # Lower latency is better
        
        combined_performance = (throughput_score + latency_score) / 2
        efficiency = combined_performance / cost
        
        return efficiency
    
    def get_autoscaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive auto-scaling report."""
        try:
            return {
                "system_status": {
                    "monitoring_active": self.metrics_collection_active,
                    "predictive_scaling_enabled": self.enable_predictive,
                    "current_metrics": self.current_metrics.to_dict() if self.current_metrics else {},
                    "policies_count": len(self.scaling_policies)
                },
                "scaling_policies": {
                    name: {
                        "enabled": policy.enabled,
                        "trigger": policy.trigger.value,
                        "scale_up_threshold": policy.scale_up_threshold,
                        "scale_down_threshold": policy.scale_down_threshold,
                        "cooldown_remaining": max(0, policy.cooldown_period_seconds - 
                                                (time.time() - self.policy_cooldowns.get(name, 0)))
                    }
                    for name, policy in self.scaling_policies.items()
                },
                "recent_scaling_history": self.scaling_history[-10:],  # Last 10 actions
                "performance_summary": {
                    "total_scaling_actions": len(self.scaling_history),
                    "scale_up_actions": len([a for a in self.scaling_history if "scale_up" in a.get("action", "")]),
                    "scale_down_actions": len([a for a in self.scaling_history if "scale_down" in a.get("action", "")]),
                    "average_response_time": np.mean([
                        a.get("response_time", 0) for a in self.scaling_history if "response_time" in a
                    ]) if self.scaling_history else 0
                },
                "cost_analysis": {
                    "current_hourly_cost": self.current_metrics.cost_per_hour if self.current_metrics else 0,
                    "cost_optimization": self._analyze_cost_optimization() if self.current_metrics else {}
                },
                "recommendations": self.get_scaling_recommendations() if self.current_metrics else {}
            }
            
        except Exception as e:
            self.logger.error(f"Auto-scaling report generation failed: {e}")
            return {"error": str(e)}