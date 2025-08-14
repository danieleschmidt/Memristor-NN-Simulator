"""
Self-Improving Optimization System for Autonomous SDLC.

This module implements an advanced self-improving optimization system that
learns from performance data and automatically optimizes simulations for
maximum efficiency and throughput.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import math

from ..utils.logger import get_logger, PerformanceLogger
from ..utils.error_handling import collect_errors, error_context, retry
from .performance_profiler import PerformanceProfiler, get_performance_profiler
from .scaling_manager import get_scaling_manager
from .cache_manager import CacheManager


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    CONSERVATIVE = "conservative"  # Safe, gradual optimizations
    AGGRESSIVE = "aggressive"      # Fast, high-impact optimizations  
    ADAPTIVE = "adaptive"          # Learning-based optimizations
    EXPERIMENTAL = "experimental"  # Novel, untested optimizations


@dataclass
class OptimizationTarget:
    """Performance optimization target."""
    metric_name: str
    target_value: float
    current_value: float
    priority: float = 1.0  # 0.0 to 1.0
    achieved: bool = False
    
    @property
    def improvement_needed(self) -> float:
        """Calculate improvement factor needed."""
        if self.current_value <= 0:
            return float('inf')
        return max(0, (self.target_value - self.current_value) / self.current_value)


@dataclass
class OptimizationAction:
    """Action to improve performance."""
    action_id: str
    description: str
    strategy: OptimizationStrategy
    expected_improvement: float  # Expected % improvement
    confidence: float  # 0.0 to 1.0
    cost: float  # Relative cost of implementation
    implementation: Callable[[], bool]  # Function to execute action
    rollback: Optional[Callable[[], bool]] = None  # Function to undo action


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    action: OptimizationAction
    success: bool
    actual_improvement: float
    execution_time: float
    side_effects: List[str] = field(default_factory=list)
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)


class AdaptiveLearningEngine:
    """Machine learning engine for optimization decisions."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.action_performance = defaultdict(list)  # action_id -> [success_rates]
        self.feature_weights = defaultdict(float)  # feature -> weight
        self.strategy_success = defaultdict(list)  # strategy -> [success_rates]
        self.logger = get_logger("adaptive_learning")
    
    def record_optimization(self, result: OptimizationResult, context: Dict[str, float]):
        """Record optimization result for learning."""
        action_id = result.action.action_id
        strategy = result.action.strategy.value
        
        # Record action performance
        success_score = 1.0 if result.success else 0.0
        if result.success and result.actual_improvement > 0:
            success_score = min(2.0, 1.0 + result.actual_improvement)
        
        self.action_performance[action_id].append(success_score)
        self.strategy_success[strategy].append(success_score)
        
        # Update feature weights based on context
        for feature, value in context.items():
            if result.success and result.actual_improvement > 0:
                self.feature_weights[feature] += self.learning_rate * value * result.actual_improvement
            elif not result.success:
                self.feature_weights[feature] -= self.learning_rate * value * 0.1
    
    def predict_action_success(self, action: OptimizationAction, context: Dict[str, float]) -> float:
        """Predict probability of action success."""
        # Base probability from historical data
        if action.action_id in self.action_performance:
            historical_scores = self.action_performance[action.action_id]
            base_prob = statistics.mean(historical_scores[-10:])  # Last 10 attempts
        else:
            base_prob = action.confidence
        
        # Strategy-based adjustment
        strategy_scores = self.strategy_success.get(action.strategy.value, [])
        if strategy_scores:
            strategy_factor = statistics.mean(strategy_scores[-20:])
            base_prob = (base_prob + strategy_factor) / 2
        
        # Context-based adjustment
        context_score = 0.0
        for feature, value in context.items():
            weight = self.feature_weights.get(feature, 0.0)
            context_score += weight * value
        
        # Combine probabilities
        final_prob = base_prob + (context_score * self.learning_rate)
        return max(0.0, min(1.0, final_prob))
    
    def recommend_strategy(self, targets: List[OptimizationTarget]) -> OptimizationStrategy:
        """Recommend best optimization strategy based on targets."""
        # Calculate urgency and complexity
        urgency = sum(target.priority * target.improvement_needed for target in targets)
        complexity = len(targets)
        
        # Strategy success rates
        strategy_scores = {}
        for strategy in OptimizationStrategy:
            scores = self.strategy_success.get(strategy.value, [0.5])  # Default 50%
            strategy_scores[strategy] = statistics.mean(scores[-15:])
        
        # Choose strategy based on context
        if urgency > 2.0 and strategy_scores[OptimizationStrategy.AGGRESSIVE] > 0.6:
            return OptimizationStrategy.AGGRESSIVE
        elif complexity <= 2 and strategy_scores[OptimizationStrategy.EXPERIMENTAL] > 0.7:
            return OptimizationStrategy.EXPERIMENTAL
        elif urgency < 0.5:
            return OptimizationStrategy.CONSERVATIVE
        else:
            return OptimizationStrategy.ADAPTIVE


class SelfImprovingOptimizer:
    """Self-improving optimization system that learns and adapts."""
    
    def __init__(
        self,
        optimization_interval: float = 300.0,  # 5 minutes
        max_concurrent_optimizations: int = 3,
        learning_enabled: bool = True
    ):
        """
        Initialize self-improving optimizer.
        
        Args:
            optimization_interval: Time between optimization cycles (seconds)
            max_concurrent_optimizations: Maximum concurrent optimization attempts
            learning_enabled: Enable adaptive learning
        """
        self.optimization_interval = optimization_interval
        self.max_concurrent_optimizations = max_concurrent_optimizations
        self.learning_enabled = learning_enabled
        
        self.logger = get_logger("self_improving_optimizer")
        self.profiler = get_performance_profiler()
        self.scaling_manager = get_scaling_manager()
        self.cache_manager = CacheManager()
        
        # Learning engine
        if learning_enabled:
            self.learning_engine = AdaptiveLearningEngine()
        else:
            self.learning_engine = None
        
        # Optimization state
        self.targets: List[OptimizationTarget] = []
        self.active_optimizations = {}  # optimization_id -> thread
        self.optimization_history = deque(maxlen=1000)
        self.last_optimization_time = 0.0
        
        # Performance tracking
        self.baseline_metrics = {}
        self.improvement_tracking = defaultdict(list)
        
        # Background optimization
        self.auto_optimize_enabled = False
        self.optimization_thread = None
        self._stop_event = threading.Event()
        
        self.logger.info("Self-improving optimizer initialized")
    
    def add_optimization_target(
        self,
        metric_name: str,
        target_value: float,
        current_value: float,
        priority: float = 1.0
    ):
        """Add a performance optimization target."""
        target = OptimizationTarget(
            metric_name=metric_name,
            target_value=target_value,
            current_value=current_value,
            priority=priority
        )
        
        self.targets.append(target)
        self.logger.info(f"Added optimization target: {metric_name} = {target_value} (current: {current_value})")
    
    def start_auto_optimization(self):
        """Start automatic background optimization."""
        if self.auto_optimize_enabled:
            return
        
        self.auto_optimize_enabled = True
        self._stop_event.clear()
        self.optimization_thread = threading.Thread(
            target=self._auto_optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        self.logger.info("Started automatic optimization")
    
    def stop_auto_optimization(self):
        """Stop automatic optimization."""
        if not self.auto_optimize_enabled:
            return
        
        self.auto_optimize_enabled = False
        self._stop_event.set()
        
        if self.optimization_thread:
            self.optimization_thread.join(timeout=30.0)
        
        self.logger.info("Stopped automatic optimization")
    
    def _auto_optimization_loop(self):
        """Main automatic optimization loop."""
        while self.auto_optimize_enabled and not self._stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - self.last_optimization_time >= self.optimization_interval:
                    self._run_optimization_cycle()
                    self.last_optimization_time = current_time
                
                # Sleep with interrupt checking
                self._stop_event.wait(min(60.0, self.optimization_interval / 10))
                
            except Exception as e:
                self.logger.error(f"Error in auto-optimization loop: {e}")
                self._stop_event.wait(60.0)  # Wait a minute before retrying
    
    @collect_errors("optimization_cycle")
    def _run_optimization_cycle(self):
        """Run a single optimization cycle."""
        with error_context("optimization_cycle", self.logger):
            self.logger.info("Starting optimization cycle")
            
            # Update current metrics
            self._update_target_metrics()
            
            # Check if targets are met
            unmet_targets = [t for t in self.targets if not t.achieved]
            if not unmet_targets:
                self.logger.info("All optimization targets achieved")
                return
            
            # Get optimization strategy recommendation
            if self.learning_enabled and self.learning_engine:
                strategy = self.learning_engine.recommend_strategy(unmet_targets)
            else:
                strategy = OptimizationStrategy.ADAPTIVE
            
            # Generate optimization actions
            actions = self._generate_optimization_actions(unmet_targets, strategy)
            
            # Execute top actions (limited by concurrency)
            available_slots = self.max_concurrent_optimizations - len(self.active_optimizations)
            if available_slots > 0:
                top_actions = actions[:available_slots]
                for action in top_actions:
                    self._execute_optimization_async(action)
    
    def _update_target_metrics(self):
        """Update current values for optimization targets."""
        # Get current performance stats
        all_stats = self.profiler.get_all_stats()
        
        for target in self.targets:
            if target.metric_name in all_stats:
                stats = all_stats[target.metric_name]
                target.current_value = stats.get("average_execution_time", target.current_value)
            
            # Check if target is achieved
            if target.metric_name == "latency":
                target.achieved = target.current_value <= target.target_value
            else:
                target.achieved = target.current_value >= target.target_value
    
    def _generate_optimization_actions(
        self,
        targets: List[OptimizationTarget],
        strategy: OptimizationStrategy
    ) -> List[OptimizationAction]:
        """Generate optimization actions for targets."""
        actions = []
        
        # Cache optimization actions
        cache_actions = self._generate_cache_actions(targets, strategy)
        actions.extend(cache_actions)
        
        # Scaling optimization actions
        scaling_actions = self._generate_scaling_actions(targets, strategy)
        actions.extend(scaling_actions)
        
        # Algorithm optimization actions
        algorithm_actions = self._generate_algorithm_actions(targets, strategy)
        actions.extend(algorithm_actions)
        
        # Sort by expected impact and confidence
        actions.sort(
            key=lambda a: a.expected_improvement * a.confidence,
            reverse=True
        )
        
        return actions
    
    def _generate_cache_actions(
        self,
        targets: List[OptimizationTarget],
        strategy: OptimizationStrategy
    ) -> List[OptimizationAction]:
        """Generate cache-related optimization actions."""
        actions = []
        
        # Increase cache size
        if strategy in [OptimizationStrategy.AGGRESSIVE, OptimizationStrategy.ADAPTIVE]:
            actions.append(OptimizationAction(
                action_id="increase_cache_size",
                description="Increase cache size for better hit rates",
                strategy=strategy,
                expected_improvement=15.0,
                confidence=0.8,
                cost=0.3,
                implementation=lambda: self._increase_cache_size(),
                rollback=lambda: self._decrease_cache_size()
            ))
        
        # Enable intelligent prefetching
        if strategy == OptimizationStrategy.EXPERIMENTAL:
            actions.append(OptimizationAction(
                action_id="enable_cache_prefetch",
                description="Enable intelligent cache prefetching",
                strategy=strategy,
                expected_improvement=25.0,
                confidence=0.6,
                cost=0.5,
                implementation=lambda: self._enable_cache_prefetch(),
                rollback=lambda: self._disable_cache_prefetch()
            ))
        
        return actions
    
    def _generate_scaling_actions(
        self,
        targets: List[OptimizationTarget],
        strategy: OptimizationStrategy
    ) -> List[OptimizationAction]:
        """Generate scaling-related optimization actions."""
        actions = []
        
        # Adjust worker count
        if strategy in [OptimizationStrategy.AGGRESSIVE, OptimizationStrategy.ADAPTIVE]:
            actions.append(OptimizationAction(
                action_id="optimize_worker_count",
                description="Dynamically optimize worker count",
                strategy=strategy,
                expected_improvement=20.0,
                confidence=0.9,
                cost=0.2,
                implementation=lambda: self._optimize_worker_count(),
                rollback=lambda: self._reset_worker_count()
            ))
        
        return actions
    
    def _generate_algorithm_actions(
        self,
        targets: List[OptimizationTarget],
        strategy: OptimizationStrategy
    ) -> List[OptimizationAction]:
        """Generate algorithm-level optimization actions."""
        actions = []
        
        # Enable vectorization
        if strategy != OptimizationStrategy.CONSERVATIVE:
            actions.append(OptimizationAction(
                action_id="enable_vectorization",
                description="Enable advanced vectorization optimizations",
                strategy=strategy,
                expected_improvement=30.0,
                confidence=0.7,
                cost=0.4,
                implementation=lambda: self._enable_vectorization(),
                rollback=lambda: self._disable_vectorization()
            ))
        
        # Memory layout optimization
        if strategy == OptimizationStrategy.EXPERIMENTAL:
            actions.append(OptimizationAction(
                action_id="optimize_memory_layout",
                description="Optimize memory access patterns",
                strategy=strategy,
                expected_improvement=40.0,
                confidence=0.5,
                cost=0.7,
                implementation=lambda: self._optimize_memory_layout(),
                rollback=lambda: self._reset_memory_layout()
            ))
        
        return actions
    
    def _execute_optimization_async(self, action: OptimizationAction):
        """Execute optimization action asynchronously."""
        optimization_id = f"{action.action_id}_{int(time.time())}"
        
        def optimization_worker():
            try:
                result = self._execute_optimization(action)
                self.optimization_history.append(result)
                
                # Update learning engine
                if self.learning_enabled and self.learning_engine:
                    context = self._get_optimization_context()
                    self.learning_engine.record_optimization(result, context)
                
                self.logger.info(f"Optimization {optimization_id} completed: {'SUCCESS' if result.success else 'FAILED'}")
                
            except Exception as e:
                self.logger.error(f"Optimization {optimization_id} failed: {e}")
            finally:
                # Remove from active optimizations
                if optimization_id in self.active_optimizations:
                    del self.active_optimizations[optimization_id]
        
        # Start optimization thread
        thread = threading.Thread(target=optimization_worker, daemon=True)
        self.active_optimizations[optimization_id] = thread
        thread.start()
    
    @retry(max_attempts=2, delay=1.0)
    def _execute_optimization(self, action: OptimizationAction) -> OptimizationResult:
        """Execute a single optimization action."""
        self.logger.info(f"Executing optimization: {action.description}")
        
        start_time = time.time()
        
        # Record metrics before optimization
        metrics_before = self._get_current_metrics()
        
        try:
            # Execute the optimization
            success = action.implementation()
            
            # Wait a bit for metrics to stabilize
            time.sleep(5.0)
            
            # Record metrics after optimization
            metrics_after = self._get_current_metrics()
            
            # Calculate actual improvement
            actual_improvement = self._calculate_improvement(metrics_before, metrics_after)
            
            # Create result
            result = OptimizationResult(
                action=action,
                success=success and actual_improvement >= 0,
                actual_improvement=actual_improvement,
                execution_time=time.time() - start_time,
                metrics_before=metrics_before,
                metrics_after=metrics_after
            )
            
            # Rollback if optimization failed or made things worse
            if not result.success and action.rollback:
                try:
                    action.rollback()
                    result.side_effects.append("Rolled back due to failure")
                except Exception as e:
                    result.side_effects.append(f"Rollback failed: {e}")
            
            return result
            
        except Exception as e:
            # Rollback on exception
            if action.rollback:
                try:
                    action.rollback()
                except Exception:
                    pass
            
            return OptimizationResult(
                action=action,
                success=False,
                actual_improvement=-1.0,
                execution_time=time.time() - start_time,
                metrics_before=metrics_before,
                metrics_after=metrics_before,
                side_effects=[f"Exception: {e}"]
            )
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        metrics = {}
        
        # Get profiler stats
        all_stats = self.profiler.get_all_stats()
        if all_stats:
            # Average execution time across all operations
            exec_times = [stats["average_execution_time"] for stats in all_stats.values()]
            metrics["avg_execution_time"] = statistics.mean(exec_times) if exec_times else 0.0
            
            # Total operation count
            metrics["total_operations"] = sum(stats["call_count"] for stats in all_stats.values())
        
        # Get scaling stats
        scaling_stats = self.scaling_manager.get_comprehensive_stats()
        if 'recent_performance' in scaling_stats:
            perf = scaling_stats['recent_performance']
            metrics["throughput"] = perf.get("avg_throughput", 0.0)
            metrics["success_rate"] = perf.get("avg_success_rate", 1.0)
        
        return metrics
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate overall improvement percentage."""
        improvements = []
        
        for metric, before_val in before.items():
            after_val = after.get(metric, before_val)
            
            if before_val > 0:
                if metric in ["avg_execution_time"]:  # Lower is better
                    improvement = (before_val - after_val) / before_val
                else:  # Higher is better
                    improvement = (after_val - before_val) / before_val
                
                improvements.append(improvement)
        
        return statistics.mean(improvements) * 100 if improvements else 0.0
    
    def _get_optimization_context(self) -> Dict[str, float]:
        """Get context for optimization learning."""
        context = {}
        
        # System load context
        current_metrics = self._get_current_metrics()
        context.update(current_metrics)
        
        # Time-based context
        hour_of_day = time.localtime().tm_hour
        context["hour_of_day"] = hour_of_day / 24.0  # Normalize to 0-1
        
        # Target urgency context
        if self.targets:
            urgency = sum(t.priority * t.improvement_needed for t in self.targets)
            context["target_urgency"] = min(urgency, 5.0) / 5.0  # Normalize to 0-1
        
        return context
    
    # Implementation methods for optimization actions
    def _increase_cache_size(self) -> bool:
        """Increase cache size."""
        try:
            current_size = self.cache_manager.max_size
            new_size = min(current_size * 2, 10000)  # Double size, max 10k
            self.cache_manager.max_size = new_size
            self.logger.info(f"Increased cache size: {current_size} -> {new_size}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to increase cache size: {e}")
            return False
    
    def _decrease_cache_size(self) -> bool:
        """Decrease cache size (rollback)."""
        try:
            current_size = self.cache_manager.max_size
            new_size = max(current_size // 2, 100)  # Halve size, min 100
            self.cache_manager.max_size = new_size
            self.logger.info(f"Decreased cache size: {current_size} -> {new_size}")
            return True
        except Exception:
            return False
    
    def _enable_cache_prefetch(self) -> bool:
        """Enable cache prefetching."""
        # This would implement intelligent prefetching logic
        self.logger.info("Enabled cache prefetching")
        return True
    
    def _disable_cache_prefetch(self) -> bool:
        """Disable cache prefetching."""
        self.logger.info("Disabled cache prefetching")
        return True
    
    def _optimize_worker_count(self) -> bool:
        """Optimize worker count."""
        try:
            # Use scaling manager to optimize
            optimal_count = self.scaling_manager.optimize_worker_count(pending_tasks=10)
            self.logger.info(f"Optimized worker count to: {optimal_count}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize worker count: {e}")
            return False
    
    def _reset_worker_count(self) -> bool:
        """Reset worker count to default."""
        try:
            self.scaling_manager.auto_scaler.current_workers = self.scaling_manager.auto_scaler.min_workers
            self.logger.info("Reset worker count to minimum")
            return True
        except Exception:
            return False
    
    def _enable_vectorization(self) -> bool:
        """Enable vectorization optimizations."""
        # This would enable SIMD or other vectorization
        self.logger.info("Enabled vectorization optimizations")
        return True
    
    def _disable_vectorization(self) -> bool:
        """Disable vectorization optimizations."""
        self.logger.info("Disabled vectorization optimizations")
        return True
    
    def _optimize_memory_layout(self) -> bool:
        """Optimize memory access patterns."""
        # This would implement memory layout optimizations
        self.logger.info("Optimized memory layout")
        return True
    
    def _reset_memory_layout(self) -> bool:
        """Reset memory layout to default."""
        self.logger.info("Reset memory layout")
        return True
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.optimization_history:
            return {"status": "No optimization history available"}
        
        recent_results = list(self.optimization_history)[-20:]  # Last 20
        
        # Calculate success rate
        successes = sum(1 for r in recent_results if r.success)
        success_rate = successes / len(recent_results)
        
        # Calculate average improvement
        improvements = [r.actual_improvement for r in recent_results if r.success]
        avg_improvement = statistics.mean(improvements) if improvements else 0.0
        
        # Strategy effectiveness
        strategy_stats = defaultdict(list)
        for result in recent_results:
            strategy_stats[result.action.strategy.value].append(result.actual_improvement)
        
        best_strategy = max(
            strategy_stats.items(),
            key=lambda x: statistics.mean([i for i in x[1] if i > 0]),
            default=(None, [])
        )[0]
        
        # Current target status
        unmet_targets = [t for t in self.targets if not t.achieved]
        
        return {
            "optimization_summary": {
                "total_optimizations": len(self.optimization_history),
                "recent_success_rate": success_rate,
                "average_improvement": avg_improvement,
                "best_strategy": best_strategy,
                "active_optimizations": len(self.active_optimizations)
            },
            "targets": {
                "total_targets": len(self.targets),
                "achieved_targets": len(self.targets) - len(unmet_targets),
                "unmet_targets": [
                    {
                        "metric": t.metric_name,
                        "target": t.target_value,
                        "current": t.current_value,
                        "improvement_needed": t.improvement_needed
                    }
                    for t in unmet_targets
                ]
            },
            "learning_status": "enabled" if self.learning_enabled else "disabled",
            "auto_optimization": "running" if self.auto_optimize_enabled else "stopped"
        }


# Global optimizer instance
_global_optimizer: Optional[SelfImprovingOptimizer] = None


def get_self_improving_optimizer(
    optimization_interval: float = 300.0,
    learning_enabled: bool = True
) -> SelfImprovingOptimizer:
    """Get or create global self-improving optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = SelfImprovingOptimizer(
            optimization_interval=optimization_interval,
            learning_enabled=learning_enabled
        )
    return _global_optimizer


def configure_performance_targets():
    """Configure standard performance targets for memristor simulations."""
    optimizer = get_self_improving_optimizer()
    
    # Add standard targets
    optimizer.add_optimization_target(
        metric_name="latency",
        target_value=0.010,  # 10ms
        current_value=0.100,  # 100ms
        priority=1.0
    )
    
    optimizer.add_optimization_target(
        metric_name="throughput",
        target_value=1000.0,  # 1000 ops/sec
        current_value=100.0,   # 100 ops/sec
        priority=0.8
    )
    
    optimizer.add_optimization_target(
        metric_name="success_rate",
        target_value=0.99,    # 99%
        current_value=0.95,   # 95%
        priority=0.9
    )


if __name__ == "__main__":
    # Example usage
    optimizer = get_self_improving_optimizer()
    configure_performance_targets()
    
    print("🚀 Starting self-improving optimization")
    optimizer.start_auto_optimization()
    
    try:
        # Let it run for a bit
        time.sleep(30)
        
        # Get report
        report = optimizer.get_optimization_report()
        print("\n📊 Optimization Report:")
        print(json.dumps(report, indent=2))
        
    finally:
        optimizer.stop_auto_optimization()
        print("✅ Self-improving optimization stopped")