"""
Load balancing and health checking for pipeline guard services
"""

import time
import random
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin" 
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    HEALTH_AWARE = "health_aware"


class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"


@dataclass
class ServiceInstance:
    """Represents a service instance"""
    id: str
    address: str
    port: int
    weight: float = 1.0
    status: ServiceStatus = ServiceStatus.UNKNOWN
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: Optional[datetime] = None
    average_response_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @property
    def endpoint(self) -> str:
        """Get full endpoint URL"""
        return f"{self.address}:{self.port}"
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.failed_requests / self.total_requests)
        
    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy"""
        return self.status == ServiceStatus.HEALTHY


class HealthChecker:
    """
    Health checking system for service instances
    """
    
    def __init__(self, check_interval: int = 30, timeout: int = 5):
        self.check_interval = check_interval
        self.timeout = timeout
        self.health_checks: Dict[str, Callable] = {}
        self.checking = False
        self.check_thread = None
        self.logger = logging.getLogger(__name__)
        
    def register_health_check(self, service_type: str, check_function: Callable):
        """Register health check function for service type"""
        self.health_checks[service_type] = check_function
        
    def start_checking(self):
        """Start health checking"""
        if self.checking:
            return
            
        self.checking = True
        self.check_thread = threading.Thread(target=self._health_check_loop)
        self.check_thread.start()
        self.logger.info("Health checking started")
        
    def stop_checking(self):
        """Stop health checking"""
        self.checking = False
        if self.check_thread:
            self.check_thread.join()
        self.logger.info("Health checking stopped")
        
    def _health_check_loop(self):
        """Main health checking loop"""
        while self.checking:
            try:
                # Health checks would be performed here
                # This is a placeholder for the actual health checking logic
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                
    def check_instance_health(self, instance: ServiceInstance, 
                            service_type: str = "default") -> bool:
        """Check health of a specific instance"""
        if service_type not in self.health_checks:
            # Default health check - simple connection test
            return self._default_health_check(instance)
            
        try:
            health_check_fn = self.health_checks[service_type]
            is_healthy = health_check_fn(instance)
            
            instance.last_health_check = datetime.now()
            previous_status = instance.status
            instance.status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
            
            if previous_status != instance.status:
                self.logger.info(f"Instance {instance.id} status changed: {previous_status} -> {instance.status}")
                
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed for {instance.id}: {e}")
            instance.status = ServiceStatus.UNHEALTHY
            return False
            
    def _default_health_check(self, instance: ServiceInstance) -> bool:
        """Default health check implementation"""
        # Simple availability check - in real implementation would make HTTP request
        import socket
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((instance.address, instance.port))
            sock.close()
            return result == 0
        except Exception:
            return False


class LoadBalancer:
    """
    Intelligent load balancer for pipeline guard services
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_AWARE):
        self.strategy = strategy
        self.instances: Dict[str, ServiceInstance] = {}
        self.instance_groups: Dict[str, List[str]] = defaultdict(list)
        self.current_index = 0
        self.health_checker = HealthChecker()
        self.request_history = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.response_times = deque(maxlen=100)
        
    def add_instance(self, instance: ServiceInstance, group: str = "default"):
        """Add service instance to load balancer"""
        self.instances[instance.id] = instance
        self.instance_groups[group].append(instance.id)
        self.logger.info(f"Added instance {instance.id} to group {group}")
        
    def remove_instance(self, instance_id: str):
        """Remove service instance from load balancer"""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            
            # Mark as draining and wait for connections to finish
            instance.status = ServiceStatus.DRAINING
            
            # Remove from groups
            for group, instance_list in self.instance_groups.items():
                if instance_id in instance_list:
                    instance_list.remove(instance_id)
                    
            del self.instances[instance_id]
            self.logger.info(f"Removed instance {instance_id}")
            
    def get_instance(self, group: str = "default", 
                    exclude_unhealthy: bool = True) -> Optional[ServiceInstance]:
        """Get next instance based on load balancing strategy"""
        available_instances = self._get_available_instances(group, exclude_unhealthy)
        
        if not available_instances:
            self.logger.warning(f"No available instances in group {group}")
            return None
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random_selection(available_instances)
        elif self.strategy == LoadBalancingStrategy.HEALTH_AWARE:
            return self._health_aware_selection(available_instances)
        else:
            return available_instances[0]  # Fallback
            
    def _get_available_instances(self, group: str, 
                               exclude_unhealthy: bool) -> List[ServiceInstance]:
        """Get list of available instances for load balancing"""
        if group not in self.instance_groups:
            return []
            
        instances = []
        for instance_id in self.instance_groups[group]:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                
                if exclude_unhealthy and not instance.is_healthy:
                    continue
                    
                if instance.status == ServiceStatus.DRAINING:
                    continue
                    
                instances.append(instance)
                
        return instances
        
    def _round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        selected = instances[self.current_index % len(instances)]
        self.current_index += 1
        return selected
        
    def _weighted_round_robin_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection"""
        total_weight = sum(instance.weight for instance in instances)
        
        if total_weight == 0:
            return self._round_robin_selection(instances)
            
        # Create weighted list
        weighted_instances = []
        for instance in instances:
            weight_count = max(1, int(instance.weight * 10))  # Scale weights
            weighted_instances.extend([instance] * weight_count)
            
        return self._round_robin_selection(weighted_instances)
        
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections"""
        return min(instances, key=lambda x: x.current_connections)
        
    def _least_response_time_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with lowest average response time"""
        return min(instances, key=lambda x: x.average_response_time)
        
    def _random_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection"""
        return random.choice(instances)
        
    def _health_aware_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Health-aware selection combining multiple factors"""
        scored_instances = []
        
        for instance in instances:
            # Calculate composite score (lower is better)
            score = 0
            
            # Connection count factor (normalized)
            max_connections = max((i.current_connections for i in instances), default=1)
            connection_factor = instance.current_connections / max_connections if max_connections > 0 else 0
            score += connection_factor * 0.3
            
            # Response time factor (normalized)
            max_response_time = max((i.average_response_time for i in instances), default=1)
            response_time_factor = instance.average_response_time / max_response_time if max_response_time > 0 else 0
            score += response_time_factor * 0.3
            
            # Success rate factor (inverted - higher success rate = lower score)
            score += (1.0 - instance.success_rate) * 0.2
            
            # Weight factor (inverted - higher weight = lower score)
            score += (1.0 / instance.weight) * 0.2
            
            scored_instances.append((score, instance))
            
        # Select instance with lowest score
        scored_instances.sort(key=lambda x: x[0])
        return scored_instances[0][1]
        
    def record_request(self, instance: ServiceInstance, response_time: float, 
                      success: bool):
        """Record request metrics for an instance"""
        self.total_requests += 1
        instance.total_requests += 1
        
        if not success:
            self.failed_requests += 1
            instance.failed_requests += 1
            
        # Update response time
        self.response_times.append(response_time)
        
        # Update instance average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        instance.average_response_time = (
            alpha * response_time + 
            (1 - alpha) * instance.average_response_time
        )
        
        # Record in history
        self.request_history.append({
            "instance_id": instance.id,
            "response_time": response_time,
            "success": success,
            "timestamp": time.time()
        })
        
    def get_instance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive instance statistics"""
        instance_stats = {}
        
        for instance_id, instance in self.instances.items():
            instance_stats[instance_id] = {
                "endpoint": instance.endpoint,
                "status": instance.status.value,
                "weight": instance.weight,
                "current_connections": instance.current_connections,
                "total_requests": instance.total_requests,
                "failed_requests": instance.failed_requests,
                "success_rate": instance.success_rate,
                "average_response_time": instance.average_response_time,
                "last_health_check": instance.last_health_check.isoformat() if instance.last_health_check else None
            }
            
        return {
            "strategy": self.strategy.value,
            "total_instances": len(self.instances),
            "healthy_instances": len([i for i in self.instances.values() if i.is_healthy]),
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "overall_success_rate": 1.0 - (self.failed_requests / self.total_requests) if self.total_requests > 0 else 1.0,
            "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0.0,
            "instances": instance_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    def start_health_checking(self):
        """Start health checking for all instances"""
        self.health_checker.start_checking()
        
    def stop_health_checking(self):
        """Stop health checking"""
        self.health_checker.stop_checking()
        
    def force_health_check(self, group: str = "default"):
        """Force health check for all instances in group"""
        for instance_id in self.instance_groups.get(group, []):
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                self.health_checker.check_instance_health(instance)
                
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across instances"""
        total_requests = sum(instance.total_requests for instance in self.instances.values())
        
        distribution = {}
        for instance_id, instance in self.instances.items():
            percentage = (instance.total_requests / total_requests * 100) if total_requests > 0 else 0
            distribution[instance_id] = {
                "requests": instance.total_requests,
                "percentage": percentage,
                "current_connections": instance.current_connections
            }
            
        return {
            "total_requests": total_requests,
            "distribution": distribution,
            "timestamp": datetime.now().isoformat()
        }