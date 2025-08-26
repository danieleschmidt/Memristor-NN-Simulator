"""
Distributed simulation engine for large-scale memristor neural networks.

Implements:
- Multi-node distributed computing
- Communication-efficient partitioning
- Load balancing and fault tolerance
- Hierarchical simulation strategies
"""

import numpy as np
import asyncio
import threading
import queue
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from pathlib import Path

from ..utils.logger import LoggingMixin
from ..utils.validators import validate_positive_number, validate_numpy_array


class PartitionStrategy(Enum):
    """Strategies for partitioning crossbar arrays across nodes."""
    ROW_WISE = "row_wise"
    COLUMN_WISE = "column_wise"
    BLOCK_WISE = "block_wise"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class CommunicationProtocol(Enum):
    """Communication protocols for distributed nodes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BULK_SYNCHRONOUS = "bulk_synchronous"
    PIPELINE = "pipeline"


@dataclass
class NodeConfig:
    """Configuration for distributed compute nodes."""
    node_id: int
    cpu_cores: int = 4
    memory_gb: int = 8
    gpu_available: bool = False
    network_bandwidth_gbps: float = 1.0
    processing_capability: float = 1.0  # Relative capability score
    
    
@dataclass
class DistributionConfig:
    """Configuration for distributed simulation."""
    partition_strategy: PartitionStrategy = PartitionStrategy.BLOCK_WISE
    communication_protocol: CommunicationProtocol = CommunicationProtocol.BULK_SYNCHRONOUS
    max_nodes: int = 32
    load_balance_threshold: float = 0.2
    fault_tolerance_enabled: bool = True
    checkpoint_interval: int = 100
    compression_enabled: bool = True
    prefetch_enabled: bool = True


class DistributedNode:
    """Individual compute node in distributed simulation."""
    
    def __init__(self, config: NodeConfig, crossbar_partition: np.ndarray):
        self.config = config
        self.partition = crossbar_partition
        self.local_state = {}
        self.communication_queue = queue.Queue()
        self.performance_metrics = {
            "operations_per_second": 0,
            "memory_usage_mb": 0,
            "communication_latency_ms": 0,
            "error_count": 0
        }
        
        # Initialize local computation engine
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.cpu_cores
        )
        
    async def process_partition(
        self, 
        input_data: np.ndarray, 
        simulation_params: Dict
    ) -> Dict[str, Any]:
        """Process assigned partition of the crossbar."""
        start_time = time.time()
        
        try:
            # Simulate local partition
            local_result = await self._simulate_local_partition(
                input_data, simulation_params
            )
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.performance_metrics["operations_per_second"] = (
                len(input_data) / processing_time if processing_time > 0 else 0
            )
            
            return {
                "node_id": self.config.node_id,
                "result": local_result,
                "processing_time": processing_time,
                "partition_shape": self.partition.shape,
                "success": True
            }
            
        except Exception as e:
            return {
                "node_id": self.config.node_id,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    async def _simulate_local_partition(
        self, 
        input_data: np.ndarray, 
        params: Dict
    ) -> np.ndarray:
        """Simulate the local crossbar partition."""
        # Simple matrix multiplication simulation
        # In practice, this would use the full memristor device models
        
        # Apply input to partition
        if len(input_data.shape) == 1:
            # Single vector input
            result = self.partition @ input_data
        else:
            # Batch processing
            result = self.partition @ input_data.T
        
        # Add device-level effects (simplified)
        if params.get("add_noise", True):
            noise_level = params.get("noise_sigma", 0.01)
            noise = np.random.normal(0, noise_level, result.shape)
            result += noise
        
        # Apply non-linearity if specified
        if params.get("apply_nonlinearity", False):
            result = np.tanh(result)
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics for this node."""
        return self.performance_metrics.copy()


class DistributedSimulator(LoggingMixin):
    """Main distributed simulation orchestrator."""
    
    def __init__(
        self,
        crossbar_array,
        config: Optional[DistributionConfig] = None,
        available_nodes: Optional[List[NodeConfig]] = None
    ):
        """
        Initialize distributed simulator.
        
        Args:
            crossbar_array: Crossbar array to distribute
            config: Distribution configuration
            available_nodes: List of available compute nodes
        """
        super().__init__()
        self.crossbar = crossbar_array
        self.config = config or DistributionConfig()
        
        # Initialize compute nodes
        if available_nodes is None:
            available_nodes = self._create_default_nodes()
        
        self.nodes = {}
        self.partition_map = {}
        self.performance_history = []
        
        # Distribute crossbar across nodes
        self._partition_crossbar(available_nodes)
        
        # Communication and synchronization
        self.coordination_lock = threading.Lock()
        self.global_state = {}
        self.checkpoint_data = {}
        
        self.logger.info(f"Distributed simulator initialized with {len(self.nodes)} nodes")
    
    def _create_default_nodes(self) -> List[NodeConfig]:
        """Create default node configuration for local simulation."""
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        default_nodes = []
        
        # Create nodes based on available CPU cores
        nodes_count = min(self.config.max_nodes, cpu_count)
        cores_per_node = max(1, cpu_count // nodes_count)
        
        for i in range(nodes_count):
            node_config = NodeConfig(
                node_id=i,
                cpu_cores=cores_per_node,
                memory_gb=8,  # Assume 8GB per node
                network_bandwidth_gbps=10.0,  # Local network
                processing_capability=1.0
            )
            default_nodes.append(node_config)
        
        return default_nodes
    
    def _partition_crossbar(self, node_configs: List[NodeConfig]) -> None:
        """Partition crossbar array across available nodes."""
        try:
            if self.config.partition_strategy == PartitionStrategy.ROW_WISE:
                partitions = self._row_wise_partition(node_configs)
            elif self.config.partition_strategy == PartitionStrategy.COLUMN_WISE:
                partitions = self._column_wise_partition(node_configs)
            elif self.config.partition_strategy == PartitionStrategy.BLOCK_WISE:
                partitions = self._block_wise_partition(node_configs)
            elif self.config.partition_strategy == PartitionStrategy.HIERARCHICAL:
                partitions = self._hierarchical_partition(node_configs)
            else:  # ADAPTIVE
                partitions = self._adaptive_partition(node_configs)
            
            # Create distributed nodes
            for i, (node_config, partition) in enumerate(zip(node_configs, partitions)):
                self.nodes[node_config.node_id] = DistributedNode(node_config, partition)
                self.partition_map[node_config.node_id] = {
                    "shape": partition.shape,
                    "memory_size_mb": partition.nbytes / (1024 * 1024)
                }
            
            self.logger.info(f"Crossbar partitioned using {self.config.partition_strategy.value} strategy")
            
        except Exception as e:
            self.logger.error(f"Crossbar partitioning failed: {e}")
            raise
    
    def _row_wise_partition(self, node_configs: List[NodeConfig]) -> List[np.ndarray]:
        """Partition crossbar by rows."""
        conductance_matrix = self.crossbar.get_conductance_matrix()
        rows_per_node = conductance_matrix.shape[0] // len(node_configs)
        
        partitions = []
        for i, config in enumerate(node_configs):
            start_row = i * rows_per_node
            if i == len(node_configs) - 1:  # Last node gets remaining rows
                end_row = conductance_matrix.shape[0]
            else:
                end_row = (i + 1) * rows_per_node
            
            partition = conductance_matrix[start_row:end_row, :]
            partitions.append(partition)
        
        return partitions
    
    def _column_wise_partition(self, node_configs: List[NodeConfig]) -> List[np.ndarray]:
        """Partition crossbar by columns."""
        conductance_matrix = self.crossbar.get_conductance_matrix()
        cols_per_node = conductance_matrix.shape[1] // len(node_configs)
        
        partitions = []
        for i, config in enumerate(node_configs):
            start_col = i * cols_per_node
            if i == len(node_configs) - 1:  # Last node gets remaining columns
                end_col = conductance_matrix.shape[1]
            else:
                end_col = (i + 1) * cols_per_node
            
            partition = conductance_matrix[:, start_col:end_col]
            partitions.append(partition)
        
        return partitions
    
    def _block_wise_partition(self, node_configs: List[NodeConfig]) -> List[np.ndarray]:
        """Partition crossbar into 2D blocks."""
        conductance_matrix = self.crossbar.get_conductance_matrix()
        
        # Calculate optimal grid dimensions
        num_nodes = len(node_configs)
        grid_rows = int(np.ceil(np.sqrt(num_nodes)))
        grid_cols = int(np.ceil(num_nodes / grid_rows))
        
        rows_per_block = conductance_matrix.shape[0] // grid_rows
        cols_per_block = conductance_matrix.shape[1] // grid_cols
        
        partitions = []
        node_idx = 0
        
        for grid_i in range(grid_rows):
            for grid_j in range(grid_cols):
                if node_idx >= num_nodes:
                    break
                
                start_row = grid_i * rows_per_block
                start_col = grid_j * cols_per_block
                
                if grid_i == grid_rows - 1:  # Last row
                    end_row = conductance_matrix.shape[0]
                else:
                    end_row = (grid_i + 1) * rows_per_block
                
                if grid_j == grid_cols - 1:  # Last column
                    end_col = conductance_matrix.shape[1]
                else:
                    end_col = (grid_j + 1) * cols_per_block
                
                partition = conductance_matrix[start_row:end_row, start_col:end_col]
                partitions.append(partition)
                node_idx += 1
        
        return partitions
    
    def _hierarchical_partition(self, node_configs: List[NodeConfig]) -> List[np.ndarray]:
        """Hierarchical partitioning based on node capabilities."""
        # Sort nodes by processing capability
        sorted_configs = sorted(node_configs, key=lambda x: x.processing_capability, reverse=True)
        
        conductance_matrix = self.crossbar.get_conductance_matrix()
        total_elements = conductance_matrix.size
        
        partitions = []
        remaining_matrix = conductance_matrix
        
        for i, config in enumerate(sorted_configs):
            # Allocate partition size based on capability
            capability_ratio = config.processing_capability / sum(c.processing_capability for c in sorted_configs[i:])
            partition_size = int(total_elements * capability_ratio)
            
            # Create partition (simplified - would need more sophisticated allocation)
            if i == len(sorted_configs) - 1:  # Last node gets everything remaining
                partition = remaining_matrix
            else:
                elements_needed = min(partition_size, remaining_matrix.size)
                partition_rows = min(remaining_matrix.shape[0], 
                                   int(np.ceil(elements_needed / remaining_matrix.shape[1])))
                partition = remaining_matrix[:partition_rows, :]
                remaining_matrix = remaining_matrix[partition_rows:, :]
            
            partitions.append(partition)
        
        return partitions
    
    def _adaptive_partition(self, node_configs: List[NodeConfig]) -> List[np.ndarray]:
        """Adaptive partitioning based on runtime characteristics."""
        # For now, use block-wise as baseline
        # In practice, this would analyze crossbar characteristics and adapt
        return self._block_wise_partition(node_configs)
    
    async def distributed_simulate(
        self,
        input_data: np.ndarray,
        simulation_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform distributed simulation across all nodes.
        
        Args:
            input_data: Input vectors/matrices for simulation
            simulation_params: Simulation parameters
            
        Returns:
            Aggregated simulation results
        """
        try:
            simulation_params = simulation_params or {}
            start_time = time.time()
            
            # Validate input
            input_data = validate_numpy_array(input_data, name="input_data")
            
            self.logger.info(f"Starting distributed simulation on {len(self.nodes)} nodes")
            
            # Distribute simulation tasks
            tasks = []
            for node_id, node in self.nodes.items():
                task = node.process_partition(input_data, simulation_params)
                tasks.append(task)
            
            # Execute simulation on all nodes
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate results
            aggregated_result = self._aggregate_results(results)
            
            # Update performance history
            total_time = time.time() - start_time
            performance_entry = {
                "timestamp": time.time(),
                "total_simulation_time": total_time,
                "input_size": input_data.size,
                "throughput_ops_per_second": input_data.size / total_time if total_time > 0 else 0,
                "nodes_used": len(self.nodes),
                "success_rate": sum(1 for r in results if isinstance(r, dict) and r.get("success", False)) / len(results)
            }
            self.performance_history.append(performance_entry)
            
            aggregated_result["distributed_performance"] = performance_entry
            
            return aggregated_result
            
        except Exception as e:
            self.logger.error(f"Distributed simulation failed: {e}")
            raise
    
    def _aggregate_results(self, node_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from distributed nodes."""
        successful_results = []
        failed_results = []
        
        for result in node_results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "success": False})
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        if not successful_results:
            raise RuntimeError("All distributed nodes failed")
        
        # Combine successful results
        # This is simplified - actual aggregation depends on partition strategy
        combined_output = []
        processing_times = []
        
        for result in successful_results:
            if "result" in result:
                combined_output.append(result["result"])
                processing_times.append(result["processing_time"])
        
        if self.config.partition_strategy == PartitionStrategy.ROW_WISE:
            # Concatenate row-wise results
            aggregated_output = np.concatenate(combined_output, axis=0)
        elif self.config.partition_strategy == PartitionStrategy.COLUMN_WISE:
            # Sum column-wise results (for matrix multiplication)
            aggregated_output = np.sum(combined_output, axis=0)
        else:
            # Block-wise or other strategies - more complex aggregation needed
            aggregated_output = np.concatenate([r.flatten() for r in combined_output])
        
        return {
            "output": aggregated_output,
            "successful_nodes": len(successful_results),
            "failed_nodes": len(failed_results),
            "average_processing_time": np.mean(processing_times) if processing_times else 0,
            "max_processing_time": np.max(processing_times) if processing_times else 0,
            "aggregation_strategy": self.config.partition_strategy.value,
            "node_results": successful_results
        }
    
    def scale_nodes(self, target_nodes: int) -> Dict[str, Any]:
        """Dynamically scale the number of compute nodes."""
        try:
            current_nodes = len(self.nodes)
            
            if target_nodes == current_nodes:
                return {"action": "no_change", "current_nodes": current_nodes}
            
            elif target_nodes > current_nodes:
                # Scale up - add nodes
                new_node_configs = []
                for i in range(current_nodes, target_nodes):
                    new_config = NodeConfig(
                        node_id=i,
                        cpu_cores=2,
                        memory_gb=4,
                        processing_capability=1.0
                    )
                    new_node_configs.append(new_config)
                
                # Repartition crossbar with new nodes
                all_configs = list(node.config for node in self.nodes.values()) + new_node_configs
                self._partition_crossbar(all_configs)
                
                return {
                    "action": "scale_up",
                    "previous_nodes": current_nodes,
                    "current_nodes": len(self.nodes),
                    "new_nodes_added": target_nodes - current_nodes
                }
            
            else:
                # Scale down - remove nodes
                nodes_to_remove = current_nodes - target_nodes
                
                # Remove least capable nodes first
                sorted_nodes = sorted(
                    self.nodes.values(),
                    key=lambda n: n.config.processing_capability
                )
                
                for i in range(nodes_to_remove):
                    node_to_remove = sorted_nodes[i]
                    del self.nodes[node_to_remove.config.node_id]
                
                # Repartition remaining crossbar
                remaining_configs = [node.config for node in self.nodes.values()]
                if remaining_configs:
                    self._partition_crossbar(remaining_configs)
                
                return {
                    "action": "scale_down",
                    "previous_nodes": current_nodes,
                    "current_nodes": len(self.nodes),
                    "nodes_removed": nodes_to_remove
                }
                
        except Exception as e:
            self.logger.error(f"Node scaling failed: {e}")
            raise
    
    def get_load_balance_metrics(self) -> Dict[str, float]:
        """Calculate load balancing metrics across nodes."""
        try:
            if not self.nodes:
                return {"load_balance_score": 0.0, "utilization_variance": 0.0}
            
            # Get performance metrics from all nodes
            utilizations = []
            processing_times = []
            
            for node in self.nodes.values():
                metrics = node.get_performance_metrics()
                utilizations.append(metrics.get("operations_per_second", 0))
                processing_times.append(metrics.get("communication_latency_ms", 0))
            
            # Calculate load balance metrics
            if utilizations:
                util_variance = np.var(utilizations)
                util_mean = np.mean(utilizations)
                load_balance_score = 1.0 - (util_variance / (util_mean + 1e-12))
            else:
                util_variance = 0.0
                load_balance_score = 1.0
            
            return {
                "load_balance_score": float(max(0.0, min(1.0, load_balance_score))),
                "utilization_variance": float(util_variance),
                "average_utilization": float(np.mean(utilizations)) if utilizations else 0.0,
                "processing_time_variance": float(np.var(processing_times)) if processing_times else 0.0,
                "active_nodes": len(self.nodes)
            }
            
        except Exception as e:
            self.logger.error(f"Load balance metrics calculation failed: {e}")
            return {"error": str(e)}
    
    def optimize_distribution(self) -> Dict[str, Any]:
        """Optimize node distribution based on performance metrics."""
        try:
            load_metrics = self.get_load_balance_metrics()
            
            # Check if rebalancing is needed
            if load_metrics["load_balance_score"] < (1.0 - self.config.load_balance_threshold):
                self.logger.info("Load imbalance detected, optimizing distribution")
                
                # Analyze node performance
                node_performances = {}
                for node_id, node in self.nodes.items():
                    metrics = node.get_performance_metrics()
                    node_performances[node_id] = metrics["operations_per_second"]
                
                # Identify underperforming nodes
                avg_performance = np.mean(list(node_performances.values()))
                underperforming_nodes = [
                    node_id for node_id, perf in node_performances.items()
                    if perf < avg_performance * 0.8
                ]
                
                optimization_actions = []
                
                # Add more nodes if system is overloaded
                if load_metrics["average_utilization"] > 0.8:
                    current_node_count = len(self.nodes)
                    target_nodes = min(self.config.max_nodes, int(current_node_count * 1.2))
                    scale_result = self.scale_nodes(target_nodes)
                    optimization_actions.append(scale_result)
                
                # Remove underperforming nodes if we have excess capacity
                elif len(underperforming_nodes) > 0 and load_metrics["average_utilization"] < 0.4:
                    target_nodes = len(self.nodes) - min(2, len(underperforming_nodes))
                    scale_result = self.scale_nodes(max(1, target_nodes))
                    optimization_actions.append(scale_result)
                
                return {
                    "optimization_performed": True,
                    "initial_load_balance": load_metrics["load_balance_score"],
                    "actions_taken": optimization_actions,
                    "underperforming_nodes": len(underperforming_nodes)
                }
            
            else:
                return {
                    "optimization_performed": False,
                    "load_balance_score": load_metrics["load_balance_score"],
                    "reason": "System already well-balanced"
                }
                
        except Exception as e:
            self.logger.error(f"Distribution optimization failed: {e}")
            raise
    
    def save_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Save distributed simulation checkpoint."""
        try:
            checkpoint_data = {
                "timestamp": time.time(),
                "config": {
                    "partition_strategy": self.config.partition_strategy.value,
                    "communication_protocol": self.config.communication_protocol.value,
                    "max_nodes": self.config.max_nodes
                },
                "nodes": {
                    node_id: {
                        "config": {
                            "node_id": node.config.node_id,
                            "cpu_cores": node.config.cpu_cores,
                            "memory_gb": node.config.memory_gb,
                            "processing_capability": node.config.processing_capability
                        },
                        "partition_shape": node.partition.shape,
                        "performance_metrics": node.get_performance_metrics()
                    }
                    for node_id, node in self.nodes.items()
                },
                "performance_history": self.performance_history[-10:],  # Last 10 entries
                "partition_map": self.partition_map
            }
            
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            
            self.logger.info(f"Checkpoint saved to {checkpoint_path} ({checkpoint_size_mb:.2f} MB)")
            
            return {
                "checkpoint_saved": True,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_mb": checkpoint_size_mb,
                "nodes_checkpointed": len(self.nodes)
            }
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
            raise
    
    def get_distributed_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive distributed simulation status report."""
        try:
            load_metrics = self.get_load_balance_metrics()
            
            return {
                "system_overview": {
                    "total_nodes": len(self.nodes),
                    "partition_strategy": self.config.partition_strategy.value,
                    "communication_protocol": self.config.communication_protocol.value,
                    "fault_tolerance_enabled": self.config.fault_tolerance_enabled
                },
                "performance_metrics": {
                    "load_balance_score": load_metrics["load_balance_score"],
                    "average_utilization": load_metrics["average_utilization"],
                    "utilization_variance": load_metrics["utilization_variance"],
                    "recent_throughput": self.performance_history[-1]["throughput_ops_per_second"] if self.performance_history else 0
                },
                "node_details": {
                    node_id: {
                        "partition_shape": self.partition_map[node_id]["shape"],
                        "memory_usage_mb": self.partition_map[node_id]["memory_size_mb"],
                        "performance": node.get_performance_metrics()
                    }
                    for node_id, node in self.nodes.items()
                },
                "scaling_status": {
                    "current_capacity": len(self.nodes),
                    "max_capacity": self.config.max_nodes,
                    "scaling_headroom": self.config.max_nodes - len(self.nodes),
                    "auto_scaling_enabled": True  # Could be configurable
                },
                "reliability": {
                    "fault_tolerance_active": self.config.fault_tolerance_enabled,
                    "checkpoint_interval": self.config.checkpoint_interval,
                    "recent_success_rate": self.performance_history[-1]["success_rate"] if self.performance_history else 1.0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Status report generation failed: {e}")
            raise