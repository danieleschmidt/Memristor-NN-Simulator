"""
Pipeline Monitor: Real-time monitoring of CI/CD pipelines
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue


@dataclass
class PipelineStatus:
    """Pipeline status data structure"""
    pipeline_id: str
    name: str
    status: str  # running, success, failed, cancelled
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration: Optional[float] = None
    failure_reason: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    success_rate: float
    avg_duration: float
    failure_patterns: Dict[str, int]
    recent_failures: List[str]
    peak_concurrency: int


class PipelineMonitor:
    """
    Real-time pipeline monitoring with health metrics and failure detection
    """
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.pipelines: Dict[str, PipelineStatus] = {}
        self.metrics = PipelineMetrics(
            success_rate=0.0,
            avg_duration=0.0,
            failure_patterns={},
            recent_failures=[],
            peak_concurrency=0
        )
        self.monitoring = False
        self.monitor_thread = None
        self.event_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start pipeline monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        self.logger.info("Pipeline monitoring started")
        
    def stop_monitoring(self):
        """Stop pipeline monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Pipeline monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._check_pipelines()
                self._update_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                
    def _check_pipelines(self):
        """Check status of all registered pipelines"""
        current_time = datetime.now()
        
        for pipeline_id, pipeline in self.pipelines.items():
            if pipeline.status == "running":
                # Check for timeouts
                if self._is_pipeline_timeout(pipeline, current_time):
                    self._handle_timeout(pipeline)
                    
                # Check for resource issues
                if self._check_resource_issues(pipeline):
                    self._handle_resource_issue(pipeline)
                    
    def _is_pipeline_timeout(self, pipeline: PipelineStatus, current_time: datetime) -> bool:
        """Check if pipeline has timed out"""
        if not pipeline.started_at:
            return False
            
        timeout_threshold = timedelta(hours=2)  # Default 2 hour timeout
        return (current_time - pipeline.started_at) > timeout_threshold
        
    def _check_resource_issues(self, pipeline: PipelineStatus) -> bool:
        """Check for resource-related issues"""
        # Simulate resource checking
        return False
        
    def _handle_timeout(self, pipeline: PipelineStatus):
        """Handle pipeline timeout"""
        pipeline.status = "failed"
        pipeline.failure_reason = "timeout"
        pipeline.finished_at = datetime.now()
        
        self.event_queue.put({
            "type": "timeout",
            "pipeline_id": pipeline.pipeline_id,
            "timestamp": datetime.now().isoformat()
        })
        
    def _handle_resource_issue(self, pipeline: PipelineStatus):
        """Handle resource issues"""
        self.event_queue.put({
            "type": "resource_issue", 
            "pipeline_id": pipeline.pipeline_id,
            "timestamp": datetime.now().isoformat()
        })
        
    def _update_metrics(self):
        """Update pipeline metrics"""
        if not self.pipelines:
            return
            
        completed_pipelines = [p for p in self.pipelines.values() 
                             if p.status in ["success", "failed"]]
        
        if completed_pipelines:
            successful = [p for p in completed_pipelines if p.status == "success"]
            self.metrics.success_rate = len(successful) / len(completed_pipelines)
            
            durations = [p.duration for p in completed_pipelines if p.duration]
            if durations:
                self.metrics.avg_duration = sum(durations) / len(durations)
                
        # Update failure patterns
        failed_pipelines = [p for p in completed_pipelines if p.status == "failed"]
        for pipeline in failed_pipelines:
            if pipeline.failure_reason:
                self.metrics.failure_patterns[pipeline.failure_reason] = \
                    self.metrics.failure_patterns.get(pipeline.failure_reason, 0) + 1
                    
        # Update concurrency
        running_count = len([p for p in self.pipelines.values() if p.status == "running"])
        self.metrics.peak_concurrency = max(self.metrics.peak_concurrency, running_count)
        
    def register_pipeline(self, pipeline_id: str, name: str, metadata: Dict = None):
        """Register a new pipeline for monitoring"""
        pipeline = PipelineStatus(
            pipeline_id=pipeline_id,
            name=name,
            status="running",
            started_at=datetime.now(),
            metadata=metadata or {}
        )
        
        self.pipelines[pipeline_id] = pipeline
        self.logger.info(f"Registered pipeline: {name} ({pipeline_id})")
        
    def update_pipeline_status(self, pipeline_id: str, status: str, 
                             failure_reason: str = None):
        """Update pipeline status"""
        if pipeline_id not in self.pipelines:
            self.logger.warning(f"Unknown pipeline: {pipeline_id}")
            return
            
        pipeline = self.pipelines[pipeline_id]
        pipeline.status = status
        
        if status in ["success", "failed", "cancelled"]:
            pipeline.finished_at = datetime.now()
            if pipeline.started_at:
                pipeline.duration = (pipeline.finished_at - pipeline.started_at).total_seconds()
                
        if failure_reason:
            pipeline.failure_reason = failure_reason
            
        self.logger.info(f"Pipeline {pipeline_id} status: {status}")
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        running_pipelines = [p for p in self.pipelines.values() if p.status == "running"]
        failed_pipelines = [p for p in self.pipelines.values() if p.status == "failed"]
        
        return {
            "total_pipelines": len(self.pipelines),
            "running_pipelines": len(running_pipelines),
            "failed_pipelines": len(failed_pipelines),
            "success_rate": self.metrics.success_rate,
            "avg_duration_seconds": self.metrics.avg_duration,
            "top_failure_reasons": dict(sorted(
                self.metrics.failure_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]),
            "timestamp": datetime.now().isoformat()
        }
        
    def get_events(self) -> List[Dict]:
        """Get recent events from queue"""
        events = []
        while not self.event_queue.empty():
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events