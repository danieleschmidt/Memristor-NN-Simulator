"""
GitLab CI Integration for Pipeline Guard
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.pipeline_monitor import PipelineMonitor
from ..core.failure_detector import FailureDetector
from ..core.healing_engine import HealingEngine


@dataclass
class GitLabCIConfig:
    """Configuration for GitLab CI integration"""
    base_url: str  # e.g., https://gitlab.com
    project_id: str  # Project ID or path
    private_token: str  # Personal or project access token


class GitLabCIIntegration:
    """
    Integration with GitLab CI for pipeline monitoring and healing
    """
    
    def __init__(self, config: GitLabCIConfig,
                 monitor: PipelineMonitor,
                 detector: FailureDetector,
                 healer: HealingEngine):
        self.config = config
        self.monitor = monitor
        self.detector = detector
        self.healer = healer
        self.logger = logging.getLogger(__name__)
        
        self.session = requests.Session()
        self.session.headers.update({
            "PRIVATE-TOKEN": config.private_token,
            "Content-Type": "application/json"
        })
        
    def start_monitoring(self):
        """Start monitoring GitLab CI pipelines"""
        self.logger.info("Starting GitLab CI monitoring")
        self.monitor.start_monitoring()
        
        # Register recent pipelines
        pipelines = self._get_recent_pipelines()
        for pipeline in pipelines:
            if pipeline["status"] == "running":
                self._register_pipeline(pipeline)
            elif pipeline["status"] == "failed":
                self._handle_failed_pipeline(pipeline)
                
    def _get_recent_pipelines(self) -> List[Dict[str, Any]]:
        """Get recent pipelines for the project"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/pipelines"
        
        try:
            response = self.session.get(url, params={"per_page": 20})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch GitLab pipelines: {e}")
            return []
            
    def _register_pipeline(self, pipeline: Dict[str, Any]):
        """Register a pipeline with the monitor"""
        pipeline_id = str(pipeline["id"])
        
        self.monitor.register_pipeline(
            pipeline_id=pipeline_id,
            name=f"Pipeline #{pipeline['id']} ({pipeline.get('ref', 'unknown')})",
            metadata={
                "gitlab_pipeline_id": pipeline["id"],
                "ref": pipeline.get("ref"),
                "sha": pipeline.get("sha"),
                "status": pipeline["status"],
                "web_url": pipeline.get("web_url"),
                "created_at": pipeline.get("created_at"),
                "updated_at": pipeline.get("updated_at")
            }
        )
        
    def _handle_failed_pipeline(self, pipeline: Dict[str, Any]):
        """Handle a failed GitLab pipeline"""
        pipeline_id = str(pipeline["id"])
        
        # Get detailed pipeline information including jobs
        pipeline_details = self._get_pipeline_details(pipeline["id"])
        failed_jobs = self._get_failed_jobs(pipeline["id"])
        
        # Collect logs from failed jobs
        all_logs = []
        for job in failed_jobs:
            job_log = self._get_job_log(job["id"])
            if job_log:
                all_logs.append(f"Job: {job['name']}\n{job_log}")
                
        combined_logs = "\n\n".join(all_logs)
        
        # Detect failure pattern
        detection = self.detector.detect_failure(combined_logs, {
            "pipeline_id": pipeline_id,
            "gitlab_pipeline_id": pipeline["id"],
            "ref": pipeline.get("ref"),
            "failed_jobs": [job["name"] for job in failed_jobs]
        })
        
        if detection.detected:
            self.logger.info(f"Detected failure pattern '{detection.pattern_id}' in pipeline {pipeline_id}")
            
            # Attempt healing if enabled
            if self.healer.healing_enabled:
                healing_results = self.healer.heal_pipeline(
                    pipeline_id=pipeline_id,
                    failure_pattern=detection.suggested_remediation,
                    context={
                        "logs": combined_logs,
                        "pipeline_data": pipeline_details,
                        "failed_jobs": failed_jobs,
                        "detection": detection
                    }
                )
                
                # If healing was successful, consider retrying failed jobs
                if any(r.status.value == "success" for r in healing_results):
                    self._consider_job_retry(pipeline["id"], failed_jobs, detection)
                    
    def _get_pipeline_details(self, pipeline_id: int) -> Dict[str, Any]:
        """Get detailed information about a pipeline"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/pipelines/{pipeline_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to get pipeline details for {pipeline_id}: {e}")
            return {}
            
    def _get_failed_jobs(self, pipeline_id: int) -> List[Dict[str, Any]]:
        """Get failed jobs for a pipeline"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/pipelines/{pipeline_id}/jobs"
        
        try:
            response = self.session.get(url, params={"scope": "failed"})
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to get failed jobs for pipeline {pipeline_id}: {e}")
            return []
            
    def _get_job_log(self, job_id: int) -> str:
        """Get log output for a specific job"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/jobs/{job_id}/trace"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.error(f"Failed to get job log for {job_id}: {e}")
            return ""
            
    def _consider_job_retry(self, pipeline_id: int, failed_jobs: List[Dict[str, Any]], detection):
        """Consider whether to retry failed jobs after healing"""
        retryable_patterns = [
            "dependency_failure",
            "network_failure",
            "timeout_failure",
            "resource_failure"
        ]
        
        if detection.pattern_id in retryable_patterns and detection.confidence > 0.8:
            for job in failed_jobs:
                if self._retry_job(job["id"]):
                    self.logger.info(f"Successfully retried job {job['name']} ({job['id']})")
                    
    def _retry_job(self, job_id: int) -> bool:
        """Retry a failed job"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/jobs/{job_id}/retry"
        
        try:
            response = self.session.post(url)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            self.logger.error(f"Failed to retry job {job_id}: {e}")
            return False
            
    def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitLab CI webhook events"""
        object_kind = payload.get("object_kind")
        
        if object_kind == "pipeline":
            return self._handle_pipeline_webhook(payload)
        elif object_kind == "job":
            return self._handle_job_webhook(payload)
        else:
            return {"status": "ignored", "object_kind": object_kind}
            
    def _handle_pipeline_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline webhook events"""
        pipeline_data = payload.get("object_attributes", {})
        pipeline_id = str(pipeline_data.get("id", "unknown"))
        status = pipeline_data.get("status")
        
        if status == "running":
            # Pipeline started
            self._register_pipeline(pipeline_data)
            return {"status": "registered", "pipeline_id": pipeline_id}
            
        elif status in ["success", "failed", "canceled"]:
            # Pipeline completed
            self.monitor.update_pipeline_status(
                pipeline_id=pipeline_id,
                status="success" if status == "success" else "failed",
                failure_reason=status if status != "success" else None
            )
            
            if status == "failed":
                self._handle_failed_pipeline(pipeline_data)
                
            return {"status": "processed", "pipeline_id": pipeline_id, "result": status}
            
        return {"status": "ignored", "pipeline_status": status}
        
    def _handle_job_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle job webhook events"""
        job_data = payload.get("object_attributes", {})
        job_id = job_data.get("id")
        status = job_data.get("status")
        
        # For now, we primarily track at pipeline level
        # But we could extend this for job-level monitoring
        
        return {"status": "job_event_logged", "job_id": job_id, "status": status}
        
    def get_project_health(self) -> Dict[str, Any]:
        """Get overall project health metrics"""
        try:
            # Get recent pipelines
            pipelines = self._get_recent_pipelines()
            
            if not pipelines:
                return {"project_id": self.config.project_id, "error": "No pipelines found"}
                
            total_pipelines = len(pipelines)
            successful_pipelines = len([p for p in pipelines if p["status"] == "success"])
            failed_pipelines = len([p for p in pipelines if p["status"] == "failed"])
            
            # Calculate average duration for completed pipelines
            completed_pipelines = [p for p in pipelines if p["status"] in ["success", "failed"]]
            avg_duration = 0
            if completed_pipelines:
                durations = []
                for pipeline in completed_pipelines:
                    if pipeline.get("created_at") and pipeline.get("updated_at"):
                        try:
                            start = datetime.fromisoformat(pipeline["created_at"].replace("Z", "+00:00"))
                            end = datetime.fromisoformat(pipeline["updated_at"].replace("Z", "+00:00"))
                            durations.append((end - start).total_seconds())
                        except ValueError:
                            continue
                            
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    
            return {
                "project_id": self.config.project_id,
                "total_pipelines": total_pipelines,
                "successful_pipelines": successful_pipelines,
                "failed_pipelines": failed_pipelines,
                "success_rate": successful_pipelines / total_pipelines if total_pipelines > 0 else 0,
                "average_duration_seconds": avg_duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get project health: {e}")
            return {"project_id": self.config.project_id, "error": str(e)}
            
    def get_pipeline_variables(self, pipeline_id: int) -> Dict[str, Any]:
        """Get variables for a specific pipeline"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/pipelines/{pipeline_id}/variables"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return {var["key"]: var["value"] for var in response.json()}
        except requests.RequestException as e:
            self.logger.error(f"Failed to get pipeline variables for {pipeline_id}: {e}")
            return {}
            
    def create_pipeline(self, ref: str, variables: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Create a new pipeline"""
        url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}/pipeline"
        
        data = {"ref": ref}
        if variables:
            data["variables"] = [{"key": k, "value": v} for k, v in variables.items()]
            
        try:
            response = self.session.post(url, json=data)
            response.raise_for_status()
            
            pipeline_data = response.json()
            self.logger.info(f"Created pipeline {pipeline_data['id']} for ref {ref}")
            return pipeline_data
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to create pipeline for ref {ref}: {e}")
            return None
            
    def get_project_statistics(self) -> Dict[str, Any]:
        """Get comprehensive project statistics"""
        try:
            # Get project info
            project_url = f"{self.config.base_url}/api/v4/projects/{self.config.project_id}"
            project_response = self.session.get(project_url)
            project_data = project_response.json() if project_response.status_code == 200 else {}
            
            # Get pipeline statistics
            pipelines = self._get_recent_pipelines()
            
            # Analyze failure patterns over time
            failure_trends = {}
            for pipeline in pipelines:
                if pipeline["status"] == "failed":
                    date = pipeline.get("created_at", "")[:10]  # YYYY-MM-DD
                    failure_trends[date] = failure_trends.get(date, 0) + 1
                    
            return {
                "project_name": project_data.get("name", "Unknown"),
                "project_path": project_data.get("path_with_namespace", "Unknown"),
                "default_branch": project_data.get("default_branch", "main"),
                "total_pipelines_analyzed": len(pipelines),
                "pipeline_health": self.get_project_health(),
                "failure_trends": failure_trends,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get project statistics: {e}")
            return {"error": str(e)}