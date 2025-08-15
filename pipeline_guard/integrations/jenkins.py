"""
Jenkins Integration for Pipeline Guard
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from ..core.pipeline_monitor import PipelineMonitor  
from ..core.failure_detector import FailureDetector
from ..core.healing_engine import HealingEngine


@dataclass
class JenkinsConfig:
    """Configuration for Jenkins integration"""
    base_url: str
    username: str
    api_token: str
    verify_ssl: bool = True


class JenkinsIntegration:
    """
    Integration with Jenkins for pipeline monitoring and healing
    """
    
    def __init__(self, config: JenkinsConfig,
                 monitor: PipelineMonitor,
                 detector: FailureDetector, 
                 healer: HealingEngine):
        self.config = config
        self.monitor = monitor
        self.detector = detector
        self.healer = healer
        self.logger = logging.getLogger(__name__)
        
        self.session = requests.Session()
        self.session.auth = (config.username, config.api_token)
        self.session.verify = config.verify_ssl
        
    def start_monitoring(self):
        """Start monitoring Jenkins jobs"""
        self.logger.info("Starting Jenkins monitoring")
        self.monitor.start_monitoring()
        
        # Register existing jobs
        jobs = self._get_jobs()
        for job in jobs:
            self._register_job_builds(job["name"])
            
    def _get_jobs(self) -> List[Dict[str, Any]]:
        """Get all Jenkins jobs"""
        url = f"{self.config.base_url}/api/json?tree=jobs[name,url,color]"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("jobs", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch Jenkins jobs: {e}")
            return []
            
    def _register_job_builds(self, job_name: str):
        """Register recent builds for a job"""
        url = f"{self.config.base_url}/job/{job_name}/api/json?tree=builds[number,id,result,building,timestamp,duration,url]"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            builds = response.json().get("builds", [])[:10]  # Last 10 builds
            
            for build in builds:
                if build["building"]:
                    self._register_pipeline_build(job_name, build)
                elif build["result"] in ["FAILURE", "ABORTED", "UNSTABLE"]:
                    self._handle_failed_build(job_name, build)
                    
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch builds for job {job_name}: {e}")
            
    def _register_pipeline_build(self, job_name: str, build: Dict[str, Any]):
        """Register a running build with the monitor"""
        pipeline_id = f"{job_name}#{build['number']}"
        
        self.monitor.register_pipeline(
            pipeline_id=pipeline_id,
            name=f"{job_name} #{build['number']}",
            metadata={
                "job_name": job_name,
                "build_number": build["number"],
                "build_id": build["id"],
                "jenkins_url": build["url"],
                "timestamp": build["timestamp"]
            }
        )
        
    def _handle_failed_build(self, job_name: str, build: Dict[str, Any]):
        """Handle a failed Jenkins build"""
        pipeline_id = f"{job_name}#{build['number']}"
        
        # Get build console output for analysis
        console_output = self._get_console_output(job_name, build["number"])
        
        # Detect failure pattern
        detection = self.detector.detect_failure(console_output, {
            "pipeline_id": pipeline_id,
            "job_name": job_name,
            "build_number": build["number"],
            "result": build["result"]
        })
        
        if detection.detected:
            self.logger.info(f"Detected failure pattern '{detection.pattern_id}' in build {pipeline_id}")
            
            # Attempt healing if enabled
            if self.healer.healing_enabled:
                healing_results = self.healer.heal_pipeline(
                    pipeline_id=pipeline_id,
                    failure_pattern=detection.suggested_remediation,
                    context={
                        "console_output": console_output,
                        "build_data": build,
                        "job_name": job_name,
                        "detection": detection
                    }
                )
                
                # If healing was successful, consider rebuilding
                if any(r.status.value == "success" for r in healing_results):
                    self._consider_rebuild(job_name, build, detection)
                    
    def _get_console_output(self, job_name: str, build_number: int) -> str:
        """Get console output for a build"""
        url = f"{self.config.base_url}/job/{job_name}/{build_number}/consoleText"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch console output for {job_name}#{build_number}: {e}")
            return ""
            
    def _consider_rebuild(self, job_name: str, build: Dict[str, Any], detection):
        """Consider whether to trigger a rebuild after healing"""
        rebuildable_patterns = [
            "dependency_failure",
            "network_failure",
            "timeout_failure", 
            "resource_failure"
        ]
        
        if detection.pattern_id in rebuildable_patterns and detection.confidence > 0.8:
            self.logger.info(f"Considering rebuild for {job_name}#{build['number']}")
            return self._trigger_rebuild(job_name)
            
        return False
        
    def _trigger_rebuild(self, job_name: str) -> bool:
        """Trigger a rebuild of a Jenkins job"""
        url = f"{self.config.base_url}/job/{job_name}/build"
        
        try:
            # Add a note about the automated rebuild
            response = self.session.post(url, data={
                "cause": "Automated rebuild triggered by Pipeline Guard after healing"
            })
            
            if response.status_code in [200, 201]:
                self.logger.info(f"Successfully triggered rebuild for {job_name}")
                return True
            else:
                self.logger.warning(f"Rebuild trigger returned status {response.status_code}")
                return False
                
        except requests.RequestException as e:
            self.logger.error(f"Failed to trigger rebuild for {job_name}: {e}")
            return False
            
    def get_job_health(self, job_name: str) -> Dict[str, Any]:
        """Get health metrics for a specific job"""
        url = f"{self.config.base_url}/job/{job_name}/api/json?tree=builds[number,result,timestamp,duration]"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            builds = response.json().get("builds", [])[:50]  # Last 50 builds
            
            if not builds:
                return {"job_name": job_name, "error": "No builds found"}
                
            total_builds = len(builds)
            successful_builds = len([b for b in builds if b["result"] == "SUCCESS"])
            failed_builds = len([b for b in builds if b["result"] == "FAILURE"])
            
            # Calculate average duration
            completed_builds = [b for b in builds if b["duration"] and b["duration"] > 0]
            avg_duration = 0
            if completed_builds:
                avg_duration = sum(b["duration"] for b in completed_builds) / len(completed_builds) / 1000  # Convert to seconds
                
            return {
                "job_name": job_name,
                "total_builds": total_builds,
                "successful_builds": successful_builds,
                "failed_builds": failed_builds,
                "success_rate": successful_builds / total_builds if total_builds > 0 else 0,
                "average_duration_seconds": avg_duration,
                "last_build_result": builds[0]["result"] if builds else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to get job health for {job_name}: {e}")
            return {"job_name": job_name, "error": str(e)}
            
    def get_jenkins_health(self) -> Dict[str, Any]:
        """Get overall Jenkins health metrics"""
        try:
            # Get Jenkins system info
            url = f"{self.config.base_url}/api/json?tree=jobs[name,color]"
            response = self.session.get(url)
            response.raise_for_status()
            
            jobs = response.json().get("jobs", [])
            
            total_jobs = len(jobs)
            healthy_jobs = len([j for j in jobs if j["color"] in ["blue", "green"]])
            failing_jobs = len([j for j in jobs if j["color"] in ["red", "yellow"]])
            
            # Get queue info
            queue_url = f"{self.config.base_url}/queue/api/json"
            queue_response = self.session.get(queue_url)
            queue_size = 0
            if queue_response.status_code == 200:
                queue_size = len(queue_response.json().get("items", []))
                
            return {
                "jenkins_url": self.config.base_url,
                "total_jobs": total_jobs,
                "healthy_jobs": healthy_jobs,
                "failing_jobs": failing_jobs,
                "health_percentage": (healthy_jobs / total_jobs * 100) if total_jobs > 0 else 0,
                "queue_size": queue_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to get Jenkins health: {e}")
            return {"error": str(e)}
            
    def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Jenkins webhook notifications"""
        # Jenkins webhook payloads vary based on plugins
        # This is a generic handler for common webhook formats
        
        job_name = payload.get("name") or payload.get("job", {}).get("name")
        build_number = payload.get("number") or payload.get("build", {}).get("number")
        
        if not job_name or not build_number:
            return {"status": "ignored", "reason": "Missing job name or build number"}
            
        pipeline_id = f"{job_name}#{build_number}"
        
        # Handle different webhook events
        if payload.get("status") == "STARTED":
            # Build started
            build_data = {
                "number": build_number,
                "id": pipeline_id,
                "building": True,
                "timestamp": datetime.now().timestamp() * 1000,
                "url": f"{self.config.base_url}/job/{job_name}/{build_number}/"
            }
            self._register_pipeline_build(job_name, build_data)
            return {"status": "registered", "pipeline_id": pipeline_id}
            
        elif payload.get("status") in ["SUCCESS", "FAILURE", "ABORTED", "UNSTABLE"]:
            # Build completed
            result = payload.get("status")
            
            self.monitor.update_pipeline_status(
                pipeline_id=pipeline_id,
                status="success" if result == "SUCCESS" else "failed",
                failure_reason=result if result != "SUCCESS" else None
            )
            
            if result in ["FAILURE", "ABORTED", "UNSTABLE"]:
                build_data = {
                    "number": build_number,
                    "result": result,
                    "building": False,
                    "url": f"{self.config.base_url}/job/{job_name}/{build_number}/"
                }
                self._handle_failed_build(job_name, build_data)
                
            return {"status": "processed", "pipeline_id": pipeline_id, "result": result}
            
        return {"status": "ignored", "payload": payload}
        
    def get_build_artifacts(self, job_name: str, build_number: int) -> List[Dict[str, Any]]:
        """Get artifacts from a Jenkins build"""
        url = f"{self.config.base_url}/job/{job_name}/{build_number}/api/json?tree=artifacts[fileName,relativePath]"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("artifacts", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to get artifacts for {job_name}#{build_number}: {e}")
            return []
            
    def download_artifact(self, job_name: str, build_number: int, 
                         artifact_path: str) -> Optional[bytes]:
        """Download a specific artifact from a build"""
        url = f"{self.config.base_url}/job/{job_name}/{build_number}/artifact/{artifact_path}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            self.logger.error(f"Failed to download artifact {artifact_path}: {e}")
            return None