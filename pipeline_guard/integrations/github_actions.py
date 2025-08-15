"""
GitHub Actions Integration for Pipeline Guard
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..core.pipeline_monitor import PipelineStatus, PipelineMonitor
from ..core.failure_detector import FailureDetector
from ..core.healing_engine import HealingEngine


@dataclass
class GitHubActionsConfig:
    """Configuration for GitHub Actions integration"""
    token: str
    repo_owner: str
    repo_name: str
    webhook_secret: Optional[str] = None
    api_base_url: str = "https://api.github.com"


class GitHubActionsIntegration:
    """
    Integration with GitHub Actions for pipeline monitoring and healing
    """
    
    def __init__(self, config: GitHubActionsConfig, 
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
            "Authorization": f"token {config.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Pipeline-Guard/1.0"
        })
        
    def start_monitoring(self):
        """Start monitoring GitHub Actions workflows"""
        self.logger.info("Starting GitHub Actions monitoring")
        self.monitor.start_monitoring()
        
        # Register existing workflows
        workflows = self._get_workflows()
        for workflow in workflows:
            self._register_workflow_runs(workflow["id"])
            
    def _get_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows for the repository"""
        url = f"{self.config.api_base_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/actions/workflows"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json().get("workflows", [])
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch workflows: {e}")
            return []
            
    def _register_workflow_runs(self, workflow_id: int):
        """Register recent workflow runs for monitoring"""
        url = f"{self.config.api_base_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/actions/workflows/{workflow_id}/runs"
        
        try:
            response = self.session.get(url, params={"per_page": 10})
            response.raise_for_status()
            
            runs = response.json().get("workflow_runs", [])
            for run in runs:
                if run["status"] in ["in_progress", "queued"]:
                    self._register_pipeline_run(run)
                elif run["conclusion"] in ["failure", "cancelled"]:
                    self._handle_failed_run(run)
                    
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch workflow runs: {e}")
            
    def _register_pipeline_run(self, run: Dict[str, Any]):
        """Register a workflow run with the monitor"""
        pipeline_id = str(run["id"])
        
        self.monitor.register_pipeline(
            pipeline_id=pipeline_id,
            name=f"{run['name']} #{run['run_number']}",
            metadata={
                "workflow_id": run["workflow_id"],
                "run_number": run["run_number"],
                "head_sha": run["head_sha"],
                "head_branch": run["head_branch"],
                "event": run["event"],
                "html_url": run["html_url"]
            }
        )
        
    def _handle_failed_run(self, run: Dict[str, Any]):
        """Handle a failed workflow run"""
        pipeline_id = str(run["id"])
        
        # Get detailed logs for failure analysis
        logs = self._get_workflow_logs(run["id"])
        
        # Detect failure pattern
        detection = self.detector.detect_failure(logs, {
            "pipeline_id": pipeline_id,
            "workflow_name": run["name"],
            "conclusion": run["conclusion"]
        })
        
        if detection.detected:
            self.logger.info(f"Detected failure pattern '{detection.pattern_id}' in run {pipeline_id}")
            
            # Attempt healing if enabled
            if self.healer.healing_enabled:
                healing_results = self.healer.heal_pipeline(
                    pipeline_id=pipeline_id,
                    failure_pattern=detection.suggested_remediation,
                    context={
                        "logs": logs,
                        "run_data": run,
                        "detection": detection
                    }
                )
                
                # If healing was successful, potentially re-trigger the workflow
                if any(r.status.value == "success" for r in healing_results):
                    self._consider_workflow_retrigger(run, detection)
                    
    def _get_workflow_logs(self, run_id: int) -> str:
        """Get logs for a workflow run"""
        url = f"{self.config.api_base_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/actions/runs/{run_id}/logs"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            # GitHub returns a ZIP file with logs
            # For simplicity, we'll return a placeholder
            # In production, you'd extract and parse the ZIP
            return f"Logs for run {run_id} (ZIP content would be parsed here)"
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch logs for run {run_id}: {e}")
            return ""
            
    def _consider_workflow_retrigger(self, run: Dict[str, Any], detection):
        """Consider whether to re-trigger a workflow after healing"""
        # Only re-trigger for certain failure types
        retriggerable_patterns = [
            "dependency_failure",
            "network_failure", 
            "timeout_failure",
            "resource_failure"
        ]
        
        if detection.pattern_id in retriggerable_patterns and detection.confidence > 0.8:
            self.logger.info(f"Considering re-trigger for run {run['id']}")
            # Note: GitHub doesn't have a direct re-run API for failed runs
            # This would typically involve creating a new workflow dispatch
            return True
            
        return False
        
    def handle_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub Actions webhook events"""
        action = payload.get("action")
        workflow_run = payload.get("workflow_run")
        
        if not workflow_run:
            return {"status": "ignored", "reason": "No workflow_run in payload"}
            
        pipeline_id = str(workflow_run["id"])
        
        if action == "requested":
            # Workflow run started
            self._register_pipeline_run(workflow_run)
            return {"status": "registered", "pipeline_id": pipeline_id}
            
        elif action == "completed":
            # Workflow run completed
            conclusion = workflow_run.get("conclusion")
            
            self.monitor.update_pipeline_status(
                pipeline_id=pipeline_id,
                status="success" if conclusion == "success" else "failed",
                failure_reason=conclusion if conclusion != "success" else None
            )
            
            if conclusion in ["failure", "cancelled"]:
                self._handle_failed_run(workflow_run)
                
            return {"status": "processed", "pipeline_id": pipeline_id, "conclusion": conclusion}
            
        return {"status": "ignored", "action": action}
        
    def get_repository_health(self) -> Dict[str, Any]:
        """Get overall repository health metrics"""
        # Get recent workflow runs
        url = f"{self.config.api_base_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/actions/runs"
        
        try:
            response = self.session.get(url, params={"per_page": 50})
            response.raise_for_status()
            
            runs = response.json().get("workflow_runs", [])
            
            total_runs = len(runs)
            successful_runs = len([r for r in runs if r["conclusion"] == "success"])
            failed_runs = len([r for r in runs if r["conclusion"] == "failure"])
            
            # Calculate average duration
            completed_runs = [r for r in runs if r["conclusion"] in ["success", "failure"]]
            avg_duration = 0
            if completed_runs:
                durations = []
                for run in completed_runs:
                    if run["created_at"] and run["updated_at"]:
                        start = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                        end = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                        durations.append((end - start).total_seconds())
                        
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    
            return {
                "repository": f"{self.config.repo_owner}/{self.config.repo_name}",
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
                "average_duration_seconds": avg_duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to get repository health: {e}")
            return {"error": str(e)}
            
    def create_issue_for_persistent_failure(self, failure_pattern: str, 
                                          occurrences: int) -> Optional[int]:
        """Create GitHub issue for persistent failures"""
        if occurrences < 3:  # Only create issue after multiple failures
            return None
            
        title = f"Persistent Pipeline Failure: {failure_pattern}"
        body = f"""
## Persistent Pipeline Failure Detected

**Failure Pattern**: {failure_pattern}
**Occurrences**: {occurrences}
**Detection Time**: {datetime.now().isoformat()}

### Automated Analysis
This issue was automatically created by the Pipeline Guard system due to repeated failures of the same type.

### Recommended Actions
1. Review recent pipeline runs for this failure pattern
2. Check if the automated healing attempts have been successful
3. Consider updating the healing strategy for this failure type
4. Investigate if this indicates a systemic issue

### Pipeline Guard Data
- Monitor: Active
- Healing Engine: {'Enabled' if self.healer.healing_enabled else 'Disabled'}
- Last Healing Attempt: Check Pipeline Guard logs

/label bug pipeline-failure automated
        """
        
        url = f"{self.config.api_base_url}/repos/{self.config.repo_owner}/{self.config.repo_name}/issues"
        
        try:
            response = self.session.post(url, json={
                "title": title,
                "body": body,
                "labels": ["bug", "pipeline-failure", "automated"]
            })
            response.raise_for_status()
            
            issue_data = response.json()
            self.logger.info(f"Created issue #{issue_data['number']} for persistent failure")
            return issue_data["number"]
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to create issue: {e}")
            return None