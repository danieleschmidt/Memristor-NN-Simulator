"""
Self-Healing Pipeline Guard

A comprehensive CI/CD pipeline monitoring and self-healing system that automatically
detects, diagnoses, and repairs common pipeline failures.
"""

__version__ = "1.0.0"
__author__ = "Terragon Labs"

from .core.pipeline_monitor import PipelineMonitor
from .core.healing_engine import HealingEngine
from .core.failure_detector import FailureDetector

# Import integrations conditionally to handle missing dependencies
try:
    from .integrations.github_actions import GitHubActionsIntegration
    _GITHUB_AVAILABLE = True
except ImportError:
    _GITHUB_AVAILABLE = False

try:
    from .integrations.jenkins import JenkinsIntegration
    _JENKINS_AVAILABLE = True
except ImportError:
    _JENKINS_AVAILABLE = False

try:
    from .integrations.gitlab_ci import GitLabCIIntegration
    _GITLAB_AVAILABLE = True
except ImportError:
    _GITLAB_AVAILABLE = False

__all__ = [
    "PipelineMonitor",
    "HealingEngine", 
    "FailureDetector"
]

if _GITHUB_AVAILABLE:
    __all__.append("GitHubActionsIntegration")
if _JENKINS_AVAILABLE:
    __all__.append("JenkinsIntegration")
if _GITLAB_AVAILABLE:
    __all__.append("GitLabCIIntegration")