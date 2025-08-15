"""
Pipeline Guard - Self-Healing CI/CD Pipeline Monitoring System
Main application entry point with CLI interface
"""

import os
import sys
import json
import click
import logging
from pathlib import Path
from typing import Dict, Any

# Add pipeline_guard to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pipeline_guard.core.pipeline_monitor import PipelineMonitor
from pipeline_guard.core.failure_detector import FailureDetector
from pipeline_guard.core.healing_engine import HealingEngine
from pipeline_guard.integrations.github_actions import GitHubActionsIntegration, GitHubActionsConfig
from pipeline_guard.integrations.jenkins import JenkinsIntegration, JenkinsConfig
from pipeline_guard.integrations.gitlab_ci import GitLabCIIntegration, GitLabCIConfig
from pipeline_guard.utils.security import SecurityManager, SecurityConfig
from pipeline_guard.utils.error_handling import ErrorHandler
from pipeline_guard.utils.logging import setup_logging, StructuredLogger
from pipeline_guard.utils.validators import ConfigValidator
from pipeline_guard.scaling.auto_scaler import AutoScaler, ScalingPolicy, ScalingMetric
from pipeline_guard.scaling.performance_monitor import PerformanceMonitor


class PipelineGuardApp:
    """Main Pipeline Guard application"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = {}
        self.monitor = None
        self.detector = None
        self.healer = None
        self.security_manager = None
        self.error_handler = None
        self.performance_monitor = None
        self.auto_scaler = None
        self.integrations = {}
        self.logger = None
        
        # Load configuration
        self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self):
        """Load configuration from file"""
        default_config = {
            "logging": {
                "level": "INFO",
                "file": "pipeline_guard.log",
                "structured": True
            },
            "monitoring": {
                "check_interval": 30,
                "enable_auto_healing": True
            },
            "security": {
                "max_requests_per_window": 1000,
                "rate_limit_window": 3600,
                "max_file_size": 50 * 1024 * 1024
            },
            "scaling": {
                "enabled": True,
                "min_instances": 1,
                "max_instances": 10,
                "scale_up_cooldown": 300,
                "scale_down_cooldown": 600
            },
            "integrations": {}
        }
        
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    default_config.update(file_config)
            except Exception as e:
                print(f"Error loading config file: {e}")
                
        self.config = default_config
        
    def _setup_logging(self):
        """Setup application logging"""
        log_config = self.config.get("logging", {})
        
        setup_logging(
            log_level=log_config.get("level", "INFO"),
            log_file=log_config.get("file"),
            structured=log_config.get("structured", True)
        )
        
        self.logger = StructuredLogger(__name__)
        self.logger.logger.info("Pipeline Guard application starting")
        
    def _initialize_components(self):
        """Initialize all application components"""
        try:
            # Security manager
            security_config = SecurityConfig(**self.config.get("security", {}))
            self.security_manager = SecurityManager(security_config)
            
            # Error handler
            self.error_handler = ErrorHandler()
            
            # Core components
            monitor_config = self.config.get("monitoring", {})
            self.monitor = PipelineMonitor(
                check_interval=monitor_config.get("check_interval", 30)
            )
            
            self.detector = FailureDetector()
            self.healer = HealingEngine()
            self.healer.healing_enabled = monitor_config.get("enable_auto_healing", True)
            
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor()
            
            # Auto-scaling
            scaling_config = self.config.get("scaling", {})
            if scaling_config.get("enabled", True):
                scaling_policy = ScalingPolicy(
                    min_instances=scaling_config.get("min_instances", 1),
                    max_instances=scaling_config.get("max_instances", 10),
                    scale_up_cooldown=scaling_config.get("scale_up_cooldown", 300),
                    scale_down_cooldown=scaling_config.get("scale_down_cooldown", 600)
                )
                self.auto_scaler = AutoScaler(scaling_policy)
                
            # Initialize integrations
            self._setup_integrations()
            
            self.logger.logger.info("All components initialized successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Failed to initialize components: {e}")
            else:
                print(f"Failed to initialize components: {e}")
            raise
            
    def _setup_integrations(self):
        """Setup CI/CD platform integrations"""
        integrations_config = self.config.get("integrations", {})
        
        # GitHub Actions
        if "github" in integrations_config:
            try:
                github_config = GitHubActionsConfig(**integrations_config["github"])
                self.integrations["github"] = GitHubActionsIntegration(
                    github_config, self.monitor, self.detector, self.healer
                )
                self.logger.logger.info("GitHub Actions integration configured")
            except Exception as e:
                self.logger.logger.error(f"Failed to setup GitHub integration: {e}")
                
        # Jenkins
        if "jenkins" in integrations_config:
            try:
                jenkins_config = JenkinsConfig(**integrations_config["jenkins"])
                self.integrations["jenkins"] = JenkinsIntegration(
                    jenkins_config, self.monitor, self.detector, self.healer
                )
                self.logger.logger.info("Jenkins integration configured")
            except Exception as e:
                self.logger.logger.error(f"Failed to setup Jenkins integration: {e}")
                
        # GitLab CI
        if "gitlab" in integrations_config:
            try:
                gitlab_config = GitLabCIConfig(**integrations_config["gitlab"])
                self.integrations["gitlab"] = GitLabCIIntegration(
                    gitlab_config, self.monitor, self.detector, self.healer
                )
                self.logger.logger.info("GitLab CI integration configured")
            except Exception as e:
                self.logger.logger.error(f"Failed to setup GitLab integration: {e}")
                
    def start(self):
        """Start the Pipeline Guard application"""
        try:
            self.logger.logger.info("Starting Pipeline Guard services")
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Start performance monitoring
            if self.performance_monitor:
                # Performance monitor starts automatically
                pass
                
            # Start auto-scaling
            if self.auto_scaler:
                self.auto_scaler.start_monitoring()
                
            # Start integrations
            for name, integration in self.integrations.items():
                integration.start_monitoring()
                self.logger.logger.info(f"Started {name} integration")
                
            self.logger.logger.info("Pipeline Guard started successfully")
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "start_application"})
            raise
            
    def stop(self):
        """Stop the Pipeline Guard application"""
        try:
            self.logger.logger.info("Stopping Pipeline Guard services")
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
                
            # Stop auto-scaling
            if self.auto_scaler:
                self.auto_scaler.stop_monitoring()
                
            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop()
                
            self.logger.logger.info("Pipeline Guard stopped successfully")
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Error stopping application: {e}")
            else:
                print(f"Error stopping application: {e}")
                
    def get_status(self) -> Dict[str, Any]:
        """Get application status"""
        try:
            status = {
                "application": "Pipeline Guard",
                "version": "1.0.0",
                "status": "running",
                "timestamp": self.monitor.get_health_summary()["timestamp"] if self.monitor else None
            }
            
            if self.monitor:
                status["monitoring"] = self.monitor.get_health_summary()
                
            if self.performance_monitor:
                status["performance"] = self.performance_monitor.get_performance_report()
                
            if self.auto_scaler:
                status["scaling"] = self.auto_scaler.get_scaling_status()
                
            if self.security_manager:
                status["security"] = self.security_manager.get_security_summary()
                
            # Integration status
            integration_status = {}
            for name, integration in self.integrations.items():
                if hasattr(integration, 'get_repository_health'):
                    integration_status[name] = integration.get_repository_health()
                elif hasattr(integration, 'get_project_health'):
                    integration_status[name] = integration.get_project_health()
                elif hasattr(integration, 'get_jenkins_health'):
                    integration_status[name] = integration.get_jenkins_health()
                else:
                    integration_status[name] = {"status": "active"}
                    
            status["integrations"] = integration_status
            
            return status
            
        except Exception as e:
            self.error_handler.handle_error(e, {"operation": "get_status"})
            return {"status": "error", "message": str(e)}


# CLI Interface
@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Pipeline Guard - Self-Healing CI/CD Pipeline Monitoring System"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose


@cli.command()
@click.pass_context
def start(ctx):
    """Start Pipeline Guard monitoring"""
    try:
        app = PipelineGuardApp(ctx.obj.get('config'))
        app.start()
        
        click.echo("Pipeline Guard started successfully!")
        click.echo("Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        import signal
        import time
        
        def signal_handler(sig, frame):
            click.echo("\nStopping Pipeline Guard...")
            app.stop()
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while True:
            time.sleep(1)
            
    except Exception as e:
        click.echo(f"Error starting Pipeline Guard: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Get Pipeline Guard status"""
    try:
        app = PipelineGuardApp(ctx.obj.get('config'))
        status_info = app.get_status()
        
        if ctx.obj.get('verbose'):
            click.echo(json.dumps(status_info, indent=2))
        else:
            click.echo(f"Status: {status_info.get('status', 'unknown')}")
            if 'monitoring' in status_info:
                monitoring = status_info['monitoring']
                click.echo(f"Pipelines: {monitoring.get('total_pipelines', 0)} total, "
                          f"{monitoring.get('running_pipelines', 0)} running, "
                          f"{monitoring.get('failed_pipelines', 0)} failed")
                          
    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--pattern', help='Failure pattern to test')
@click.option('--logs', help='Log file to analyze')
@click.pass_context
def detect(ctx, pattern, logs):
    """Test failure detection"""
    try:
        app = PipelineGuardApp(ctx.obj.get('config'))
        
        if logs and os.path.exists(logs):
            with open(logs, 'r') as f:
                log_content = f.read()
        elif pattern:
            log_content = pattern
        else:
            click.echo("Please provide either --pattern or --logs", err=True)
            return
            
        detection = app.detector.detect_failure(log_content)
        
        click.echo(f"Detection Result:")
        click.echo(f"  Detected: {detection.detected}")
        click.echo(f"  Pattern: {detection.pattern_id}")
        click.echo(f"  Confidence: {detection.confidence:.2f}")
        click.echo(f"  Type: {detection.failure_type}")
        click.echo(f"  Suggested Remediation: {detection.suggested_remediation}")
        
    except Exception as e:
        click.echo(f"Error in detection: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--format', type=click.Choice(['json', 'table']), default='table')
@click.pass_context
def metrics(ctx, format):
    """Show performance metrics"""
    try:
        app = PipelineGuardApp(ctx.obj.get('config'))
        
        if not app.performance_monitor:
            click.echo("Performance monitoring not enabled", err=True)
            return
            
        report = app.performance_monitor.get_performance_report()
        
        if format == 'json':
            click.echo(json.dumps(report, indent=2))
        else:
            click.echo("Performance Metrics:")
            click.echo("=" * 50)
            
            # System status
            system_status = report.get('system_status', {})
            click.echo(f"System Status: {system_status.get('status', 'unknown')}")
            click.echo(f"Message: {system_status.get('message', 'N/A')}")
            click.echo()
            
            # Key metrics
            metrics_data = report.get('metrics', {})
            for metric_name, metric_info in metrics_data.items():
                if metric_name.startswith('system_'):
                    stats = metric_info.get('statistics', {})
                    current = metric_info.get('current_value', 'N/A')
                    click.echo(f"{metric_name}: {current} {metric_info.get('unit', '')}")
                    
    except Exception as e:
        click.echo(f"Error getting metrics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def config_template(ctx):
    """Generate configuration template"""
    template = {
        "logging": {
            "level": "INFO",
            "file": "pipeline_guard.log",
            "structured": True
        },
        "monitoring": {
            "check_interval": 30,
            "enable_auto_healing": True
        },
        "security": {
            "max_requests_per_window": 1000,
            "rate_limit_window": 3600,
            "max_file_size": 52428800
        },
        "scaling": {
            "enabled": True,
            "min_instances": 1,
            "max_instances": 10,
            "scale_up_cooldown": 300,
            "scale_down_cooldown": 600
        },
        "integrations": {
            "github": {
                "token": "${GITHUB_TOKEN}",
                "repo_owner": "${GITHUB_REPO_OWNER}",
                "repo_name": "${GITHUB_REPO_NAME}",
                "webhook_secret": "${GITHUB_WEBHOOK_SECRET}"
            },
            "jenkins": {
                "base_url": "${JENKINS_URL}",
                "username": "${JENKINS_USERNAME}",
                "api_token": "${JENKINS_API_TOKEN}"
            },
            "gitlab": {
                "base_url": "${GITLAB_URL:-https://gitlab.com}",
                "project_id": "${GITLAB_PROJECT_ID}",
                "private_token": "${GITLAB_PRIVATE_TOKEN}"
            }
        }
    }
    
    click.echo(json.dumps(template, indent=2))


if __name__ == '__main__':
    cli()