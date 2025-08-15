"""
Advanced logging utilities for pipeline guard
"""

import os
import sys
import json
import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    JSON structured logging formatter
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'pipeline_id'):
            log_entry['pipeline_id'] = record.pipeline_id
        if hasattr(record, 'error_context'):
            log_entry['error_context'] = record.error_context
        if hasattr(record, 'stack_trace'):
            log_entry['stack_trace'] = record.stack_trace
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


class PipelineContextFilter(logging.Filter):
    """
    Filter to add pipeline context to log records
    """
    
    def __init__(self):
        super().__init__()
        self.current_pipeline_id = None
        
    def set_pipeline_context(self, pipeline_id: str):
        """Set current pipeline context"""
        self.current_pipeline_id = pipeline_id
        
    def clear_pipeline_context(self):
        """Clear pipeline context"""
        self.current_pipeline_id = None
        
    def filter(self, record: logging.LogRecord) -> bool:
        """Add pipeline context to log record"""
        if self.current_pipeline_id:
            record.pipeline_id = self.current_pipeline_id
        return True


class StructuredLogger:
    """
    High-level structured logger for pipeline guard
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context_filter = PipelineContextFilter()
        self.logger.addFilter(self.context_filter)
        
    def set_pipeline_context(self, pipeline_id: str):
        """Set pipeline context for subsequent logs"""
        self.context_filter.set_pipeline_context(pipeline_id)
        
    def clear_pipeline_context(self):
        """Clear pipeline context"""
        self.context_filter.clear_pipeline_context()
        
    def log_pipeline_event(self, level: str, message: str, 
                          pipeline_id: str = None, **kwargs):
        """Log pipeline-specific event"""
        extra = kwargs.copy()
        if pipeline_id:
            extra['pipeline_id'] = pipeline_id
            
        getattr(self.logger, level.lower())(message, extra=extra)
        
    def log_failure_detection(self, pattern_id: str, confidence: float, 
                            pipeline_id: str, details: Dict[str, Any]):
        """Log failure detection event"""
        self.logger.info(
            f"Failure pattern detected: {pattern_id}",
            extra={
                'pipeline_id': pipeline_id,
                'pattern_id': pattern_id,
                'confidence': confidence,
                'detection_details': details,
                'event_type': 'failure_detection'
            }
        )
        
    def log_healing_attempt(self, pipeline_id: str, strategy: str, 
                           actions: list, results: Dict[str, Any]):
        """Log healing attempt"""
        self.logger.info(
            f"Healing attempt for pipeline {pipeline_id}",
            extra={
                'pipeline_id': pipeline_id,
                'healing_strategy': strategy,
                'actions_attempted': actions,
                'healing_results': results,
                'event_type': 'healing_attempt'
            }
        )
        
    def log_security_event(self, event_type: str, severity: str, 
                          details: Dict[str, Any]):
        """Log security event"""
        getattr(self.logger, severity.lower())(
            f"Security event: {event_type}",
            extra={
                'security_event_type': event_type,
                'security_details': details,
                'event_type': 'security'
            }
        )
        
    def log_performance_metric(self, metric_name: str, value: float, 
                              unit: str, context: Dict[str, Any] = None):
        """Log performance metric"""
        self.logger.info(
            f"Performance metric: {metric_name} = {value} {unit}",
            extra={
                'metric_name': metric_name,
                'metric_value': value,
                'metric_unit': unit,
                'metric_context': context or {},
                'event_type': 'performance_metric'
            }
        )


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging for pipeline guard
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Setup formatters
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler (always in structured format)
    if log_file:
        error_log_file = str(log_path.with_suffix('.error.log'))
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    root_logger.info(f"Logging setup complete - Level: {log_level}, Structured: {structured}")
    
    return root_logger


class LogAnalyzer:
    """
    Analyze log files for patterns and insights
    """
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        
    def analyze_error_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns from log file"""
        if not os.path.exists(self.log_file):
            return {"error": "Log file not found"}
            
        error_counts = {}
        pipeline_errors = {}
        recent_cutoff = datetime.now().timestamp() - (hours * 3600)
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Check if log entry is recent enough
                        log_time = datetime.fromisoformat(log_entry.get('timestamp', '')).timestamp()
                        if log_time < recent_cutoff:
                            continue
                            
                        # Count error levels
                        level = log_entry.get('level', '')
                        if level in ['ERROR', 'CRITICAL']:
                            error_counts[level] = error_counts.get(level, 0) + 1
                            
                        # Count pipeline-specific errors
                        pipeline_id = log_entry.get('pipeline_id')
                        if pipeline_id and level in ['ERROR', 'CRITICAL']:
                            pipeline_errors[pipeline_id] = pipeline_errors.get(pipeline_id, 0) + 1
                            
                    except (json.JSONDecodeError, ValueError):
                        continue  # Skip malformed log entries
                        
        except IOError as e:
            return {"error": f"Failed to read log file: {e}"}
            
        return {
            "analysis_period_hours": hours,
            "error_counts_by_level": error_counts,
            "pipeline_error_counts": pipeline_errors,
            "total_errors": sum(error_counts.values()),
            "most_problematic_pipelines": sorted(
                pipeline_errors.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
        
    def get_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Extract performance metrics from logs"""
        if not os.path.exists(self.log_file):
            return {"error": "Log file not found"}
            
        metrics = {}
        recent_cutoff = datetime.now().timestamp() - (hours * 3600)
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Check if log entry is recent enough
                        log_time = datetime.fromisoformat(log_entry.get('timestamp', '')).timestamp()
                        if log_time < recent_cutoff:
                            continue
                            
                        # Extract performance metrics
                        if log_entry.get('event_type') == 'performance_metric':
                            metric_name = log_entry.get('metric_name')
                            metric_value = log_entry.get('metric_value')
                            
                            if metric_name and metric_value is not None:
                                if metric_name not in metrics:
                                    metrics[metric_name] = []
                                metrics[metric_name].append(metric_value)
                                
                    except (json.JSONDecodeError, ValueError):
                        continue
                        
        except IOError as e:
            return {"error": f"Failed to read log file: {e}"}
            
        # Calculate statistics
        metric_stats = {}
        for metric_name, values in metrics.items():
            if values:
                metric_stats[metric_name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
                
        return {
            "analysis_period_hours": hours,
            "metric_statistics": metric_stats,
            "timestamp": datetime.now().isoformat()
        }