"""
Security utilities for pipeline guard
"""

import os
import re
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    max_concurrent_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    max_requests_per_window: int = 1000
    allowed_file_extensions: Set[str] = None
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    webhook_timeout: int = 30
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = {'.txt', '.log', '.json', '.yaml', '.yml'}


class SecurityValidator:
    """
    Security validation utilities for input sanitization and safety checks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common injection patterns
        self.injection_patterns = [
            r'<script.*?>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'on\w+\s*=',  # Event handlers
            r'\\x[0-9a-f]{2}',  # Hex encoding
            r'%[0-9a-f]{2}',  # URL encoding
            r'\b(union|select|insert|update|delete|drop|exec|script)\b',  # SQL injection
            r'\.\./',  # Directory traversal
            r'\\\\',  # Windows path traversal
            r'\$\{.*?\}',  # Template injection
            r'`[^`]*`',  # Command injection
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.injection_patterns]
        
    def validate_input(self, input_string: str, max_length: int = 10000) -> bool:
        """
        Validate input string for potential security issues
        """
        if not isinstance(input_string, str):
            return False
            
        if len(input_string) > max_length:
            self.logger.warning(f"Input exceeds maximum length: {len(input_string)} > {max_length}")
            return False
            
        # Check for injection patterns
        for pattern in self.compiled_patterns:
            if pattern.search(input_string):
                self.logger.warning(f"Potential injection detected: {pattern.pattern}")
                return False
                
        return True
        
    def sanitize_log_content(self, content: str) -> str:
        """
        Sanitize log content to prevent log injection
        """
        if not isinstance(content, str):
            return str(content)
            
        # Remove potential ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        content = ansi_escape.sub('', content)
        
        # Replace newlines to prevent log injection
        content = content.replace('\n', '\\n').replace('\r', '\\r')
        
        # Limit length
        max_log_length = 5000
        if len(content) > max_log_length:
            content = content[:max_log_length] + "... [TRUNCATED]"
            
        return content
        
    def validate_file_path(self, file_path: str, allowed_dirs: List[str] = None) -> bool:
        """
        Validate file path to prevent directory traversal
        """
        if not isinstance(file_path, str):
            return False
            
        # Normalize path
        normalized_path = os.path.normpath(file_path)
        
        # Check for directory traversal
        if '..' in normalized_path or normalized_path.startswith('/'):
            self.logger.warning(f"Potential directory traversal: {file_path}")
            return False
            
        # Check against allowed directories
        if allowed_dirs:
            path_allowed = any(normalized_path.startswith(allowed_dir) for allowed_dir in allowed_dirs)
            if not path_allowed:
                self.logger.warning(f"Path not in allowed directories: {file_path}")
                return False
                
        return True
        
    def validate_webhook_signature(self, payload: str, signature: str, secret: str) -> bool:
        """
        Validate webhook signature for authenticity
        """
        if not all([payload, signature, secret]):
            return False
            
        # GitHub style signature (sha256=...)
        if signature.startswith('sha256='):
            expected_signature = 'sha256=' + hmac.new(
                secret.encode('utf-8'),
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        # GitLab style signature (X-Gitlab-Token)
        return hmac.compare_digest(signature, secret)
        
    def check_resource_limits(self, content_size: int, config: SecurityConfig) -> bool:
        """
        Check if resource usage is within limits
        """
        if content_size > config.max_log_size:
            self.logger.warning(f"Content size exceeds limit: {content_size} > {config.max_log_size}")
            return False
            
        return True


class RateLimiter:
    """
    Rate limiting for API endpoints and webhook handlers
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.request_counts: Dict[str, List[float]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed based on rate limits
        """
        current_time = time.time()
        window_start = current_time - self.config.rate_limit_window
        
        # Clean old requests
        self.request_counts[identifier] = [
            req_time for req_time in self.request_counts[identifier]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.request_counts[identifier]) >= self.config.max_requests_per_window:
            self.logger.warning(f"Rate limit exceeded for {identifier}")
            return False
            
        # Add current request
        self.request_counts[identifier].append(current_time)
        return True
        
    def get_remaining_requests(self, identifier: str) -> int:
        """
        Get remaining requests in current window
        """
        current_count = len(self.request_counts.get(identifier, []))
        return max(0, self.config.max_requests_per_window - current_count)


class SecurityManager:
    """
    Comprehensive security management for pipeline guard
    """
    
    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.validator = SecurityValidator()
        self.rate_limiter = RateLimiter(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Security event tracking
        self.security_events: List[Dict[str, Any]] = []
        self.blocked_ips: Set[str] = set()
        
    def validate_request(self, request_data: Dict[str, Any], 
                        client_ip: str = None) -> Dict[str, Any]:
        """
        Comprehensive request validation
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Rate limiting
        if client_ip and not self.rate_limiter.is_allowed(client_ip):
            validation_result["valid"] = False
            validation_result["errors"].append("Rate limit exceeded")
            self._log_security_event("rate_limit_exceeded", {"ip": client_ip})
            
        # Input validation
        for key, value in request_data.items():
            if isinstance(value, str):
                if not self.validator.validate_input(value):
                    validation_result["valid"] = False
                    validation_result["errors"].append(f"Invalid input in field: {key}")
                    self._log_security_event("invalid_input", {"field": key, "ip": client_ip})
                    
        # Check for blocked IPs
        if client_ip in self.blocked_ips:
            validation_result["valid"] = False
            validation_result["errors"].append("IP address blocked")
            
        return validation_result
        
    def validate_webhook_payload(self, payload: str, headers: Dict[str, str], 
                                webhook_secret: str = None) -> Dict[str, Any]:
        """
        Validate webhook payload for security
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Size validation
        if len(payload) > self.config.max_log_size:
            validation_result["valid"] = False
            validation_result["errors"].append("Payload too large")
            
        # Signature validation
        if webhook_secret:
            signature = headers.get('x-hub-signature-256') or headers.get('x-gitlab-token')
            if signature:
                if not self.validator.validate_webhook_signature(payload, signature, webhook_secret):
                    validation_result["valid"] = False
                    validation_result["errors"].append("Invalid webhook signature")
                    self._log_security_event("invalid_webhook_signature", {})
            else:
                validation_result["warnings"].append("No signature provided for webhook")
                
        # Content validation
        if not self.validator.validate_input(payload, max_length=self.config.max_log_size):
            validation_result["warnings"].append("Potentially unsafe content in payload")
            
        return validation_result
        
    def _log_security_event(self, event_type: str, metadata: Dict[str, Any]):
        """
        Log security events for monitoring
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "metadata": metadata
        }
        
        self.security_events.append(event)
        self.logger.warning(f"Security event: {event_type} - {metadata}")
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
            
    def block_ip(self, ip_address: str, reason: str):
        """
        Block an IP address
        """
        self.blocked_ips.add(ip_address)
        self._log_security_event("ip_blocked", {"ip": ip_address, "reason": reason})
        
    def unblock_ip(self, ip_address: str):
        """
        Unblock an IP address
        """
        self.blocked_ips.discard(ip_address)
        self._log_security_event("ip_unblocked", {"ip": ip_address})
        
    def get_security_summary(self) -> Dict[str, Any]:
        """
        Get security event summary
        """
        recent_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event["event_type"]] += 1
            
        return {
            "total_events_24h": len(recent_events),
            "event_types": dict(event_counts),
            "blocked_ips": list(self.blocked_ips),
            "rate_limit_status": {
                "max_requests_per_window": self.config.max_requests_per_window,
                "window_seconds": self.config.rate_limit_window
            },
            "timestamp": datetime.now().isoformat()
        }
        
    def export_security_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Export security events for analysis
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]
        
    def clean_old_events(self, days: int = 7):
        """
        Clean up old security events
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        
        self.security_events = [
            event for event in self.security_events
            if datetime.fromisoformat(event["timestamp"]) > cutoff_time
        ]
        
        self.logger.info(f"Cleaned old security events, {len(self.security_events)} events remaining")