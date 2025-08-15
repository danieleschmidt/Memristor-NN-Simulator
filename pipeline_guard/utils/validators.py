"""
Input validation utilities for pipeline guard
"""

import re
import json
import ipaddress
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse


@dataclass
class ValidationResult:
    """Result of validation operation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Any = None


class InputValidator:
    """
    Comprehensive input validation for various data types
    """
    
    def __init__(self):
        # Common patterns
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')
        self.pipeline_id_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        self.safe_string_pattern = re.compile(r'^[a-zA-Z0-9\s\-_.,:;!?()]+$')
        
    def validate_string(self, value: str, 
                       max_length: int = 1000,
                       min_length: int = 0,
                       pattern: Optional[re.Pattern] = None,
                       allow_empty: bool = True) -> ValidationResult:
        """Validate string input"""
        errors = []
        warnings = []
        
        if not isinstance(value, str):
            errors.append("Value must be a string")
            return ValidationResult(False, errors, warnings)
            
        if not allow_empty and len(value) == 0:
            errors.append("Value cannot be empty")
            
        if len(value) < min_length:
            errors.append(f"Value too short (minimum {min_length} characters)")
            
        if len(value) > max_length:
            errors.append(f"Value too long (maximum {max_length} characters)")
            
        if pattern and not pattern.match(value):
            errors.append("Value doesn't match required pattern")
            
        # Check for potential security issues
        if self._contains_suspicious_content(value):
            warnings.append("Value contains potentially suspicious content")
            
        # Sanitize the value
        sanitized = self._sanitize_string(value)
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized
        )
        
    def validate_pipeline_id(self, pipeline_id: str) -> ValidationResult:
        """Validate pipeline ID format"""
        return self.validate_string(
            pipeline_id,
            max_length=100,
            min_length=1,
            pattern=self.pipeline_id_pattern,
            allow_empty=False
        )
        
    def validate_email(self, email: str) -> ValidationResult:
        """Validate email address"""
        result = self.validate_string(email, max_length=254, allow_empty=False)
        
        if result.valid and not self.email_pattern.match(email):
            result.valid = False
            result.errors.append("Invalid email format")
            
        return result
        
    def validate_url(self, url: str, allowed_schemes: List[str] = None) -> ValidationResult:
        """Validate URL format and scheme"""
        if allowed_schemes is None:
            allowed_schemes = ['http', 'https']
            
        result = self.validate_string(url, max_length=2048, allow_empty=False)
        
        if not result.valid:
            return result
            
        try:
            parsed = urlparse(url)
            
            if not parsed.scheme:
                result.errors.append("URL missing scheme")
                result.valid = False
                
            if parsed.scheme not in allowed_schemes:
                result.errors.append(f"URL scheme must be one of: {allowed_schemes}")
                result.valid = False
                
            if not parsed.netloc:
                result.errors.append("URL missing domain")
                result.valid = False
                
        except Exception as e:
            result.errors.append(f"Invalid URL format: {e}")
            result.valid = False
            
        return result
        
    def validate_ip_address(self, ip_str: str) -> ValidationResult:
        """Validate IP address (IPv4 or IPv6)"""
        errors = []
        warnings = []
        
        try:
            ip = ipaddress.ip_address(ip_str)
            
            # Check for private/local addresses
            if ip.is_private:
                warnings.append("IP address is private")
            if ip.is_loopback:
                warnings.append("IP address is loopback")
            if ip.is_multicast:
                warnings.append("IP address is multicast")
                
            return ValidationResult(True, errors, warnings, str(ip))
            
        except ValueError as e:
            errors.append(f"Invalid IP address: {e}")
            return ValidationResult(False, errors, warnings)
            
    def validate_json(self, json_str: str, schema: Dict = None) -> ValidationResult:
        """Validate JSON string and optionally against schema"""
        errors = []
        warnings = []
        
        try:
            parsed_json = json.loads(json_str)
            
            # Basic size check
            if len(json_str) > 1024 * 1024:  # 1MB
                warnings.append("JSON payload is very large")
                
            # Optional schema validation would go here
            if schema:
                # Could integrate with jsonschema library
                warnings.append("Schema validation not implemented")
                
            return ValidationResult(True, errors, warnings, parsed_json)
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return ValidationResult(False, errors, warnings)
            
    def validate_integer(self, value: Union[int, str], 
                        min_value: Optional[int] = None,
                        max_value: Optional[int] = None) -> ValidationResult:
        """Validate integer value"""
        errors = []
        warnings = []
        
        try:
            if isinstance(value, str):
                int_value = int(value)
            elif isinstance(value, int):
                int_value = value
            else:
                errors.append("Value must be an integer or string")
                return ValidationResult(False, errors, warnings)
                
            if min_value is not None and int_value < min_value:
                errors.append(f"Value must be at least {min_value}")
                
            if max_value is not None and int_value > max_value:
                errors.append(f"Value must be at most {max_value}")
                
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=int_value
            )
            
        except ValueError as e:
            errors.append(f"Invalid integer: {e}")
            return ValidationResult(False, errors, warnings)
            
    def validate_timestamp(self, timestamp: str) -> ValidationResult:
        """Validate ISO timestamp format"""
        errors = []
        warnings = []
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Check if timestamp is reasonable (not too far in past/future)
            now = datetime.now()
            if dt.year < 2020:
                warnings.append("Timestamp is very old")
            if dt > now:
                warnings.append("Timestamp is in the future")
                
            return ValidationResult(True, errors, warnings, dt)
            
        except ValueError as e:
            errors.append(f"Invalid timestamp format: {e}")
            return ValidationResult(False, errors, warnings)
            
    def _contains_suspicious_content(self, value: str) -> bool:
        """Check for potentially suspicious content"""
        suspicious_patterns = [
            r'<script.*?>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',   # Event handlers
            r'\\x[0-9a-f]{2}',  # Hex encoding
            r'\$\{.*?\}',   # Template injection
            r'`[^`]*`',     # Backticks (command injection)
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return True
                
        return False
        
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string by removing/escaping dangerous content"""
        # Remove null bytes
        sanitized = value.replace('\x00', '')
        
        # Remove or escape HTML/XML entities
        sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
        
        # Remove control characters except newlines and tabs
        sanitized = ''.join(char for char in sanitized 
                           if ord(char) >= 32 or char in '\n\t')
        
        return sanitized


class ConfigValidator:
    """
    Validate configuration objects and dictionaries
    """
    
    def __init__(self):
        self.input_validator = InputValidator()
        
    def validate_webhook_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate webhook configuration"""
        errors = []
        warnings = []
        required_fields = ['url', 'secret']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
                
        # Validate URL
        if 'url' in config:
            url_result = self.input_validator.validate_url(config['url'])
            if not url_result.valid:
                errors.extend([f"Invalid webhook URL: {error}" for error in url_result.errors])
                
        # Validate secret
        if 'secret' in config:
            secret_result = self.input_validator.validate_string(
                config['secret'], min_length=8, max_length=128
            )
            if not secret_result.valid:
                errors.extend([f"Invalid webhook secret: {error}" for error in secret_result.errors])
                
        # Validate optional timeout
        if 'timeout' in config:
            timeout_result = self.input_validator.validate_integer(
                config['timeout'], min_value=1, max_value=300
            )
            if not timeout_result.valid:
                errors.extend([f"Invalid timeout: {error}" for error in timeout_result.errors])
                
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_pipeline_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate pipeline configuration"""
        errors = []
        warnings = []
        
        # Validate pipeline ID
        if 'pipeline_id' in config:
            id_result = self.input_validator.validate_pipeline_id(config['pipeline_id'])
            if not id_result.valid:
                errors.extend([f"Invalid pipeline ID: {error}" for error in id_result.errors])
                
        # Validate name
        if 'name' in config:
            name_result = self.input_validator.validate_string(
                config['name'], max_length=200, min_length=1
            )
            if not name_result.valid:
                errors.extend([f"Invalid pipeline name: {error}" for error in name_result.errors])
                
        # Validate timeout settings
        if 'timeout' in config:
            timeout_result = self.input_validator.validate_integer(
                config['timeout'], min_value=60, max_value=3600  # 1 min to 1 hour
            )
            if not timeout_result.valid:
                errors.extend([f"Invalid timeout: {error}" for error in timeout_result.errors])
                
        # Validate retry settings
        if 'max_retries' in config:
            retry_result = self.input_validator.validate_integer(
                config['max_retries'], min_value=0, max_value=10
            )
            if not retry_result.valid:
                errors.extend([f"Invalid max_retries: {error}" for error in retry_result.errors])
                
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_integration_config(self, config: Dict[str, Any], 
                                  integration_type: str) -> ValidationResult:
        """Validate CI/CD integration configuration"""
        errors = []
        warnings = []
        
        if integration_type == 'github':
            required_fields = ['token', 'repo_owner', 'repo_name']
            
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field for GitHub integration: {field}")
                    
            # Validate API URL
            if 'api_base_url' in config:
                url_result = self.input_validator.validate_url(config['api_base_url'])
                if not url_result.valid:
                    errors.extend([f"Invalid GitHub API URL: {error}" for error in url_result.errors])
                    
        elif integration_type == 'jenkins':
            required_fields = ['base_url', 'username', 'api_token']
            
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field for Jenkins integration: {field}")
                    
        elif integration_type == 'gitlab':
            required_fields = ['base_url', 'project_id', 'private_token']
            
            for field in required_fields:
                if field not in config:
                    errors.append(f"Missing required field for GitLab integration: {field}")
                    
        else:
            errors.append(f"Unknown integration type: {integration_type}")
            
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    def validate_security_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate security configuration"""
        errors = []
        warnings = []
        
        # Validate rate limiting settings
        if 'max_requests_per_window' in config:
            rate_result = self.input_validator.validate_integer(
                config['max_requests_per_window'], min_value=1, max_value=10000
            )
            if not rate_result.valid:
                errors.extend([f"Invalid rate limit: {error}" for error in rate_result.errors])
                
        if 'rate_limit_window' in config:
            window_result = self.input_validator.validate_integer(
                config['rate_limit_window'], min_value=60, max_value=86400  # 1 min to 1 day
            )
            if not window_result.valid:
                errors.extend([f"Invalid rate limit window: {error}" for error in window_result.errors])
                
        # Validate file size limits
        if 'max_file_size' in config:
            size_result = self.input_validator.validate_integer(
                config['max_file_size'], min_value=1024, max_value=100*1024*1024  # 1KB to 100MB
            )
            if not size_result.valid:
                errors.extend([f"Invalid max file size: {error}" for error in size_result.errors])
                
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )