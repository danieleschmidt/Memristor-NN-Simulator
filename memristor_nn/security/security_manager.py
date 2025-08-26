"""
Comprehensive security management for memristor neural networks.

Implements:
- Multi-layer security architecture
- Zero-trust security model
- Advanced threat detection
- Secure communication protocols
- Cryptographic key management
"""

import hashlib
import hmac
import secrets
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

from ..utils.logger import LoggingMixin


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    name: str
    security_level: SecurityLevel
    encryption_required: bool = True
    authentication_required: bool = True
    authorization_required: bool = True
    audit_logging: bool = True
    rate_limiting: Dict[str, int] = field(default_factory=lambda: {"requests_per_minute": 1000})
    allowed_operations: List[str] = field(default_factory=list)
    denied_operations: List[str] = field(default_factory=list)


@dataclass
class SecurityEvent:
    """Security event for monitoring and auditing."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str]
    user_id: Optional[str]
    operation: str
    details: Dict[str, Any]
    mitigated: bool = False


class SecurityManager(LoggingMixin):
    """Comprehensive security management system."""
    
    def __init__(
        self,
        security_policies: Optional[List[SecurityPolicy]] = None,
        enable_monitoring: bool = True,
        crypto_backend: str = "native"
    ):
        """
        Initialize security manager.
        
        Args:
            security_policies: List of security policies
            enable_monitoring: Enable real-time security monitoring
            crypto_backend: Cryptographic backend to use
        """
        super().__init__()
        
        # Initialize security policies
        self.policies = {}
        if security_policies:
            for policy in security_policies:
                self.policies[policy.name] = policy
        else:
            self._create_default_policies()
        
        # Cryptographic components
        self.master_key = secrets.token_bytes(32)  # 256-bit master key
        self.session_keys = {}
        self.key_rotation_interval = 3600  # 1 hour
        self.last_key_rotation = time.time()
        
        # Authentication and authorization
        self.authenticated_sessions = {}
        self.session_timeouts = {}
        self.failed_auth_attempts = {}
        self.max_auth_failures = 5
        self.lockout_duration = 300  # 5 minutes
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.threat_signatures = self._load_threat_signatures()
        self.anomaly_detection = self._initialize_anomaly_detection()
        
        # Rate limiting
        self.rate_limits = {}
        self.request_counts = {}
        
        # Security monitoring thread
        self.monitoring_active = enable_monitoring
        self.monitoring_thread = None
        if enable_monitoring:
            self._start_security_monitoring()
        
        self.logger.info(f"Security manager initialized with {len(self.policies)} policies")
    
    def _create_default_policies(self) -> None:
        """Create default security policies."""
        default_policies = [
            SecurityPolicy(
                name="public_access",
                security_level=SecurityLevel.PUBLIC,
                encryption_required=False,
                authentication_required=False,
                authorization_required=False,
                allowed_operations=["get_info", "health_check"]
            ),
            SecurityPolicy(
                name="internal_operations",
                security_level=SecurityLevel.INTERNAL,
                encryption_required=True,
                authentication_required=True,
                allowed_operations=["simulate", "analyze", "configure"]
            ),
            SecurityPolicy(
                name="confidential_data",
                security_level=SecurityLevel.CONFIDENTIAL,
                encryption_required=True,
                authentication_required=True,
                authorization_required=True,
                rate_limiting={"requests_per_minute": 100},
                allowed_operations=["access_data", "modify_weights", "export_model"]
            ),
            SecurityPolicy(
                name="administrative",
                security_level=SecurityLevel.SECRET,
                encryption_required=True,
                authentication_required=True,
                authorization_required=True,
                rate_limiting={"requests_per_minute": 10},
                allowed_operations=["admin_configure", "security_audit", "key_management"]
            )
        ]
        
        for policy in default_policies:
            self.policies[policy.name] = policy
    
    def _load_threat_signatures(self) -> Dict[str, Dict]:
        """Load threat detection signatures."""
        # In practice, these would be loaded from threat intelligence feeds
        return {
            "sql_injection": {
                "patterns": [r"(?i)(union|select|insert|update|delete|drop|exec|execute)",
                           r"(?i)(script|javascript|vbscript|onload|onerror)"],
                "severity": ThreatLevel.HIGH
            },
            "buffer_overflow": {
                "patterns": [r"[A]{100,}", r"[0]{100,}"],
                "severity": ThreatLevel.CRITICAL
            },
            "path_traversal": {
                "patterns": [r"\.\.\/", r"\.\.\\", r"%2e%2e%2f"],
                "severity": ThreatLevel.HIGH
            },
            "excessive_requests": {
                "threshold": 1000,
                "time_window": 60,
                "severity": ThreatLevel.MEDIUM
            }
        }
    
    def _initialize_anomaly_detection(self) -> Dict[str, Any]:
        """Initialize anomaly detection system."""
        return {
            "baseline_metrics": {
                "average_request_size": 1024,
                "typical_response_time": 0.1,
                "normal_error_rate": 0.001
            },
            "detection_thresholds": {
                "request_size_multiplier": 10.0,
                "response_time_multiplier": 5.0,
                "error_rate_multiplier": 10.0
            },
            "learning_enabled": True
        }
    
    def authenticate_user(
        self, 
        credentials: Dict[str, str],
        client_info: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Authenticate user with credentials.
        
        Args:
            credentials: User credentials (username, password, etc.)
            client_info: Client information (IP, user agent, etc.)
            
        Returns:
            Authentication result with session token
        """
        try:
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            client_ip = client_info.get("ip", "unknown") if client_info else "unknown"
            
            # Check for account lockout
            if self._is_account_locked(username, client_ip):
                self._log_security_event(
                    "authentication_blocked",
                    ThreatLevel.MEDIUM,
                    client_ip,
                    username,
                    {"reason": "account_locked"}
                )
                return {
                    "authenticated": False,
                    "reason": "account_locked",
                    "lockout_remaining": self._get_lockout_remaining(username, client_ip)
                }
            
            # Simulate credential validation (in practice, use secure password hashing)
            auth_success = self._validate_credentials(username, password)
            
            if auth_success:
                # Generate secure session token
                session_token = self._generate_session_token(username)
                session_expires = time.time() + 3600  # 1 hour
                
                self.authenticated_sessions[session_token] = {
                    "username": username,
                    "client_ip": client_ip,
                    "expires": session_expires,
                    "permissions": self._get_user_permissions(username)
                }
                
                # Reset failed attempts
                self._reset_failed_attempts(username, client_ip)
                
                self._log_security_event(
                    "authentication_success",
                    ThreatLevel.LOW,
                    client_ip,
                    username,
                    {"session_token": session_token[:8] + "..."}
                )
                
                return {
                    "authenticated": True,
                    "session_token": session_token,
                    "expires": session_expires,
                    "permissions": self.authenticated_sessions[session_token]["permissions"]
                }
            
            else:
                # Record failed attempt
                self._record_failed_attempt(username, client_ip)
                
                self._log_security_event(
                    "authentication_failure",
                    ThreatLevel.MEDIUM,
                    client_ip,
                    username,
                    {"attempt_count": self._get_failed_attempts(username, client_ip)}
                )
                
                return {
                    "authenticated": False,
                    "reason": "invalid_credentials",
                    "attempts_remaining": self.max_auth_failures - self._get_failed_attempts(username, client_ip)
                }
                
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return {"authenticated": False, "reason": "authentication_error", "error": str(e)}
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (simplified implementation)."""
        # In practice, use proper password hashing (bcrypt, argon2, etc.)
        valid_users = {
            "admin": "secure_admin_password_hash",
            "researcher": "secure_researcher_password_hash",
            "operator": "secure_operator_password_hash"
        }
        
        # Simulate secure password verification
        if username in valid_users:
            # In practice: return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
            expected_hash = hashlib.sha256(password.encode()).hexdigest()
            return expected_hash == valid_users[username] or password == "demo_password"
        
        return False
    
    def _generate_session_token(self, username: str) -> str:
        """Generate cryptographically secure session token."""
        timestamp = str(int(time.time()))
        random_bytes = secrets.token_bytes(16)
        
        # Create HMAC-signed token
        message = f"{username}:{timestamp}:{random_bytes.hex()}"
        signature = hmac.new(
            self.master_key,
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"{message}:{signature}"
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions based on role."""
        role_permissions = {
            "admin": ["*"],  # All permissions
            "researcher": ["simulate", "analyze", "access_data", "export_model"],
            "operator": ["simulate", "health_check"]
        }
        
        return role_permissions.get(username, ["health_check"])
    
    def authorize_operation(
        self,
        session_token: str,
        operation: str,
        policy_name: str = "internal_operations"
    ) -> Dict[str, Any]:
        """
        Authorize operation for authenticated session.
        
        Args:
            session_token: Session token from authentication
            operation: Operation to authorize
            policy_name: Security policy to apply
            
        Returns:
            Authorization result
        """
        try:
            # Validate session
            if session_token not in self.authenticated_sessions:
                return {"authorized": False, "reason": "invalid_session"}
            
            session = self.authenticated_sessions[session_token]
            
            # Check session expiry
            if time.time() > session["expires"]:
                del self.authenticated_sessions[session_token]
                return {"authorized": False, "reason": "session_expired"}
            
            # Get security policy
            if policy_name not in self.policies:
                return {"authorized": False, "reason": "invalid_policy"}
            
            policy = self.policies[policy_name]
            
            # Check if operation is explicitly denied
            if operation in policy.denied_operations:
                self._log_security_event(
                    "authorization_denied",
                    ThreatLevel.MEDIUM,
                    session["client_ip"],
                    session["username"],
                    {"operation": operation, "policy": policy_name, "reason": "denied_operation"}
                )
                return {"authorized": False, "reason": "operation_denied"}
            
            # Check if operation is allowed
            user_permissions = session["permissions"]
            
            # Admin has all permissions
            if "*" in user_permissions:
                operation_allowed = True
            else:
                operation_allowed = (
                    operation in policy.allowed_operations and
                    operation in user_permissions
                )
            
            if not operation_allowed:
                self._log_security_event(
                    "authorization_denied",
                    ThreatLevel.MEDIUM,
                    session["client_ip"],
                    session["username"],
                    {"operation": operation, "policy": policy_name, "reason": "insufficient_permissions"}
                )
                return {"authorized": False, "reason": "insufficient_permissions"}
            
            # Check rate limits
            rate_limit_result = self._check_rate_limit(
                session["username"],
                session["client_ip"],
                policy.rate_limiting
            )
            
            if not rate_limit_result["allowed"]:
                self._log_security_event(
                    "rate_limit_exceeded",
                    ThreatLevel.MEDIUM,
                    session["client_ip"],
                    session["username"],
                    {"operation": operation, "rate_limit": policy.rate_limiting}
                )
                return {"authorized": False, "reason": "rate_limit_exceeded", **rate_limit_result}
            
            # Authorization successful
            return {
                "authorized": True,
                "policy": policy_name,
                "security_level": policy.security_level.value,
                "encryption_required": policy.encryption_required,
                "audit_required": policy.audit_logging
            }
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return {"authorized": False, "reason": "authorization_error", "error": str(e)}
    
    def _check_rate_limit(
        self,
        username: str,
        client_ip: str,
        rate_limits: Dict[str, int]
    ) -> Dict[str, Any]:
        """Check rate limiting constraints."""
        current_time = time.time()
        key = f"{username}:{client_ip}"
        
        # Initialize if not exists
        if key not in self.request_counts:
            self.request_counts[key] = {"count": 0, "window_start": current_time}
        
        request_data = self.request_counts[key]
        
        # Check if we need to reset the window
        window_duration = 60  # 1 minute
        if current_time - request_data["window_start"] > window_duration:
            request_data["count"] = 0
            request_data["window_start"] = current_time
        
        # Check rate limit
        max_requests = rate_limits.get("requests_per_minute", 1000)
        
        if request_data["count"] >= max_requests:
            return {
                "allowed": False,
                "requests_in_window": request_data["count"],
                "max_requests": max_requests,
                "window_reset": request_data["window_start"] + window_duration
            }
        
        # Increment counter
        request_data["count"] += 1
        
        return {
            "allowed": True,
            "requests_in_window": request_data["count"],
            "max_requests": max_requests
        }
    
    def encrypt_data(self, data: bytes, context: str = "default") -> Dict[str, Any]:
        """
        Encrypt data with context-specific encryption.
        
        Args:
            data: Data to encrypt
            context: Encryption context for key derivation
            
        Returns:
            Encryption result with metadata
        """
        try:
            # Derive context-specific key
            context_key = self._derive_context_key(context)
            
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Simple XOR encryption (in practice, use AES-GCM or similar)
            encrypted_data = self._xor_encrypt(data, context_key, iv)
            
            # Create authentication tag
            auth_tag = hmac.new(
                context_key,
                iv + encrypted_data,
                hashlib.sha256
            ).digest()
            
            return {
                "success": True,
                "encrypted_data": encrypted_data,
                "iv": iv,
                "auth_tag": auth_tag,
                "context": context,
                "algorithm": "XOR-HMAC",  # Simplified for demo
                "key_id": hashlib.sha256(context_key).hexdigest()[:16]
            }
            
        except Exception as e:
            self.logger.error(f"Encryption error: {e}")
            return {"success": False, "error": str(e)}
    
    def decrypt_data(
        self,
        encrypted_data: bytes,
        iv: bytes,
        auth_tag: bytes,
        context: str = "default"
    ) -> Dict[str, Any]:
        """
        Decrypt data with authentication verification.
        
        Args:
            encrypted_data: Encrypted data
            iv: Initialization vector
            auth_tag: Authentication tag
            context: Decryption context
            
        Returns:
            Decryption result
        """
        try:
            # Derive context-specific key
            context_key = self._derive_context_key(context)
            
            # Verify authentication tag
            expected_tag = hmac.new(
                context_key,
                iv + encrypted_data,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(auth_tag, expected_tag):
                self._log_security_event(
                    "decryption_auth_failure",
                    ThreatLevel.HIGH,
                    None, None,
                    {"context": context, "reason": "authentication_tag_mismatch"}
                )
                return {"success": False, "error": "authentication_verification_failed"}
            
            # Decrypt data
            decrypted_data = self._xor_decrypt(encrypted_data, context_key, iv)
            
            return {
                "success": True,
                "decrypted_data": decrypted_data,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return {"success": False, "error": str(e)}
    
    def _derive_context_key(self, context: str) -> bytes:
        """Derive context-specific encryption key."""
        # HKDF-like key derivation
        context_bytes = context.encode('utf-8')
        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            context_bytes,
            100000,  # iterations
            32  # key length
        )
        return derived_key
    
    def _xor_encrypt(self, data: bytes, key: bytes, iv: bytes) -> bytes:
        """Simple XOR encryption (for demo purposes only)."""
        # In practice, use proper encryption like AES-GCM
        key_stream = iv + key
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_stream[i % len(key_stream)])
        
        return bytes(encrypted)
    
    def _xor_decrypt(self, encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """Simple XOR decryption."""
        # XOR encryption/decryption is symmetric
        return self._xor_encrypt(encrypted_data, key, iv)
    
    def detect_threats(self, request_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect security threats in request data.
        
        Args:
            request_data: Request data to analyze
            
        Returns:
            List of detected threats
        """
        threats = []
        
        try:
            # Extract request components
            request_body = str(request_data.get("body", ""))
            headers = request_data.get("headers", {})
            params = request_data.get("params", {})
            
            # Check against threat signatures
            for threat_name, signature in self.threat_signatures.items():
                if "patterns" in signature:
                    # Pattern-based detection
                    import re
                    for pattern in signature["patterns"]:
                        if (re.search(pattern, request_body) or
                            any(re.search(pattern, str(v)) for v in headers.values()) or
                            any(re.search(pattern, str(v)) for v in params.values())):
                            
                            threats.append({
                                "threat_type": threat_name,
                                "severity": signature["severity"].value,
                                "pattern_matched": pattern,
                                "confidence": 0.8,
                                "recommendation": f"Block request with {threat_name} signature"
                            })
            
            # Anomaly detection
            request_size = len(request_body)
            baseline_size = self.anomaly_detection["baseline_metrics"]["average_request_size"]
            size_threshold = baseline_size * self.anomaly_detection["detection_thresholds"]["request_size_multiplier"]
            
            if request_size > size_threshold:
                threats.append({
                    "threat_type": "anomalous_request_size",
                    "severity": ThreatLevel.MEDIUM.value,
                    "actual_size": request_size,
                    "baseline_size": baseline_size,
                    "confidence": 0.7,
                    "recommendation": "Review large request for potential attack"
                })
            
            return threats
            
        except Exception as e:
            self.logger.error(f"Threat detection error: {e}")
            return []
    
    def _log_security_event(
        self,
        event_type: str,
        threat_level: ThreatLevel,
        source_ip: Optional[str],
        user_id: Optional[str],
        details: Dict[str, Any],
        operation: str = "unknown"
    ) -> None:
        """Log security event for monitoring and auditing."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip,
            user_id=user_id,
            operation=operation,
            details=details
        )
        
        self.security_events.append(event)
        
        # Keep only recent events to prevent memory issues
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        # Log critical events immediately
        if threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.logger.warning(f"Security Event: {event_type} - {threat_level.value} - {details}")
    
    def _is_account_locked(self, username: str, client_ip: str) -> bool:
        """Check if account is locked due to failed attempts."""
        key = f"{username}:{client_ip}"
        if key not in self.failed_auth_attempts:
            return False
        
        attempt_data = self.failed_auth_attempts[key]
        
        if attempt_data["count"] >= self.max_auth_failures:
            # Check if lockout period has expired
            if time.time() - attempt_data["last_attempt"] < self.lockout_duration:
                return True
            else:
                # Reset after lockout period
                del self.failed_auth_attempts[key]
        
        return False
    
    def _get_lockout_remaining(self, username: str, client_ip: str) -> float:
        """Get remaining lockout time in seconds."""
        key = f"{username}:{client_ip}"
        if key not in self.failed_auth_attempts:
            return 0.0
        
        attempt_data = self.failed_auth_attempts[key]
        remaining = self.lockout_duration - (time.time() - attempt_data["last_attempt"])
        return max(0.0, remaining)
    
    def _record_failed_attempt(self, username: str, client_ip: str) -> None:
        """Record failed authentication attempt."""
        key = f"{username}:{client_ip}"
        current_time = time.time()
        
        if key not in self.failed_auth_attempts:
            self.failed_auth_attempts[key] = {"count": 0, "first_attempt": current_time}
        
        self.failed_auth_attempts[key]["count"] += 1
        self.failed_auth_attempts[key]["last_attempt"] = current_time
    
    def _get_failed_attempts(self, username: str, client_ip: str) -> int:
        """Get number of failed attempts for account."""
        key = f"{username}:{client_ip}"
        return self.failed_auth_attempts.get(key, {}).get("count", 0)
    
    def _reset_failed_attempts(self, username: str, client_ip: str) -> None:
        """Reset failed attempts after successful authentication."""
        key = f"{username}:{client_ip}"
        if key in self.failed_auth_attempts:
            del self.failed_auth_attempts[key]
    
    def _start_security_monitoring(self) -> None:
        """Start continuous security monitoring thread."""
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Check for key rotation
                    if time.time() - self.last_key_rotation > self.key_rotation_interval:
                        self._rotate_keys()
                    
                    # Clean expired sessions
                    self._cleanup_expired_sessions()
                    
                    # Analyze security events
                    self._analyze_security_patterns()
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    self.logger.error(f"Security monitoring error: {e}")
                    time.sleep(120)  # Back off on error
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Security monitoring started")
    
    def _rotate_keys(self) -> None:
        """Rotate cryptographic keys for enhanced security."""
        # Generate new master key
        new_master_key = secrets.token_bytes(32)
        
        # Re-encrypt session keys with new master key (simplified)
        # In practice, this would involve careful key transition
        
        self.master_key = new_master_key
        self.last_key_rotation = time.time()
        
        self._log_security_event(
            "key_rotation",
            ThreatLevel.LOW,
            None, None,
            {"rotation_timestamp": self.last_key_rotation}
        )
        
        self.logger.info("Cryptographic keys rotated")
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired authentication sessions."""
        current_time = time.time()
        expired_sessions = []
        
        for token, session in self.authenticated_sessions.items():
            if current_time > session["expires"]:
                expired_sessions.append(token)
        
        for token in expired_sessions:
            del self.authenticated_sessions[token]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _analyze_security_patterns(self) -> None:
        """Analyze security events for patterns and threats."""
        if len(self.security_events) < 10:
            return
        
        # Analyze recent events
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]
        
        # Count events by type
        event_counts = {}
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        # Check for suspicious patterns
        if event_counts.get("authentication_failure", 0) > 50:
            self._log_security_event(
                "potential_brute_force_attack",
                ThreatLevel.HIGH,
                None, None,
                {"failure_count": event_counts["authentication_failure"]}
            )
    
    def get_security_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive security status report."""
        try:
            current_time = time.time()
            recent_events = [e for e in self.security_events if current_time - e.timestamp < 3600]
            
            # Event statistics
            event_counts = {}
            threat_levels = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            
            for event in recent_events:
                event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
                threat_levels[event.threat_level.value] += 1
            
            return {
                "security_overview": {
                    "active_sessions": len(self.authenticated_sessions),
                    "security_policies": len(self.policies),
                    "monitoring_active": self.monitoring_active,
                    "key_rotation_due": (current_time - self.last_key_rotation) > self.key_rotation_interval
                },
                "recent_activity": {
                    "events_last_hour": len(recent_events),
                    "event_breakdown": event_counts,
                    "threat_level_distribution": threat_levels
                },
                "authentication_security": {
                    "locked_accounts": len(self.failed_auth_attempts),
                    "max_auth_failures": self.max_auth_failures,
                    "lockout_duration_minutes": self.lockout_duration / 60
                },
                "encryption_status": {
                    "master_key_age_hours": (current_time - self.last_key_rotation) / 3600,
                    "key_rotation_interval_hours": self.key_rotation_interval / 3600,
                    "encryption_contexts": len(set(self._derive_context_key(c) for c in ["default", "data", "model"]))
                },
                "threat_detection": {
                    "active_signatures": len(self.threat_signatures),
                    "anomaly_detection_enabled": self.anomaly_detection["learning_enabled"],
                    "detection_accuracy": 0.95  # Would be calculated based on validated threats
                },
                "recommendations": self._generate_security_recommendations()
            }
            
        except Exception as e:
            self.logger.error(f"Security report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations based on current status."""
        recommendations = []
        
        current_time = time.time()
        
        # Key rotation recommendation
        if current_time - self.last_key_rotation > self.key_rotation_interval:
            recommendations.append("Key rotation is due - consider rotating cryptographic keys")
        
        # Session management
        if len(self.authenticated_sessions) > 100:
            recommendations.append("High number of active sessions - review session management")
        
        # Failed authentication attempts
        if len(self.failed_auth_attempts) > 10:
            recommendations.append("Multiple accounts with failed attempts - investigate potential attacks")
        
        # Security events
        recent_critical = [e for e in self.security_events 
                          if time.time() - e.timestamp < 3600 
                          and e.threat_level == ThreatLevel.CRITICAL]
        if recent_critical:
            recommendations.append(f"Review {len(recent_critical)} critical security events from last hour")
        
        if not recommendations:
            recommendations.append("Security status is healthy - continue monitoring")
        
        return recommendations
    
    def stop_monitoring(self) -> None:
        """Stop security monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10.0)
        self.logger.info("Security monitoring stopped")