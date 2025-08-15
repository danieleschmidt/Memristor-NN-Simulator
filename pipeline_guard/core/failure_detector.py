"""
Failure Detector: AI-powered detection and classification of pipeline failures
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class FailurePattern:
    """Represents a detected failure pattern"""
    pattern_id: str
    name: str
    regex_patterns: List[str]
    confidence_threshold: float
    remediation_strategy: str
    frequency: int = 0
    last_seen: Optional[datetime] = None


@dataclass
class FailureDetection:
    """Result of failure detection analysis"""
    detected: bool
    pattern_id: Optional[str]
    confidence: float
    failure_type: str
    error_message: str
    suggested_remediation: str
    context: Dict[str, Any]


class FailureDetector:
    """
    AI-powered failure detection system for CI/CD pipelines
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.failure_patterns = self._initialize_patterns()
        self.pattern_frequencies = defaultdict(int)
        self.learning_enabled = True
        
    def _initialize_patterns(self) -> Dict[str, FailurePattern]:
        """Initialize known failure patterns"""
        patterns = {
            "dependency_failure": FailurePattern(
                pattern_id="dependency_failure",
                name="Dependency Installation Failure",
                regex_patterns=[
                    r"npm install.*failed",
                    r"pip install.*error",
                    r"bundle install.*failed",
                    r"package.*not found",
                    r"dependency.*could not be resolved"
                ],
                confidence_threshold=0.8,
                remediation_strategy="retry_with_cache_clear"
            ),
            
            "test_failure": FailurePattern(
                pattern_id="test_failure",
                name="Test Suite Failure",
                regex_patterns=[
                    r"\d+ failed.*\d+ passed",
                    r"FAILED.*test_",
                    r"AssertionError:",
                    r"Test.*failed",
                    r"Error in test"
                ],
                confidence_threshold=0.9,
                remediation_strategy="isolate_and_rerun_tests"
            ),
            
            "build_failure": FailurePattern(
                pattern_id="build_failure", 
                name="Build Compilation Failure",
                regex_patterns=[
                    r"compilation failed",
                    r"build.*error",
                    r"error.*\.java:\d+",
                    r"\.c:\d+:\d+: error",
                    r"syntax error"
                ],
                confidence_threshold=0.85,
                remediation_strategy="check_syntax_and_dependencies"
            ),
            
            "timeout_failure": FailurePattern(
                pattern_id="timeout_failure",
                name="Pipeline Timeout",
                regex_patterns=[
                    r"timeout",
                    r"job.*cancelled.*timeout",
                    r"exceeded.*time limit",
                    r"process.*killed.*timeout"
                ],
                confidence_threshold=0.95,
                remediation_strategy="increase_timeout_or_optimize"
            ),
            
            "resource_failure": FailurePattern(
                pattern_id="resource_failure",
                name="Resource Exhaustion",
                regex_patterns=[
                    r"out of memory",
                    r"disk.*full",
                    r"no space left",
                    r"memory.*exceeded",
                    r"killed.*oom"
                ],
                confidence_threshold=0.9,
                remediation_strategy="increase_resources_or_optimize"
            ),
            
            "network_failure": FailurePattern(
                pattern_id="network_failure",
                name="Network/Connectivity Issues",
                regex_patterns=[
                    r"connection.*refused",
                    r"network.*unreachable",
                    r"timeout.*connecting",
                    r"dns.*resolution.*failed",
                    r"ssl.*certificate.*error"
                ],
                confidence_threshold=0.85,
                remediation_strategy="retry_with_backoff"
            ),
            
            "permission_failure": FailurePattern(
                pattern_id="permission_failure",
                name="Permission/Authentication Issues", 
                regex_patterns=[
                    r"permission denied",
                    r"access.*denied",
                    r"authentication.*failed",
                    r"unauthorized",
                    r"forbidden.*403"
                ],
                confidence_threshold=0.9,
                remediation_strategy="check_credentials_and_permissions"
            )
        }
        
        return patterns
        
    def detect_failure(self, logs: str, metadata: Dict[str, Any] = None) -> FailureDetection:
        """
        Detect and classify pipeline failures from logs
        """
        if not logs:
            return FailureDetection(
                detected=False,
                pattern_id=None,
                confidence=0.0,
                failure_type="unknown",
                error_message="No logs provided",
                suggested_remediation="Check pipeline configuration",
                context={}
            )
            
        # Analyze logs against known patterns
        best_match = self._analyze_patterns(logs)
        
        if best_match:
            pattern_id, confidence, matched_text = best_match
            pattern = self.failure_patterns[pattern_id]
            
            # Update pattern frequency
            self.pattern_frequencies[pattern_id] += 1
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
            
            return FailureDetection(
                detected=True,
                pattern_id=pattern_id,
                confidence=confidence,
                failure_type=pattern.name,
                error_message=matched_text,
                suggested_remediation=pattern.remediation_strategy,
                context={
                    "pattern_frequency": pattern.frequency,
                    "last_seen": pattern.last_seen.isoformat(),
                    "metadata": metadata or {}
                }
            )
        
        # If no pattern matches, use generic detection
        return self._generic_failure_detection(logs, metadata)
        
    def _analyze_patterns(self, logs: str) -> Optional[Tuple[str, float, str]]:
        """Analyze logs against known failure patterns"""
        best_match = None
        best_confidence = 0.0
        
        logs_lower = logs.lower()
        
        for pattern_id, pattern in self.failure_patterns.items():
            matches = 0
            matched_texts = []
            
            for regex_pattern in pattern.regex_patterns:
                matches_found = re.finditer(regex_pattern, logs_lower, re.IGNORECASE)
                for match in matches_found:
                    matches += 1
                    matched_texts.append(match.group())
                    
            if matches > 0:
                # Calculate confidence based on matches and pattern reliability
                base_confidence = min(matches * 0.3, 1.0)
                frequency_boost = min(pattern.frequency * 0.05, 0.2)
                confidence = min(base_confidence + frequency_boost, 1.0)
                
                if confidence >= pattern.confidence_threshold and confidence > best_confidence:
                    best_confidence = confidence
                    best_match = (pattern_id, confidence, "; ".join(matched_texts[:3]))
                    
        return best_match
        
    def _generic_failure_detection(self, logs: str, metadata: Dict[str, Any] = None) -> FailureDetection:
        """Generic failure detection for unknown patterns"""
        error_indicators = [
            "error", "failed", "exception", "panic", "abort", 
            "fatal", "critical", "emergency"
        ]
        
        logs_lower = logs.lower()
        error_count = sum(logs_lower.count(indicator) for indicator in error_indicators)
        
        if error_count > 0:
            confidence = min(error_count * 0.2, 0.7)
            
            # Extract potential error messages
            error_lines = []
            for line in logs.split('\n'):
                if any(indicator in line.lower() for indicator in error_indicators):
                    error_lines.append(line.strip())
                    
            error_message = "; ".join(error_lines[:3]) if error_lines else "Generic error detected"
            
            return FailureDetection(
                detected=True,
                pattern_id="generic_error",
                confidence=confidence,
                failure_type="Generic Error",
                error_message=error_message,
                suggested_remediation="manual_investigation",
                context={"error_count": error_count, "metadata": metadata or {}}
            )
            
        return FailureDetection(
            detected=False,
            pattern_id=None,
            confidence=0.0,
            failure_type="unknown",
            error_message="No clear failure indicators found",
            suggested_remediation="Check pipeline status and logs",
            context={"metadata": metadata or {}}
        )
        
    def learn_pattern(self, logs: str, failure_type: str, remediation: str) -> bool:
        """Learn new failure pattern from provided example"""
        if not self.learning_enabled:
            return False
            
        # Extract potential regex patterns from logs
        error_lines = [line.strip() for line in logs.split('\n') 
                      if any(indicator in line.lower() 
                            for indicator in ["error", "failed", "exception"])]
        
        if not error_lines:
            return False
            
        # Create simplified regex patterns
        patterns = []
        for line in error_lines[:3]:
            # Create a simplified pattern from the error line
            pattern = re.escape(line)
            pattern = re.sub(r'\\d+', r'\\d+', pattern)  # Preserve number patterns
            pattern = re.sub(r'\\\w+', r'\\w+', pattern)  # Generalize words
            patterns.append(pattern)
            
        pattern_id = f"learned_{failure_type.lower().replace(' ', '_')}"
        
        self.failure_patterns[pattern_id] = FailurePattern(
            pattern_id=pattern_id,
            name=f"Learned: {failure_type}",
            regex_patterns=patterns,
            confidence_threshold=0.7,
            remediation_strategy=remediation,
            frequency=1,
            last_seen=datetime.now()
        )
        
        self.logger.info(f"Learned new pattern: {pattern_id}")
        return True
        
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected patterns"""
        total_detections = sum(self.pattern_frequencies.values())
        
        pattern_stats = {}
        for pattern_id, pattern in self.failure_patterns.items():
            frequency = self.pattern_frequencies[pattern_id]
            pattern_stats[pattern_id] = {
                "name": pattern.name,
                "frequency": frequency,
                "percentage": (frequency / total_detections * 100) if total_detections > 0 else 0,
                "last_seen": pattern.last_seen.isoformat() if pattern.last_seen else None,
                "remediation": pattern.remediation_strategy
            }
            
        return {
            "total_detections": total_detections,
            "patterns": pattern_stats,
            "top_patterns": sorted(
                pattern_stats.items(),
                key=lambda x: x[1]["frequency"],
                reverse=True
            )[:5]
        }
        
    def export_patterns(self) -> str:
        """Export learned patterns as JSON"""
        exportable_patterns = {}
        for pattern_id, pattern in self.failure_patterns.items():
            exportable_patterns[pattern_id] = {
                "name": pattern.name,
                "regex_patterns": pattern.regex_patterns,
                "confidence_threshold": pattern.confidence_threshold,
                "remediation_strategy": pattern.remediation_strategy,
                "frequency": pattern.frequency
            }
            
        return json.dumps(exportable_patterns, indent=2)
        
    def import_patterns(self, patterns_json: str) -> bool:
        """Import patterns from JSON"""
        try:
            imported_patterns = json.loads(patterns_json)
            
            for pattern_id, pattern_data in imported_patterns.items():
                self.failure_patterns[pattern_id] = FailurePattern(
                    pattern_id=pattern_id,
                    name=pattern_data["name"],
                    regex_patterns=pattern_data["regex_patterns"],
                    confidence_threshold=pattern_data["confidence_threshold"],
                    remediation_strategy=pattern_data["remediation_strategy"],
                    frequency=pattern_data.get("frequency", 0)
                )
                
            self.logger.info(f"Imported {len(imported_patterns)} patterns")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Failed to import patterns: {e}")
            return False