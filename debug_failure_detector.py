"""
Debug failure detector to understand pattern detection
"""

import sys
sys.path.insert(0, '/root/repo')

from pipeline_guard.core.failure_detector import FailureDetector

def debug_detection():
    detector = FailureDetector()
    
    # Test dependency failure detection
    logs = "npm install failed with error ENOTFOUND"
    detection = detector.detect_failure(logs)
    
    print(f"Logs: {logs}")
    print(f"Detected: {detection.detected}")
    print(f"Pattern ID: {detection.pattern_id}")
    print(f"Confidence: {detection.confidence}")
    print(f"Failure Type: {detection.failure_type}")
    print(f"Error Message: {detection.error_message}")
    print(f"Suggested Remediation: {detection.suggested_remediation}")
    
    # Debug pattern matching
    for pattern_id, pattern in detector.failure_patterns.items():
        matches = 0
        for regex_pattern in pattern.regex_patterns:
            import re
            if re.search(regex_pattern, logs.lower(), re.IGNORECASE):
                matches += 1
                print(f"Pattern {pattern_id} matched: {regex_pattern}")
        
        if matches > 0:
            print(f"Pattern {pattern_id} total matches: {matches}")

if __name__ == "__main__":
    debug_detection()