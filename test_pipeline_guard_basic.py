"""
Basic test runner for Pipeline Guard (without external dependencies)
"""

import sys
import time
import json
import threading
from datetime import datetime, timedelta

# Import Pipeline Guard components
from pipeline_guard.core.pipeline_monitor import PipelineMonitor, PipelineStatus
from pipeline_guard.core.failure_detector import FailureDetector
from pipeline_guard.core.healing_engine import HealingEngine
from pipeline_guard.utils.security import SecurityManager, SecurityConfig
from pipeline_guard.utils.error_handling import CircuitBreaker, RetryStrategy
from pipeline_guard.utils.validators import InputValidator
from pipeline_guard.scaling.cache_manager import CacheManager, CacheStrategy


def test_pipeline_monitor():
    """Test pipeline monitoring functionality"""
    print("Testing Pipeline Monitor...")
    
    monitor = PipelineMonitor(check_interval=1)
    
    # Test pipeline registration
    monitor.register_pipeline("test-pipeline-1", "Test Pipeline")
    assert "test-pipeline-1" in monitor.pipelines
    assert monitor.pipelines["test-pipeline-1"].name == "Test Pipeline"
    print("✓ Pipeline registration works")
    
    # Test status update
    monitor.update_pipeline_status("test-pipeline-1", "success")
    assert monitor.pipelines["test-pipeline-1"].status == "success"
    print("✓ Pipeline status update works")
    
    # Test health summary
    summary = monitor.get_health_summary()
    assert summary["total_pipelines"] == 1
    print("✓ Health summary generation works")
    
    monitor.stop_monitoring()
    print("✓ Pipeline Monitor tests passed\n")


def test_failure_detector():
    """Test failure detection functionality"""
    print("Testing Failure Detector...")
    
    detector = FailureDetector()
    
    # Test dependency failure detection
    logs = "npm install failed with error ENOTFOUND"
    detection = detector.detect_failure(logs)
    
    assert detection.detected
    assert detection.pattern_id == "dependency_failure"
    print("✓ Dependency failure detection works")
    
    # Test test failure detection
    test_logs = "test_user_login FAILED\nAssertionError: Expected 200, got 500"
    test_detection = detector.detect_failure(test_logs)
    
    assert test_detection.detected
    assert test_detection.pattern_id == "test_failure"
    print("✓ Test failure detection works")
    
    # Test pattern statistics
    stats = detector.get_pattern_statistics()
    assert stats["total_detections"] >= 2
    print("✓ Pattern statistics work")
    
    print("✓ Failure Detector tests passed\n")


def test_healing_engine():
    """Test healing engine functionality"""
    print("Testing Healing Engine...")
    
    healer = HealingEngine()
    
    # Test healing strategy execution
    results = healer.heal_pipeline(
        "test-pipeline",
        "retry_with_cache_clear",
        {"logs": "npm install failed"}
    )
    
    assert len(results) > 0
    print("✓ Healing strategy execution works")
    
    # Test healing statistics
    stats = healer.get_healing_statistics()
    assert "total_actions" in stats
    assert "success_rate" in stats
    print("✓ Healing statistics work")
    
    print("✓ Healing Engine tests passed\n")


def test_security_manager():
    """Test security management functionality"""
    print("Testing Security Manager...")
    
    config = SecurityConfig()
    security_manager = SecurityManager(config)
    
    # Test valid input
    valid_result = security_manager.validate_request(
        {"message": "normal text"},
        "192.168.1.1"
    )
    assert valid_result["valid"]
    print("✓ Valid input validation works")
    
    # Test invalid input
    invalid_result = security_manager.validate_request(
        {"message": "<script>alert('xss')</script>"},
        "192.168.1.1"
    )
    assert not invalid_result["valid"]
    print("✓ Invalid input detection works")
    
    # Test security summary
    summary = security_manager.get_security_summary()
    assert "total_events_24h" in summary
    print("✓ Security summary generation works")
    
    print("✓ Security Manager tests passed\n")


def test_circuit_breaker():
    """Test circuit breaker functionality"""
    print("Testing Circuit Breaker...")
    
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1)
    
    def failing_function():
        raise Exception("Test failure")
    
    # Test normal failures
    failure_count = 0
    for _ in range(3):
        try:
            circuit_breaker.call(failing_function)
        except Exception:
            failure_count += 1
    
    assert failure_count == 3
    print("✓ Normal failure handling works")
    
    # Test circuit breaker state
    state = circuit_breaker.get_state()
    assert state["state"] == "open"
    print("✓ Circuit breaker opens after threshold")
    
    # Test circuit breaker blocking
    try:
        circuit_breaker.call(failing_function)
        assert False, "Should have been blocked"
    except Exception as e:
        assert "Circuit breaker is OPEN" in str(e)
    print("✓ Circuit breaker blocks calls when open")
    
    print("✓ Circuit Breaker tests passed\n")


def test_retry_strategy():
    """Test retry strategy functionality"""
    print("Testing Retry Strategy...")
    
    retry_strategy = RetryStrategy(max_attempts=3, initial_delay=0.01)
    
    call_count = 0
    def unstable_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    result = retry_strategy.execute_with_retry(unstable_function)
    
    assert result == "success"
    assert call_count == 3
    print("✓ Retry strategy works")
    
    print("✓ Retry Strategy tests passed\n")


def test_input_validator():
    """Test input validation functionality"""
    print("Testing Input Validator...")
    
    validator = InputValidator()
    
    # Test valid string
    result = validator.validate_string("Hello World", max_length=20)
    assert result.valid
    print("✓ Valid string validation works")
    
    # Test invalid string (too long)
    result = validator.validate_string("x" * 1001, max_length=1000)
    assert not result.valid
    print("✓ String length validation works")
    
    # Test email validation
    email_result = validator.validate_email("test@example.com")
    assert email_result.valid
    print("✓ Valid email validation works")
    
    invalid_email_result = validator.validate_email("invalid-email")
    assert not invalid_email_result.valid
    print("✓ Invalid email detection works")
    
    print("✓ Input Validator tests passed\n")


def test_cache_manager():
    """Test cache management functionality"""
    print("Testing Cache Manager...")
    
    cache_manager = CacheManager(strategy=CacheStrategy.LRU, max_size=10)
    
    # Test basic operations
    success = cache_manager.set("key1", "value1")
    assert success
    print("✓ Cache set operation works")
    
    value = cache_manager.get("key1")
    assert value == "value1"
    print("✓ Cache get operation works")
    
    # Test expiration
    cache_manager.set("expire_key", "expire_value", ttl=0.1)
    value = cache_manager.get("expire_key")
    assert value == "expire_value"
    
    time.sleep(0.2)
    value = cache_manager.get("expire_key")
    assert value is None
    print("✓ Cache expiration works")
    
    # Test cache info
    info = cache_manager.get_cache_info()
    assert "size" in info
    assert "statistics" in info
    print("✓ Cache info generation works")
    
    cache_manager.stop()
    print("✓ Cache Manager tests passed\n")


def test_integration():
    """Test integration between components"""
    print("Testing Integration...")
    
    # Create system components
    monitor = PipelineMonitor(check_interval=1)
    detector = FailureDetector()
    healer = HealingEngine()
    
    # Register a pipeline
    monitor.register_pipeline("integration-test", "Integration Test Pipeline")
    print("✓ Pipeline registered")
    
    # Simulate failure
    failure_logs = "npm install failed with error ENOTFOUND"
    detection = detector.detect_failure(failure_logs, {
        "pipeline_id": "integration-test"
    })
    
    assert detection.detected
    print("✓ Failure detected")
    
    # Attempt healing
    healing_results = healer.heal_pipeline(
        "integration-test",
        detection.suggested_remediation,
        {"logs": failure_logs}
    )
    
    assert len(healing_results) > 0
    print("✓ Healing attempted")
    
    # Update pipeline status
    monitor.update_pipeline_status("integration-test", "success")
    
    # Verify final state
    health_summary = monitor.get_health_summary()
    assert health_summary["total_pipelines"] == 1
    print("✓ Pipeline status updated")
    
    monitor.stop_monitoring()
    print("✓ Integration tests passed\n")


def run_performance_tests():
    """Run basic performance tests"""
    print("Running Performance Tests...")
    
    # Test concurrent pipeline registration
    monitor = PipelineMonitor(check_interval=1)
    
    def register_pipelines(start_id, count):
        for i in range(count):
            pipeline_id = f"perf-{start_id}-{i}"
            monitor.register_pipeline(pipeline_id, f"Performance Test {pipeline_id}")
    
    threads = []
    start_time = time.time()
    
    for i in range(5):
        thread = threading.Thread(target=register_pipelines, args=(i, 20))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    duration = end_time - start_time
    
    health_summary = monitor.get_health_summary()
    assert health_summary["total_pipelines"] == 100
    
    print(f"✓ Registered 100 pipelines in {duration:.2f} seconds")
    print(f"✓ Average: {(duration/100)*1000:.2f} ms per pipeline")
    
    monitor.stop_monitoring()
    print("✓ Performance tests passed\n")


def main():
    """Run all tests"""
    print("=" * 60)
    print("PIPELINE GUARD - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    try:
        # Core functionality tests
        test_pipeline_monitor()
        test_failure_detector()
        test_healing_engine()
        
        # Security and reliability tests
        test_security_manager()
        test_circuit_breaker()
        test_retry_strategy()
        
        # Utility tests
        test_input_validator()
        test_cache_manager()
        
        # Integration tests
        test_integration()
        
        # Performance tests
        run_performance_tests()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print(f"Total execution time: {total_duration:.2f} seconds")
        print("✅ Pipeline Guard system is functioning correctly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)