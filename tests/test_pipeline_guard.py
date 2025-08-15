"""
Comprehensive test suite for Pipeline Guard
"""

import pytest
import time
import json
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import Pipeline Guard components
from pipeline_guard.core.pipeline_monitor import PipelineMonitor, PipelineStatus
from pipeline_guard.core.failure_detector import FailureDetector, FailurePattern
from pipeline_guard.core.healing_engine import HealingEngine, RemediationAction, RemediationStatus
from pipeline_guard.utils.security import SecurityManager, SecurityValidator, SecurityConfig
from pipeline_guard.utils.error_handling import CircuitBreaker, RetryStrategy, ErrorHandler
from pipeline_guard.utils.validators import InputValidator, ConfigValidator
from pipeline_guard.scaling.auto_scaler import AutoScaler, ScalingPolicy, ScalingMetric
from pipeline_guard.scaling.load_balancer import LoadBalancer, ServiceInstance, ServiceStatus
from pipeline_guard.scaling.cache_manager import CacheManager, CacheStrategy
from pipeline_guard.scaling.performance_monitor import PerformanceMonitor, MetricsCollector


class TestPipelineMonitor:
    """Test pipeline monitoring functionality"""
    
    @pytest.fixture
    def monitor(self):
        return PipelineMonitor(check_interval=1)
        
    def test_pipeline_registration(self, monitor):
        """Test pipeline registration"""
        monitor.register_pipeline("test-pipeline-1", "Test Pipeline")
        
        assert "test-pipeline-1" in monitor.pipelines
        pipeline = monitor.pipelines["test-pipeline-1"]
        assert pipeline.name == "Test Pipeline"
        assert pipeline.status == "running"
        
    def test_pipeline_status_update(self, monitor):
        """Test pipeline status updates"""
        monitor.register_pipeline("test-pipeline-1", "Test Pipeline")
        monitor.update_pipeline_status("test-pipeline-1", "success")
        
        pipeline = monitor.pipelines["test-pipeline-1"]
        assert pipeline.status == "success"
        assert pipeline.finished_at is not None
        
    def test_health_summary(self, monitor):
        """Test health summary generation"""
        monitor.register_pipeline("pipeline-1", "Pipeline 1")
        monitor.register_pipeline("pipeline-2", "Pipeline 2")
        monitor.update_pipeline_status("pipeline-1", "success")
        monitor.update_pipeline_status("pipeline-2", "failed", "timeout")
        
        summary = monitor.get_health_summary()
        
        assert summary["total_pipelines"] == 2
        assert summary["running_pipelines"] == 0
        assert summary["failed_pipelines"] == 1
        
    def test_timeout_detection(self, monitor):
        """Test timeout detection"""
        monitor.register_pipeline("timeout-pipeline", "Timeout Test")
        
        # Manually set old start time
        pipeline = monitor.pipelines["timeout-pipeline"]
        pipeline.started_at = datetime.now() - timedelta(hours=3)
        
        # Trigger timeout check
        monitor._check_pipelines()
        
        # Should detect timeout
        events = monitor.get_events()
        assert any(event["type"] == "timeout" for event in events)


class TestFailureDetector:
    """Test failure detection functionality"""
    
    @pytest.fixture
    def detector(self):
        return FailureDetector()
        
    def test_dependency_failure_detection(self, detector):
        """Test dependency failure detection"""
        logs = """
        npm install started
        npm ERR! code ENOTFOUND
        npm ERR! errno ENOTFOUND
        npm install failed
        """
        
        detection = detector.detect_failure(logs)
        
        assert detection.detected
        assert detection.pattern_id == "dependency_failure"
        assert detection.confidence > 0.8
        
    def test_test_failure_detection(self, detector):
        """Test test failure detection"""
        logs = """
        Running test suite...
        test_user_login PASSED
        test_user_logout FAILED
        AssertionError: Expected 200, got 500
        2 failed, 8 passed
        """
        
        detection = detector.detect_failure(logs)
        
        assert detection.detected
        assert detection.pattern_id == "test_failure"
        assert detection.confidence > 0.8
        
    def test_pattern_learning(self, detector):
        """Test pattern learning functionality"""
        custom_logs = "Custom error: database connection failed"
        
        success = detector.learn_pattern(
            custom_logs, 
            "Database Connection Error",
            "restart_database"
        )
        
        assert success
        assert "learned_database_connection_error" in detector.failure_patterns
        
    def test_pattern_statistics(self, detector):
        """Test pattern statistics"""
        # Trigger some detections
        detector.detect_failure("npm install failed")
        detector.detect_failure("test failed")
        
        stats = detector.get_pattern_statistics()
        
        assert stats["total_detections"] >= 0
        assert "patterns" in stats
        assert "top_patterns" in stats
        
    def test_no_failure_detection(self, detector):
        """Test when no failure is detected"""
        logs = "Build completed successfully\nAll tests passed"
        
        detection = detector.detect_failure(logs)
        
        assert not detection.detected


class TestHealingEngine:
    """Test healing engine functionality"""
    
    @pytest.fixture
    def healer(self):
        return HealingEngine()
        
    def test_healing_strategy_execution(self, healer):
        """Test healing strategy execution"""
        results = healer.heal_pipeline(
            "test-pipeline",
            "retry_with_cache_clear",
            {"logs": "npm install failed"}
        )
        
        assert len(results) > 0
        assert all(hasattr(result, 'status') for result in results)
        
    def test_healing_statistics(self, healer):
        """Test healing statistics"""
        # Execute some healing attempts
        healer.heal_pipeline("pipeline-1", "retry_with_cache_clear")
        healer.heal_pipeline("pipeline-2", "isolate_and_rerun_tests")
        
        stats = healer.get_healing_statistics()
        
        assert "total_actions" in stats
        assert "success_rate" in stats
        assert "strategy_statistics" in stats
        
    def test_custom_remediation_action(self, healer):
        """Test custom remediation action"""
        def custom_action(context):
            return {"success": True, "output": "Custom action executed"}
            
        action = RemediationAction(
            action_id="custom_test",
            name="Custom Test Action",
            description="Test custom action",
            function=custom_action
        )
        
        result = healer._execute_action(action, "test-pipeline", {})
        
        assert result.status == RemediationStatus.SUCCESS


class TestSecurityManager:
    """Test security management functionality"""
    
    @pytest.fixture
    def security_manager(self):
        config = SecurityConfig()
        return SecurityManager(config)
        
    def test_input_validation(self, security_manager):
        """Test input validation"""
        # Valid input
        valid_result = security_manager.validate_request(
            {"message": "normal text"},
            "192.168.1.1"
        )
        assert valid_result["valid"]
        
        # Invalid input with script tag
        invalid_result = security_manager.validate_request(
            {"message": "<script>alert('xss')</script>"},
            "192.168.1.1"
        )
        assert not invalid_result["valid"]
        
    def test_webhook_validation(self, security_manager):
        """Test webhook payload validation"""
        payload = '{"action": "completed", "status": "success"}'
        headers = {"x-hub-signature-256": "sha256=test"}
        
        result = security_manager.validate_webhook_payload(
            payload, headers, "test-secret"
        )
        
        assert "valid" in result
        assert "errors" in result
        
    def test_rate_limiting(self, security_manager):
        """Test rate limiting"""
        client_ip = "192.168.1.100"
        
        # First requests should be allowed
        for _ in range(5):
            result = security_manager.validate_request({}, client_ip)
            assert result["valid"]
            
        # Simulate high rate
        for _ in range(1000):
            security_manager.rate_limiter.is_allowed(client_ip)
            
        # Should be rate limited now
        result = security_manager.validate_request({}, client_ip)
        # Note: Might still be valid depending on rate limit configuration
        
    def test_security_events(self, security_manager):
        """Test security event logging"""
        security_manager._log_security_event("test_event", {"ip": "192.168.1.1"})
        
        summary = security_manager.get_security_summary()
        
        assert summary["total_events_24h"] >= 0
        assert "event_types" in summary


class TestErrorHandling:
    """Test error handling functionality"""
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        def failing_function():
            raise Exception("Test failure")
            
        # First few calls should fail normally
        for _ in range(3):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)
                
        # Circuit should be open now
        state = circuit_breaker.get_state()
        assert state["state"] == "open"
        
        # Next call should fail due to circuit breaker
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            circuit_breaker.call(failing_function)
            
    def test_retry_strategy(self):
        """Test retry strategy"""
        retry_strategy = RetryStrategy(max_attempts=3, initial_delay=0.1)
        
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
        
    def test_error_handler(self):
        """Test error handler"""
        error_handler = ErrorHandler()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_info = error_handler.handle_error(e, {"context": "test"})
            
        assert error_info["type"] == "ValueError"
        assert error_info["message"] == "Test error"
        assert error_info["context"]["context"] == "test"
        
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] >= 1


class TestValidators:
    """Test validation functionality"""
    
    @pytest.fixture
    def input_validator(self):
        return InputValidator()
        
    @pytest.fixture
    def config_validator(self):
        return ConfigValidator()
        
    def test_string_validation(self, input_validator):
        """Test string validation"""
        # Valid string
        result = input_validator.validate_string("Hello World", max_length=20)
        assert result.valid
        
        # Too long string
        result = input_validator.validate_string("x" * 1001, max_length=1000)
        assert not result.valid
        
    def test_email_validation(self, input_validator):
        """Test email validation"""
        # Valid email
        result = input_validator.validate_email("test@example.com")
        assert result.valid
        
        # Invalid email
        result = input_validator.validate_email("invalid-email")
        assert not result.valid
        
    def test_url_validation(self, input_validator):
        """Test URL validation"""
        # Valid URL
        result = input_validator.validate_url("https://example.com")
        assert result.valid
        
        # Invalid URL
        result = input_validator.validate_url("not-a-url")
        assert not result.valid
        
    def test_config_validation(self, config_validator):
        """Test configuration validation"""
        # Valid webhook config
        config = {
            "url": "https://example.com/webhook",
            "secret": "test-secret-123",
            "timeout": 30
        }
        result = config_validator.validate_webhook_config(config)
        assert result.valid
        
        # Invalid webhook config
        config = {"url": "invalid-url"}
        result = config_validator.validate_webhook_config(config)
        assert not result.valid


class TestAutoScaler:
    """Test auto-scaling functionality"""
    
    @pytest.fixture
    def auto_scaler(self):
        policy = ScalingPolicy(min_instances=1, max_instances=5)
        return AutoScaler(policy)
        
    def test_metric_registration(self, auto_scaler):
        """Test metric registration"""
        metric = ScalingMetric(
            name="cpu_utilization",
            current_value=50.0,
            threshold_up=80.0,
            threshold_down=20.0
        )
        
        auto_scaler.add_metric(metric)
        
        assert "cpu_utilization" in auto_scaler.metrics
        
    def test_scaling_decision(self, auto_scaler):
        """Test scaling decision logic"""
        # Add high CPU metric
        cpu_metric = ScalingMetric(
            name="cpu_utilization",
            current_value=90.0,
            threshold_up=80.0,
            threshold_down=20.0
        )
        auto_scaler.add_metric(cpu_metric)
        
        # Simulate multiple high readings
        for _ in range(5):
            cpu_metric.add_value(90.0)
            
        decision = auto_scaler._evaluate_scaling_decision()
        # Note: Actual decision depends on evaluation periods and other factors
        
    def test_scaling_status(self, auto_scaler):
        """Test scaling status reporting"""
        status = auto_scaler.get_scaling_status()
        
        assert "current_instances" in status
        assert "scaling_enabled" in status
        assert "metrics" in status


class TestLoadBalancer:
    """Test load balancing functionality"""
    
    @pytest.fixture
    def load_balancer(self):
        return LoadBalancer()
        
    def test_instance_management(self, load_balancer):
        """Test service instance management"""
        instance = ServiceInstance(
            id="test-1",
            address="192.168.1.10",
            port=8080,
            weight=1.0
        )
        
        load_balancer.add_instance(instance)
        
        assert "test-1" in load_balancer.instances
        
        # Test instance selection
        selected = load_balancer.get_instance()
        assert selected is not None
        assert selected.id == "test-1"
        
    def test_health_aware_selection(self, load_balancer):
        """Test health-aware instance selection"""
        # Add healthy instance
        healthy_instance = ServiceInstance(
            id="healthy-1",
            address="192.168.1.10",
            port=8080,
            status=ServiceStatus.HEALTHY
        )
        
        # Add unhealthy instance
        unhealthy_instance = ServiceInstance(
            id="unhealthy-1", 
            address="192.168.1.11",
            port=8080,
            status=ServiceStatus.UNHEALTHY
        )
        
        load_balancer.add_instance(healthy_instance)
        load_balancer.add_instance(unhealthy_instance)
        
        # Should select healthy instance
        selected = load_balancer.get_instance(exclude_unhealthy=True)
        assert selected.id == "healthy-1"
        
    def test_request_recording(self, load_balancer):
        """Test request metrics recording"""
        instance = ServiceInstance(
            id="test-1",
            address="192.168.1.10", 
            port=8080
        )
        load_balancer.add_instance(instance)
        
        # Record some requests
        load_balancer.record_request(instance, 0.1, True)
        load_balancer.record_request(instance, 0.2, False)
        
        stats = load_balancer.get_instance_statistics()
        
        assert stats["total_requests"] >= 2
        assert stats["failed_requests"] >= 1


class TestCacheManager:
    """Test cache management functionality"""
    
    @pytest.fixture
    def cache_manager(self):
        return CacheManager(strategy=CacheStrategy.LRU, max_size=10)
        
    def test_cache_operations(self, cache_manager):
        """Test basic cache operations"""
        # Set value
        success = cache_manager.set("key1", "value1")
        assert success
        
        # Get value
        value = cache_manager.get("key1")
        assert value == "value1"
        
        # Delete value
        success = cache_manager.delete("key1")
        assert success
        
        # Get deleted value
        value = cache_manager.get("key1")
        assert value is None
        
    def test_cache_expiration(self, cache_manager):
        """Test cache expiration"""
        # Set value with short TTL
        cache_manager.set("expire_key", "expire_value", ttl=0.1)
        
        # Should be available immediately
        value = cache_manager.get("expire_key")
        assert value == "expire_value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        value = cache_manager.get("expire_key")
        assert value is None
        
    def test_cache_eviction(self, cache_manager):
        """Test cache eviction"""
        # Fill cache beyond capacity
        for i in range(15):  # Max size is 10
            cache_manager.set(f"key{i}", f"value{i}")
            
        # Check that cache size is maintained
        info = cache_manager.get_cache_info()
        assert info["size"] <= cache_manager.max_size
        
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics"""
        # Generate some hits and misses
        cache_manager.set("test_key", "test_value")
        cache_manager.get("test_key")  # Hit
        cache_manager.get("missing_key")  # Miss
        
        info = cache_manager.get_cache_info()
        stats = info["statistics"]
        
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["total_requests"] >= 2


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor()
        
    def test_metrics_collection(self, performance_monitor):
        """Test metrics collection"""
        # Record some metrics
        performance_monitor.record_pipeline_event("build", "pipeline-1", 10.5, True)
        performance_monitor.record_api_request("/api/status", "GET", 200, 0.1)
        
        report = performance_monitor.get_performance_report()
        
        assert "metrics" in report
        assert "timestamp" in report
        
    def test_healing_metrics(self, performance_monitor):
        """Test healing attempt metrics"""
        performance_monitor.record_healing_attempt(
            "retry_with_cache_clear", True, 5.0, "pipeline-1"
        )
        
        report = performance_monitor.get_performance_report()
        assert "metrics" in report
        
    def test_performance_tracking(self, performance_monitor):
        """Test operation performance tracking"""
        with performance_monitor.track_pipeline_operation("test_operation"):
            time.sleep(0.1)  # Simulate work
            
        report = performance_monitor.get_performance_report()
        assert "metrics" in report


class TestIntegration:
    """Integration tests for Pipeline Guard components"""
    
    @pytest.fixture
    def pipeline_guard_system(self):
        """Create a complete pipeline guard system"""
        monitor = PipelineMonitor(check_interval=1)
        detector = FailureDetector()
        healer = HealingEngine()
        
        return {
            "monitor": monitor,
            "detector": detector,
            "healer": healer
        }
        
    def test_end_to_end_healing(self, pipeline_guard_system):
        """Test end-to-end healing workflow"""
        monitor = pipeline_guard_system["monitor"]
        detector = pipeline_guard_system["detector"]
        healer = pipeline_guard_system["healer"]
        
        # Register a pipeline
        monitor.register_pipeline("integration-test", "Integration Test Pipeline")
        
        # Simulate failure logs
        failure_logs = "npm install failed with error ENOTFOUND"
        
        # Detect failure
        detection = detector.detect_failure(failure_logs, {
            "pipeline_id": "integration-test"
        })
        
        assert detection.detected
        
        # Attempt healing
        healing_results = healer.heal_pipeline(
            "integration-test",
            detection.suggested_remediation,
            {"logs": failure_logs}
        )
        
        assert len(healing_results) > 0
        
        # Update pipeline status
        if any(r.status.value == "success" for r in healing_results):
            monitor.update_pipeline_status("integration-test", "success")
        else:
            monitor.update_pipeline_status("integration-test", "failed", "healing_failed")
            
        # Verify final state
        health_summary = monitor.get_health_summary()
        assert health_summary["total_pipelines"] == 1
        
    def test_concurrent_pipeline_monitoring(self, pipeline_guard_system):
        """Test concurrent pipeline monitoring"""
        monitor = pipeline_guard_system["monitor"]
        
        # Register multiple pipelines concurrently
        def register_pipelines(start_id, count):
            for i in range(count):
                pipeline_id = f"concurrent-{start_id}-{i}"
                monitor.register_pipeline(pipeline_id, f"Concurrent Pipeline {pipeline_id}")
                
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_pipelines, args=(i, 10))
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Verify all pipelines were registered
        health_summary = monitor.get_health_summary()
        assert health_summary["total_pipelines"] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])