#!/usr/bin/env python3
"""
Comprehensive test suite for memristor neural network simulator.
Combines all testing capabilities into a unified framework.
"""

import sys
sys.path.insert(0, '/root/repo')

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any

from memristor_nn.utils.logger import get_logger

logger = get_logger(__name__)

class ComprehensiveTestSuite:
    """Unified test suite runner."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def run_test_script(self, script_path: str, description: str) -> Dict[str, Any]:
        """Run a test script and capture results."""
        print(f"\n🧪 Running {description}...")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                status = "✅ PASSED"
                success = True
            else:
                status = "❌ FAILED"
                success = False
            
            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr and not success:
                print("STDERR:", result.stderr)
            
            return {
                'success': success,
                'execution_time_s': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'status': status
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'execution_time_s': time.time() - start_time,
                'return_code': -1,
                'stdout': '',
                'stderr': 'Test timed out after 300 seconds',
                'status': "⏰ TIMEOUT"
            }
        except Exception as e:
            return {
                'success': False,
                'execution_time_s': time.time() - start_time,
                'return_code': -2,
                'stdout': '',
                'stderr': str(e),
                'status': f"💥 ERROR: {str(e)}"
            }
    
    def run_python_tests(self):
        """Run existing Python-based tests."""
        print("\n🐍 PYTHON TEST SUITE")
        print("=" * 60)
        
        # Check for existing test files
        test_files = [
            ('tests/test_basic_functionality.py', 'Basic Functionality Tests'),
            ('tests/test_error_handling.py', 'Error Handling Tests'),
            ('test_basic_core.py', 'Basic Core Tests'),
            ('test_minimal_core.py', 'Minimal Core Tests'),
            ('test_core_functionality.py', 'Core Functionality Tests'),
            ('test_scaling_performance.py', 'Scaling & Performance Tests')
        ]
        
        for test_file, description in test_files:
            test_path = Path(test_file)
            if test_path.exists():
                result = self.run_test_script(str(test_path), description)
                self.results[test_file] = result
                print(f"  {result['status']} - {description} ({result['execution_time_s']:.2f}s)")
            else:
                print(f"  ⚠️  SKIPPED - {description} (file not found: {test_file})")
                self.results[test_file] = {
                    'success': None,
                    'execution_time_s': 0,
                    'status': 'SKIPPED - File not found'
                }
    
    def run_validation_tests(self):
        """Run validation and benchmark tests."""
        print("\n🔬 VALIDATION & BENCHMARKING")
        print("=" * 60)
        
        validation_scripts = [
            ('validate_implementation.py', 'Implementation Validation'),
            ('run_quality_gates.py', 'Quality Gates'),
        ]
        
        for script, description in validation_scripts:
            script_path = Path(script)
            if script_path.exists():
                result = self.run_test_script(str(script_path), description)
                self.results[script] = result
                print(f"  {result['status']} - {description} ({result['execution_time_s']:.2f}s)")
            else:
                print(f"  ⚠️  SKIPPED - {description} (file not found: {script})")
                self.results[script] = {
                    'success': None,
                    'execution_time_s': 0,
                    'status': 'SKIPPED - File not found'
                }
    
    def run_integration_tests(self):
        """Run integration tests with examples."""
        print("\n🔗 INTEGRATION TESTS")
        print("=" * 60)
        
        # Test if examples can be imported and run
        integration_tests = [
            ('examples/basic_usage.py', 'Basic Usage Example'),
            ('examples/complete_workflow.py', 'Complete Workflow Example'),
        ]
        
        for script, description in integration_tests:
            script_path = Path(script)
            if script_path.exists():
                # For examples, we'll just check if they can run without errors
                # (they might have torch dependencies that will be handled gracefully)
                result = self.run_test_script(str(script_path), description)
                self.results[script] = result
                print(f"  {result['status']} - {description} ({result['execution_time_s']:.2f}s)")
            else:
                print(f"  ⚠️  SKIPPED - {description} (file not found: {script})")
                self.results[script] = {
                    'success': None,
                    'execution_time_s': 0,
                    'status': 'SKIPPED - File not found'
                }
    
    def run_import_tests(self):
        """Test package imports and basic functionality."""
        print("\n📦 IMPORT & BASIC FUNCTIONALITY TESTS")
        print("=" * 60)
        
        import_tests = [
            # Test basic imports
            ("import memristor_nn", "Basic package import"),
            ("from memristor_nn.core.device_models import DeviceModel, DeviceConfig", "Device models import"),
            ("from memristor_nn.utils.logger import get_logger", "Logger import"),
            ("from memristor_nn.utils.error_handling import retry, CircuitBreaker", "Error handling import"),
            ("from memristor_nn.utils.security import sanitize_input", "Security utilities import"),
            ("from memristor_nn.utils.validators import validate_positive_number", "Validators import"),
        ]
        
        for import_code, description in import_tests:
            try:
                start_time = time.time()
                exec(import_code)
                execution_time = time.time() - start_time
                
                self.results[f"import_{description}"] = {
                    'success': True,
                    'execution_time_s': execution_time,
                    'status': '✅ PASSED'
                }
                print(f"  ✅ PASSED - {description}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.results[f"import_{description}"] = {
                    'success': False,
                    'execution_time_s': execution_time,
                    'status': f'❌ FAILED: {str(e)}'
                }
                print(f"  ❌ FAILED - {description}: {str(e)}")
    
    def run_torch_compatibility_tests(self):
        """Test PyTorch compatibility and graceful degradation."""
        print("\n🔥 PYTORCH COMPATIBILITY TESTS")
        print("=" * 60)
        
        # Test if torch is available
        torch_available = False
        try:
            import torch
            torch_available = True
            print("  ✅ PyTorch is available")
        except ImportError:
            print("  ⚠️  PyTorch not available - testing graceful degradation")
        
        # Test memristor_nn behavior with/without torch
        try:
            import memristor_nn as mn
            torch_status = getattr(mn, 'TORCH_AVAILABLE', False)
            
            if torch_available == torch_status:
                print(f"  ✅ TORCH_AVAILABLE correctly detected as {torch_status}")
                success = True
            else:
                print(f"  ❌ TORCH_AVAILABLE mismatch: expected {torch_available}, got {torch_status}")
                success = False
            
            self.results['torch_compatibility'] = {
                'success': success,
                'torch_installed': torch_available,
                'torch_detected': torch_status,
                'status': '✅ PASSED' if success else '❌ FAILED'
            }
            
        except Exception as e:
            print(f"  ❌ FAILED - Error testing torch compatibility: {str(e)}")
            self.results['torch_compatibility'] = {
                'success': False,
                'error': str(e),
                'status': f'❌ FAILED: {str(e)}'
            }
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        # Count results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('success') is True)
        failed_tests = sum(1 for r in self.results.values() if r.get('success') is False)
        skipped_tests = sum(1 for r in self.results.values() if r.get('success') is None)
        
        print(f"\n📈 SUMMARY:")
        print(f"  Total Tests:   {total_tests}")
        print(f"  ✅ Passed:     {passed_tests}")
        print(f"  ❌ Failed:     {failed_tests}")
        print(f"  ⚠️  Skipped:    {skipped_tests}")
        print(f"  ⏱️  Total Time: {total_time:.2f}s")
        
        if total_tests > 0:
            pass_rate = (passed_tests / (passed_tests + failed_tests)) * 100 if (passed_tests + failed_tests) > 0 else 0
            print(f"  📊 Pass Rate:  {pass_rate:.1f}%")
        
        # Detailed results
        print(f"\n🔍 DETAILED RESULTS:")
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            time_str = f"{result.get('execution_time_s', 0):.2f}s"
            print(f"  {status:<20} {test_name:<40} {time_str}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if failed_tests == 0:
            print("  🎉 All tests passed! The system is working correctly.")
        else:
            print("  🔧 Some tests failed. Review the detailed output above.")
        
        if skipped_tests > 0:
            print(f"  📝 {skipped_tests} tests were skipped (missing files or dependencies).")
        
        # Performance insights
        total_execution_time = sum(r.get('execution_time_s', 0) for r in self.results.values())
        if total_execution_time > 0:
            print(f"  ⚡ Total execution time: {total_execution_time:.2f}s")
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'pass_rate': pass_rate if total_tests > 0 else 0,
            'total_time_s': total_time,
            'results': self.results
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🚀 MEMRISTOR-NN COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"🕐 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.run_import_tests()
        self.run_torch_compatibility_tests()
        self.run_python_tests()
        self.run_validation_tests()
        self.run_integration_tests()
        
        # Generate final report
        return self.generate_test_report()

def main():
    """Run comprehensive test suite."""
    test_suite = ComprehensiveTestSuite()
    report = test_suite.run_all_tests()
    
    # Return appropriate exit code
    if report['failed'] == 0:
        return 0  # Success
    else:
        return 1  # Some tests failed

if __name__ == "__main__":
    sys.exit(main())