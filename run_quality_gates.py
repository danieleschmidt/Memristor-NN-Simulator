#!/usr/bin/env python3
"""Run comprehensive quality gates for memristor neural network simulator."""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from memristor_nn.testing.comprehensive_test_suite import run_comprehensive_tests
from memristor_nn.testing.performance_benchmarks import run_performance_benchmarks
from memristor_nn.utils.security import SecurityError
from memristor_nn.utils.error_handling import get_error_collector


class QualityGate:
    """Individual quality gate with pass/fail criteria."""
    
    def __init__(self, name: str, description: str, required: bool = True):
        self.name = name
        self.description = description
        self.required = required
        self.passed = False
        self.score = 0.0
        self.details = {}
        self.error_message = None
    
    def run(self) -> bool:
        """Run the quality gate. Override in subclasses."""
        raise NotImplementedError
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        required = " (REQUIRED)" if self.required else " (OPTIONAL)"
        return f"{status} {self.name}{required}: {self.description}"


class TestSuiteGate(QualityGate):
    """Quality gate for comprehensive test suite."""
    
    def __init__(self):
        super().__init__(
            "TEST_SUITE",
            "Comprehensive test suite with >85% pass rate",
            required=True
        )
    
    def run(self) -> bool:
        """Run comprehensive test suite."""
        try:
            print("Running comprehensive test suite...")
            test_results = run_comprehensive_tests()
            
            self.score = test_results['success_rate']
            self.details = test_results
            
            # Pass if success rate >= 85%
            self.passed = test_results['success_rate'] >= 0.85 and test_results['passed']
            
            if not self.passed:
                self.error_message = f"Test success rate {self.score:.1%} < 85% or tests failed"
            
            return self.passed
            
        except Exception as e:
            self.error_message = f"Test suite execution failed: {e}"
            return False


class PerformanceBenchmarkGate(QualityGate):
    """Quality gate for performance benchmarks."""
    
    def __init__(self):
        super().__init__(
            "PERFORMANCE",
            "Performance benchmarks with grade C or better",
            required=True
        )
    
    def run(self) -> bool:
        """Run performance benchmarks."""
        try:
            print("Running performance benchmarks...")
            benchmark_report = run_performance_benchmarks()
            
            self.details = benchmark_report
            grade = benchmark_report.get('performance_grade', 'F')
            success_rate = benchmark_report['summary']['success_rate']
            
            # Pass if grade is C or better and success rate >= 80%
            passing_grades = ['A', 'B', 'C']
            self.passed = grade in passing_grades and success_rate >= 0.8
            self.score = success_rate
            
            if not self.passed:
                self.error_message = f"Performance grade {grade} or success rate {success_rate:.1%} insufficient"
            
            return self.passed
            
        except Exception as e:
            self.error_message = f"Performance benchmark failed: {e}"
            return False


class SecurityAuditGate(QualityGate):
    """Quality gate for security audit."""
    
    def __init__(self):
        super().__init__(
            "SECURITY",
            "Security audit with no critical vulnerabilities",
            required=True
        )
    
    def run(self) -> bool:
        """Run security audit."""
        try:
            print("Running security audit...")
            
            security_issues = []
            
            # Test input validation
            try:
                from memristor_nn.utils.security import sanitize_input, SecurityError
                
                # Test cases that should raise SecurityError
                dangerous_inputs = [
                    "x" * 2000,  # Too long
                    "<script>alert('xss')</script>",  # Script injection
                    "../../../etc/passwd",  # Path traversal
                ]
                
                for dangerous_input in dangerous_inputs:
                    try:
                        sanitize_input(dangerous_input)
                        security_issues.append(f"Failed to reject dangerous input: {dangerous_input[:50]}...")
                    except SecurityError:
                        pass  # Expected behavior
                
            except ImportError:
                security_issues.append("Security module not available")
            
            # Check for hardcoded secrets (basic scan)
            project_files = list(Path('.').rglob('*.py'))
            for file_path in project_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if any(keyword in content.lower() for keyword in ['password=', 'secret=', 'token=', 'api_key=']):
                        # Check if it's in a test file or example
                        if 'test' not in str(file_path).lower() and 'example' not in str(file_path).lower():
                            security_issues.append(f"Potential hardcoded secret in {file_path}")
                except Exception:
                    continue
            
            self.details = {
                'issues_found': len(security_issues),
                'issues': security_issues
            }
            
            # Pass if no critical security issues
            self.passed = len(security_issues) == 0
            self.score = 1.0 - min(len(security_issues) / 10, 1.0)  # Normalize to 0-1
            
            if not self.passed:
                self.error_message = f"Found {len(security_issues)} security issues"
            
            return self.passed
            
        except Exception as e:
            self.error_message = f"Security audit failed: {e}"
            return False


class CodeQualityGate(QualityGate):
    """Quality gate for code quality metrics."""
    
    def __init__(self):
        super().__init__(
            "CODE_QUALITY",
            "Code quality metrics within acceptable ranges",
            required=False
        )
    
    def run(self) -> bool:
        """Run code quality analysis."""
        try:
            print("Running code quality analysis...")
            
            # Count lines of code
            python_files = list(Path('.').rglob('*.py'))
            total_lines = 0
            total_files = len(python_files)
            
            for file_path in python_files:
                try:
                    lines = len(file_path.read_text(encoding='utf-8').splitlines())
                    total_lines += lines
                except Exception:
                    continue
            
            # Calculate basic metrics
            avg_file_length = total_lines / total_files if total_files > 0 else 0
            
            # Check for very long files (potential code smell)
            long_files = 0
            for file_path in python_files:
                try:
                    lines = len(file_path.read_text(encoding='utf-8').splitlines())
                    if lines > 500:  # Consider files over 500 lines as "long"
                        long_files += 1
                except Exception:
                    continue
            
            # Error collection analysis
            error_collector = get_error_collector()
            error_summary = error_collector.get_error_summary()
            
            self.details = {
                'total_files': total_files,
                'total_lines': total_lines,
                'avg_file_length': avg_file_length,
                'long_files': long_files,
                'error_summary': error_summary
            }
            
            # Pass if metrics are reasonable
            quality_score = 1.0
            
            # Penalize very long average file length
            if avg_file_length > 400:
                quality_score -= 0.2
            
            # Penalize too many long files
            if long_files / total_files > 0.3:  # More than 30% long files
                quality_score -= 0.2
            
            # Penalize recent errors
            if error_summary.get('recent_errors', 0) > 5:
                quality_score -= 0.3
            
            self.score = max(quality_score, 0.0)
            self.passed = self.score >= 0.7
            
            if not self.passed:
                self.error_message = f"Code quality score {self.score:.2f} < 0.70"
            
            return self.passed
            
        except Exception as e:
            self.error_message = f"Code quality analysis failed: {e}"
            return False


class DocumentationGate(QualityGate):
    """Quality gate for documentation completeness."""
    
    def __init__(self):
        super().__init__(
            "DOCUMENTATION",
            "Essential documentation files present and complete",
            required=False
        )
    
    def run(self) -> bool:
        """Check documentation completeness."""
        try:
            print("Checking documentation completeness...")
            
            required_docs = [
                'README.md',
                'CONTRIBUTING.md',
                'LICENSE',
                'SECURITY.md'
            ]
            
            missing_docs = []
            incomplete_docs = []
            
            for doc_file in required_docs:
                doc_path = Path(doc_file)
                if not doc_path.exists():
                    missing_docs.append(doc_file)
                else:
                    # Check if file is substantial (more than 100 chars)
                    try:
                        content = doc_path.read_text(encoding='utf-8')
                        if len(content.strip()) < 100:
                            incomplete_docs.append(doc_file)
                    except Exception:
                        incomplete_docs.append(doc_file)
            
            # Check for docstrings in Python modules
            python_files = list(Path('memristor_nn').rglob('*.py'))
            undocumented_modules = 0
            
            for py_file in python_files:
                if '__pycache__' in str(py_file):
                    continue
                    
                try:
                    content = py_file.read_text(encoding='utf-8')
                    # Check if file has a module-level docstring
                    lines = content.strip().split('\n')
                    has_docstring = False
                    
                    for line in lines[:10]:  # Check first 10 lines
                        line = line.strip()
                        if line.startswith('"""') or line.startswith("'''"):
                            has_docstring = True
                            break
                        elif line and not line.startswith('#') and not line.startswith('import'):
                            break
                    
                    if not has_docstring:
                        undocumented_modules += 1
                        
                except Exception:
                    continue
            
            total_modules = len(python_files)
            documentation_coverage = 1.0 - (undocumented_modules / total_modules) if total_modules > 0 else 1.0
            
            self.details = {
                'required_docs': required_docs,
                'missing_docs': missing_docs,
                'incomplete_docs': incomplete_docs,
                'total_modules': total_modules,
                'undocumented_modules': undocumented_modules,
                'documentation_coverage': documentation_coverage
            }
            
            # Pass if most required docs exist and documentation coverage > 60%
            docs_score = 1.0 - (len(missing_docs) + len(incomplete_docs)) / len(required_docs)
            overall_score = (docs_score + documentation_coverage) / 2
            
            self.score = overall_score
            self.passed = overall_score >= 0.6
            
            if not self.passed:
                self.error_message = f"Documentation score {overall_score:.2f} < 0.60"
            
            return self.passed
            
        except Exception as e:
            self.error_message = f"Documentation check failed: {e}"
            return False


def run_quality_gates() -> Dict[str, Any]:
    """Run all quality gates and return comprehensive results."""
    print("MEMRISTOR NEURAL NETWORK SIMULATOR - QUALITY GATES")
    print("=" * 60)
    
    # Initialize quality gates
    gates = [
        TestSuiteGate(),
        PerformanceBenchmarkGate(),
        SecurityAuditGate(),
        CodeQualityGate(),
        DocumentationGate()
    ]
    
    # Run all gates
    start_time = time.time()
    results = []
    
    for gate in gates:
        gate_start = time.time()
        success = gate.run()
        gate_time = time.time() - gate_start
        
        result = {
            'name': gate.name,
            'description': gate.description,
            'required': gate.required,
            'passed': gate.passed,
            'score': gate.score,
            'execution_time': gate_time,
            'details': gate.details,
            'error_message': gate.error_message
        }
        
        results.append(result)
        
        print(f"{gate} (Score: {gate.score:.2f}, Time: {gate_time:.1f}s)")
        if gate.error_message:
            print(f"  Error: {gate.error_message}")
    
    total_time = time.time() - start_time
    
    # Calculate overall results
    required_gates = [r for r in results if r['required']]
    optional_gates = [r for r in results if not r['required']]
    
    required_passed = sum(1 for r in required_gates if r['passed'])
    optional_passed = sum(1 for r in optional_gates if r['passed'])
    
    overall_passed = required_passed == len(required_gates)
    
    # Calculate quality score
    total_score = sum(r['score'] for r in results)
    avg_score = total_score / len(results) if results else 0.0
    
    summary = {
        'overall_passed': overall_passed,
        'required_gates_passed': f"{required_passed}/{len(required_gates)}",
        'optional_gates_passed': f"{optional_passed}/{len(optional_gates)}",
        'quality_score': avg_score,
        'total_execution_time': total_time,
        'gate_results': results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {'✓ PASS' if overall_passed else '✗ FAIL'}")
    print(f"Required Gates: {required_passed}/{len(required_gates)} passed")
    print(f"Optional Gates: {optional_passed}/{len(optional_gates)} passed")
    print(f"Quality Score: {avg_score:.2f}/1.00")
    print(f"Total Time: {total_time:.1f}s")
    
    # Quality grade
    if avg_score >= 0.9:
        grade = "A"
    elif avg_score >= 0.8:
        grade = "B"
    elif avg_score >= 0.7:
        grade = "C"
    elif avg_score >= 0.6:
        grade = "D"
    else:
        grade = "F"
    
    print(f"Quality Grade: {grade}")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    results = run_quality_gates()
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_passed'] else 1)