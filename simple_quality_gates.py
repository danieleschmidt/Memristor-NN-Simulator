#!/usr/bin/env python3
"""
Simplified Progressive Quality Gates - No External Dependencies
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class SimpleQualityResult:
    """Simple quality gate result."""
    gate_name: str
    passed: bool
    score: float
    execution_time: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class SimpleSecurityGate:
    """Simplified security validation."""
    
    def __init__(self):
        self.name = "SECURITY"
    
    def execute(self) -> SimpleQualityResult:
        start_time = time.time()
        
        try:
            issues = []
            
            # Check for potential secrets in code
            python_files = list(Path('.').rglob('*.py'))
            secret_patterns = ['password=', 'secret=', 'token=', 'api_key=']
            
            for file_path in python_files:
                if '__pycache__' in str(file_path) or 'test' in str(file_path).lower():
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8').lower()
                    for pattern in secret_patterns:
                        if pattern in content:
                            lines = content.split('\n')
                            for i, line in enumerate(lines):
                                if pattern in line and not line.strip().startswith('#'):
                                    issues.append(f"Potential secret in {file_path}:{i+1}")
                                    break
                except Exception:
                    continue
            
            execution_time = time.time() - start_time
            
            # Calculate score
            if len(issues) == 0:
                score = 1.0
                passed = True
            elif len(issues) <= 2:
                score = 0.8
                passed = False
            else:
                score = 0.5
                passed = False
            
            recommendations = []
            if issues:
                recommendations.append("Review and fix security issues")
            
            return SimpleQualityResult(
                gate_name=self.name,
                passed=passed,
                score=score,
                execution_time=execution_time,
                details={"issues_found": len(issues), "files_scanned": len(python_files)},
                error_message=f"Found {len(issues)} security issues" if issues else None,
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SimpleQualityResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={},
                error_message=str(e),
                recommendations=["Fix security gate execution"]
            )


class SimpleCodeQualityGate:
    """Simplified code quality validation."""
    
    def __init__(self):
        self.name = "CODE_QUALITY"
    
    def execute(self) -> SimpleQualityResult:
        start_time = time.time()
        
        try:
            python_files = list(Path('.').rglob('*.py'))
            python_files = [f for f in python_files if '__pycache__' not in str(f)]
            
            total_lines = 0
            long_files = 0
            
            for file_path in python_files:
                try:
                    content = file_path.read_text(encoding='utf-8')
                    lines = len(content.split('\n'))
                    total_lines += lines
                    
                    if lines > 500:
                        long_files += 1
                except Exception:
                    continue
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            avg_file_length = total_lines / len(python_files) if python_files else 0
            long_file_ratio = long_files / len(python_files) if python_files else 0
            
            # Calculate score
            length_score = 1.0 if avg_file_length <= 300 else max(0.5, 1.0 - (avg_file_length - 300) / 500)
            complexity_score = 1.0 - long_file_ratio
            
            overall_score = (length_score + complexity_score) / 2
            passed = avg_file_length <= 400 and long_files <= 2
            
            recommendations = []
            if avg_file_length > 400:
                recommendations.append("Consider breaking up large files")
            if long_files > 2:
                recommendations.append("Refactor very large files")
            
            return SimpleQualityResult(
                gate_name=self.name,
                passed=passed,
                score=overall_score,
                execution_time=execution_time,
                details={
                    "total_files": len(python_files),
                    "avg_file_length": avg_file_length,
                    "long_files": long_files
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SimpleQualityResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={},
                error_message=str(e),
                recommendations=["Fix code quality analysis"]
            )


class SimpleTestCoverageGate:
    """Simplified test coverage validation."""
    
    def __init__(self):
        self.name = "TEST_COVERAGE"
    
    def execute(self) -> SimpleQualityResult:
        start_time = time.time()
        
        try:
            # Find test files
            test_files = list(Path('.').rglob('test_*.py'))
            test_files.extend(list(Path('tests').rglob('*.py')))
            
            # Find source files
            source_files = list(Path('memristor_nn').rglob('*.py'))
            source_files = [f for f in source_files if '__pycache__' not in str(f)]
            
            # Count test functions
            total_test_functions = 0
            for test_file in test_files:
                try:
                    content = test_file.read_text(encoding='utf-8')
                    # Simple count of test functions
                    total_test_functions += content.count('def test_')
                except Exception:
                    continue
            
            execution_time = time.time() - start_time
            
            # Estimate coverage
            estimated_coverage = min(len(test_files) / max(len(source_files) * 0.3, 1), 1.0)
            
            # Calculate score
            coverage_score = estimated_coverage
            test_quantity_score = min(total_test_functions / 10, 1.0)  # Expect at least 10 test functions
            
            overall_score = (coverage_score + test_quantity_score) / 2
            passed = estimated_coverage >= 0.6 and total_test_functions >= 5
            
            recommendations = []
            if estimated_coverage < 0.8:
                recommendations.append("Increase test coverage")
            if total_test_functions < 10:
                recommendations.append("Add more test functions")
            
            return SimpleQualityResult(
                gate_name=self.name,
                passed=passed,
                score=overall_score,
                execution_time=execution_time,
                details={
                    "test_files": len(test_files),
                    "test_functions": total_test_functions,
                    "estimated_coverage": estimated_coverage,
                    "source_files": len(source_files)
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SimpleQualityResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                execution_time=execution_time,
                details={},
                error_message=str(e),
                recommendations=["Fix test coverage analysis"]
            )


def run_simple_quality_gates() -> Dict[str, Any]:
    """Run simplified quality gates."""
    print("🚀 Running Simplified Progressive Quality Gates")
    print("=" * 60)
    
    gates = [
        SimpleSecurityGate(),
        SimpleCodeQualityGate(),
        SimpleTestCoverageGate()
    ]
    
    start_time = time.time()
    results = []
    
    for gate in gates:
        print(f"Running {gate.name}...")
        result = gate.execute()
        results.append(result)
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status} {gate.name}: {result.score:.2f} ({result.execution_time:.2f}s)")
        
        if result.error_message:
            print(f"    Error: {result.error_message}")
        
        if result.recommendations:
            print(f"    Recommendations: {len(result.recommendations)}")
    
    total_time = time.time() - start_time
    
    # Calculate overall results
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        overall_passed = all(r.passed for r in results)
    else:
        avg_score = 0.0
        overall_passed = False
    
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
    
    summary = {
        "overall_passed": overall_passed,
        "quality_score": avg_score,
        "quality_grade": grade,
        "total_execution_time": total_time,
        "gate_results": [
            {
                "name": r.gate_name,
                "passed": r.passed,
                "score": r.score,
                "execution_time": r.execution_time,
                "details": r.details,
                "error_message": r.error_message,
                "recommendations": r.recommendations
            }
            for r in results
        ]
    }
    
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {'✅ PASS' if overall_passed else '❌ FAIL'}")
    print(f"Quality Score: {avg_score:.3f}/1.000")
    print(f"Quality Grade: {grade}")
    print(f"Total Time: {total_time:.1f}s")
    
    # Save report
    try:
        with open("simple_quality_report.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Report saved to simple_quality_report.json")
    except Exception as e:
        print(f"⚠️ Failed to save report: {e}")
    
    return summary


if __name__ == "__main__":
    import sys
    results = run_simple_quality_gates()
    sys.exit(0 if results["overall_passed"] else 1)