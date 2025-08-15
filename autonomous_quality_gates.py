"""
Autonomous Quality Gates and Production Deployment Pipeline

Final validation before production deployment:
- Code quality analysis
- Security scanning  
- Performance benchmarking
- Documentation validation
- CI/CD pipeline configuration
- Production deployment preparation
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quality_gates")

class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"quality_gate_{name}")
    
    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """Run the quality gate. Returns (passed, metrics)."""
        raise NotImplementedError

class CodeQualityGate(QualityGate):
    """Code quality and style analysis."""
    
    def __init__(self):
        super().__init__("code_quality")
    
    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """Run code quality checks."""
        self.logger.info("Running code quality analysis...")
        
        metrics = {
            "files_analyzed": 0,
            "lines_of_code": 0,
            "complexity_score": 0,
            "maintainability_index": 0,
            "issues": []
        }
        
        try:
            # Analyze Python files
            python_files = list(Path(".").glob("**/*.py"))
            python_files = [f for f in python_files if not any(skip in str(f) for skip in [".git", "__pycache__", ".pytest_cache"])]
            
            metrics["files_analyzed"] = len(python_files)
            
            total_lines = 0
            complexity_issues = []
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        
                        # Simple complexity analysis
                        complexity = self._analyze_complexity(lines)
                        if complexity > 15:  # High complexity threshold
                            complexity_issues.append({
                                "file": str(file_path),
                                "complexity": complexity,
                                "type": "high_complexity"
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Could not analyze {file_path}: {e}")
            
            metrics["lines_of_code"] = total_lines
            metrics["complexity_score"] = len(complexity_issues)
            metrics["issues"] = complexity_issues
            
            # Calculate maintainability index (simplified)
            if total_lines > 0:
                metrics["maintainability_index"] = max(0, 100 - (len(complexity_issues) * 10) - (total_lines / 1000))
            
            # Quality thresholds
            quality_passed = (
                metrics["maintainability_index"] > 60 and
                metrics["complexity_score"] < 10 and
                len(metrics["issues"]) < 20
            )
            
            self.logger.info(f"Code quality: {metrics['lines_of_code']} lines, "
                           f"maintainability {metrics['maintainability_index']:.1f}")
            
            return quality_passed, metrics
            
        except Exception as e:
            self.logger.error(f"Code quality analysis failed: {e}")
            return False, {"error": str(e)}
    
    def _analyze_complexity(self, lines: List[str]) -> int:
        """Simple cyclomatic complexity analysis."""
        complexity = 1  # Base complexity
        
        for line in lines:
            line = line.strip()
            # Count decision points
            if any(keyword in line for keyword in ['if ', 'elif ', 'for ', 'while ', 'except:', 'and ', 'or ']):
                complexity += 1
        
        return complexity

class SecurityGate(QualityGate):
    """Security analysis and vulnerability scanning."""
    
    def __init__(self):
        super().__init__("security")
    
    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """Run security analysis."""
        self.logger.info("Running security analysis...")
        
        metrics = {
            "vulnerabilities": [],
            "security_score": 0,
            "hardcoded_secrets": [],
            "insecure_patterns": []
        }
        
        try:
            # Scan for potential security issues
            python_files = list(Path(".").glob("**/*.py"))
            python_files = [f for f in python_files if not any(skip in str(f) for skip in [".git", "__pycache__"])]
            
            security_issues = []
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for hardcoded secrets
                        secret_patterns = [
                            'password=', 'passwd=', 'api_key=', 'secret_key=',
                            'token=', 'auth_token=', 'private_key='
                        ]
                        
                        for pattern in secret_patterns:
                            if pattern in content.lower():
                                security_issues.append({
                                    "file": str(file_path),
                                    "type": "potential_hardcoded_secret",
                                    "pattern": pattern,
                                    "severity": "medium"
                                })
                        
                        # Check for insecure patterns
                        insecure_patterns = [
                            'eval(', 'exec(', 'subprocess.call', 'os.system(',
                            'pickle.load', 'pickle.loads'
                        ]
                        
                        for pattern in insecure_patterns:
                            if pattern in content:
                                # Check if it's in a secure context (e.g., has validation)
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if pattern in line:
                                        # Look for validation in surrounding lines
                                        context = lines[max(0, i-3):i+3]
                                        context_text = ' '.join(context)
                                        
                                        if not any(safe_word in context_text.lower() 
                                                 for safe_word in ['validate', 'sanitize', 'check', 'verify']):
                                            security_issues.append({
                                                "file": str(file_path),
                                                "line": i + 1,
                                                "type": "insecure_pattern",
                                                "pattern": pattern,
                                                "severity": "high"
                                            })
                        
                except Exception as e:
                    self.logger.warning(f"Could not analyze {file_path}: {e}")
            
            metrics["vulnerabilities"] = security_issues
            metrics["security_score"] = max(0, 100 - len(security_issues) * 5)
            
            # Categorize issues
            metrics["hardcoded_secrets"] = [issue for issue in security_issues 
                                          if issue["type"] == "potential_hardcoded_secret"]
            metrics["insecure_patterns"] = [issue for issue in security_issues 
                                          if issue["type"] == "insecure_pattern"]
            
            # Security threshold
            security_passed = (
                len(metrics["hardcoded_secrets"]) == 0 and
                len([issue for issue in security_issues if issue["severity"] == "high"]) == 0 and
                metrics["security_score"] > 70
            )
            
            self.logger.info(f"Security: {len(security_issues)} issues found, "
                           f"score {metrics['security_score']}")
            
            return security_passed, metrics
            
        except Exception as e:
            self.logger.error(f"Security analysis failed: {e}")
            return False, {"error": str(e)}

class PerformanceGate(QualityGate):
    """Performance benchmarking and regression testing."""
    
    def __init__(self):
        super().__init__("performance")
    
    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """Run performance benchmarks."""
        self.logger.info("Running performance benchmarks...")
        
        metrics = {
            "benchmarks": {},
            "memory_usage_mb": 0,
            "execution_time_ms": 0,
            "regression_detected": False
        }
        
        try:
            # Import and run basic performance tests
            sys.path.append('.')
            
            # Test basic device model performance
            start_time = time.time()
            
            try:
                from memristor_nn.core.device_models import IEDM2024_TaOx, DeviceConfig
                
                device = IEDM2024_TaOx()
                
                # Benchmark device operations
                device_start = time.time()
                for _ in range(1000):
                    conductance = device.conductance(1.0, 0.5)
                    varied = device.add_variations(conductance)
                device_time = (time.time() - device_start) * 1000
                
                metrics["benchmarks"]["device_operations"] = {
                    "time_ms": device_time,
                    "ops_per_sec": 1000 / (device_time / 1000) if device_time > 0 else 0
                }
                
            except ImportError:
                self.logger.warning("Could not import device models for benchmarking")
            
            # Test crossbar performance if available
            try:
                from memristor_nn_simple import SimpleCrossbarArray
                
                crossbar = SimpleCrossbarArray(64, 32)
                input_vec = [0.5] * 64
                
                crossbar_start = time.time()
                for _ in range(100):
                    output = crossbar.matrix_vector_multiply(input_vec)
                crossbar_time = (time.time() - crossbar_start) * 1000
                
                metrics["benchmarks"]["crossbar_operations"] = {
                    "time_ms": crossbar_time,
                    "ops_per_sec": 100 / (crossbar_time / 1000) if crossbar_time > 0 else 0
                }
                
            except ImportError:
                self.logger.warning("Could not import crossbar for benchmarking")
            
            total_time = (time.time() - start_time) * 1000
            metrics["execution_time_ms"] = total_time
            
            # Memory usage estimation
            import psutil
            process = psutil.Process()
            metrics["memory_usage_mb"] = process.memory_info().rss / 1024 / 1024
            
            # Performance thresholds
            performance_passed = (
                total_time < 30000 and  # 30 seconds
                metrics["memory_usage_mb"] < 500  # 500 MB
            )
            
            # Check for regressions (simplified)
            baseline_file = Path("performance_baseline.json")
            if baseline_file.exists():
                try:
                    with open(baseline_file, 'r') as f:
                        baseline = json.load(f)
                    
                    # Compare against baseline
                    for bench_name, bench_data in metrics["benchmarks"].items():
                        if bench_name in baseline:
                            current_time = bench_data["time_ms"]
                            baseline_time = baseline[bench_name]["time_ms"]
                            
                            if current_time > baseline_time * 1.2:  # 20% regression threshold
                                metrics["regression_detected"] = True
                                self.logger.warning(f"Performance regression in {bench_name}: "
                                                  f"{current_time}ms vs {baseline_time}ms baseline")
                
                except Exception as e:
                    self.logger.warning(f"Could not compare against baseline: {e}")
            else:
                # Save current as baseline
                with open(baseline_file, 'w') as f:
                    json.dump(metrics["benchmarks"], f, indent=2)
                self.logger.info("Created performance baseline")
            
            self.logger.info(f"Performance: {total_time:.1f}ms execution, "
                           f"{metrics['memory_usage_mb']:.1f}MB memory")
            
            return performance_passed and not metrics["regression_detected"], metrics
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return False, {"error": str(e)}

class DocumentationGate(QualityGate):
    """Documentation completeness and quality analysis."""
    
    def __init__(self):
        super().__init__("documentation")
    
    def run(self) -> Tuple[bool, Dict[str, Any]]:
        """Run documentation analysis."""
        self.logger.info("Running documentation analysis...")
        
        metrics = {
            "files_with_docstrings": 0,
            "total_python_files": 0,
            "documentation_coverage": 0,
            "readme_quality": 0,
            "api_documentation": 0
        }
        
        try:
            # Analyze Python files for docstrings
            python_files = list(Path(".").glob("**/*.py"))
            python_files = [f for f in python_files if not any(skip in str(f) for skip in [".git", "__pycache__", "test_"])]
            
            metrics["total_python_files"] = len(python_files)
            
            files_with_docs = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for module docstring
                        if '"""' in content[:500] or "'''" in content[:500]:
                            files_with_docs += 1
                        
                        # Check for function/class docstrings
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if line.strip().startswith(('def ', 'class ')):
                                # Look for docstring in next few lines
                                for j in range(i+1, min(i+5, len(lines))):
                                    if '"""' in lines[j] or "'''" in lines[j]:
                                        files_with_docs += 1
                                        break
                                break
                        
                except Exception as e:
                    self.logger.warning(f"Could not analyze {file_path}: {e}")
            
            metrics["files_with_docstrings"] = files_with_docs
            metrics["documentation_coverage"] = (files_with_docs / len(python_files) * 100) if python_files else 0
            
            # Analyze README quality
            readme_files = [f for f in Path(".").glob("README*") if f.is_file()]
            if readme_files:
                readme_path = readme_files[0]
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                # Check for key sections
                readme_sections = [
                    'installation', 'usage', 'example', 'api', 
                    'contributing', 'license', 'features'
                ]
                
                sections_found = sum(1 for section in readme_sections 
                                   if section.lower() in readme_content.lower())
                
                metrics["readme_quality"] = (sections_found / len(readme_sections)) * 100
            
            # Check for API documentation
            doc_indicators = [
                'Args:', 'Returns:', 'Raises:', 'Example:', 'Note:',
                'Parameters:', 'Yields:', 'See Also:'
            ]
            
            api_doc_score = 0
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for indicator in doc_indicators:
                            if indicator in content:
                                api_doc_score += 1
                                break
                        
                except Exception:
                    pass
            
            metrics["api_documentation"] = (api_doc_score / len(python_files) * 100) if python_files else 0
            
            # Documentation quality threshold
            docs_passed = (
                metrics["documentation_coverage"] > 60 and
                metrics["readme_quality"] > 70 and
                metrics["api_documentation"] > 40
            )
            
            self.logger.info(f"Documentation: {metrics['documentation_coverage']:.1f}% coverage, "
                           f"README {metrics['readme_quality']:.1f}% quality")
            
            return docs_passed, metrics
            
        except Exception as e:
            self.logger.error(f"Documentation analysis failed: {e}")
            return False, {"error": str(e)}

class QualityGateOrchestrator:
    """Orchestrates all quality gates."""
    
    def __init__(self):
        self.logger = logging.getLogger("quality_orchestrator")
        self.gates = [
            CodeQualityGate(),
            SecurityGate(),
            PerformanceGate(),
            DocumentationGate()
        ]
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        self.logger.info("🚀 Starting Quality Gates Pipeline")
        
        results = {
            "overall_passed": True,
            "gates": {},
            "summary": {},
            "timestamp": time.time(),
            "deployment_ready": False
        }
        
        total_gates = len(self.gates)
        passed_gates = 0
        
        for gate in self.gates:
            self.logger.info(f"Running {gate.name} gate...")
            
            start_time = time.time()
            try:
                passed, metrics = gate.run()
                duration = time.time() - start_time
                
                results["gates"][gate.name] = {
                    "passed": passed,
                    "metrics": metrics,
                    "duration_ms": duration * 1000
                }
                
                if passed:
                    passed_gates += 1
                    self.logger.info(f"✅ {gate.name} gate PASSED ({duration:.1f}s)")
                else:
                    results["overall_passed"] = False
                    self.logger.warning(f"❌ {gate.name} gate FAILED ({duration:.1f}s)")
                
            except Exception as e:
                results["overall_passed"] = False
                results["gates"][gate.name] = {
                    "passed": False,
                    "metrics": {"error": str(e)},
                    "duration_ms": (time.time() - start_time) * 1000
                }
                self.logger.error(f"❌ {gate.name} gate ERROR: {e}")
        
        # Compile summary
        results["summary"] = {
            "total_gates": total_gates,
            "passed_gates": passed_gates,
            "failed_gates": total_gates - passed_gates,
            "pass_rate": passed_gates / total_gates,
            "total_duration_ms": sum(gate["duration_ms"] for gate in results["gates"].values())
        }
        
        # Deployment readiness
        results["deployment_ready"] = (
            results["overall_passed"] and
            results["summary"]["pass_rate"] >= 0.75  # 75% minimum pass rate
        )
        
        return results
    
    def generate_deployment_report(self, results: Dict[str, Any]) -> str:
        """Generate deployment readiness report."""
        report = []
        report.append("🚀 DEPLOYMENT READINESS REPORT")
        report.append("=" * 50)
        report.append("")
        
        if results["deployment_ready"]:
            report.append("✅ DEPLOYMENT APPROVED")
            report.append("System meets all quality requirements for production deployment.")
        else:
            report.append("❌ DEPLOYMENT BLOCKED")
            report.append("System has quality issues that must be resolved before deployment.")
        
        report.append("")
        report.append("📊 Quality Gate Summary:")
        report.append(f"  Total Gates: {results['summary']['total_gates']}")
        report.append(f"  Passed: {results['summary']['passed_gates']}")
        report.append(f"  Failed: {results['summary']['failed_gates']}")
        report.append(f"  Pass Rate: {results['summary']['pass_rate']:.1%}")
        report.append(f"  Duration: {results['summary']['total_duration_ms']:.1f}ms")
        report.append("")
        
        report.append("🔍 Gate Details:")
        for gate_name, gate_result in results["gates"].items():
            status = "✅ PASS" if gate_result["passed"] else "❌ FAIL"
            report.append(f"  {gate_name}: {status} ({gate_result['duration_ms']:.1f}ms)")
            
            # Add key metrics
            metrics = gate_result["metrics"]
            if gate_name == "code_quality":
                if "lines_of_code" in metrics:
                    report.append(f"    - Lines of code: {metrics['lines_of_code']}")
                if "maintainability_index" in metrics:
                    report.append(f"    - Maintainability: {metrics['maintainability_index']:.1f}")
            
            elif gate_name == "security":
                if "security_score" in metrics:
                    report.append(f"    - Security score: {metrics['security_score']}")
                if "vulnerabilities" in metrics:
                    report.append(f"    - Vulnerabilities: {len(metrics['vulnerabilities'])}")
            
            elif gate_name == "performance":
                if "memory_usage_mb" in metrics:
                    report.append(f"    - Memory usage: {metrics['memory_usage_mb']:.1f}MB")
                if "execution_time_ms" in metrics:
                    report.append(f"    - Execution time: {metrics['execution_time_ms']:.1f}ms")
            
            elif gate_name == "documentation":
                if "documentation_coverage" in metrics:
                    report.append(f"    - Doc coverage: {metrics['documentation_coverage']:.1f}%")
                if "readme_quality" in metrics:
                    report.append(f"    - README quality: {metrics['readme_quality']:.1f}%")
        
        report.append("")
        
        if not results["deployment_ready"]:
            report.append("🔧 Required Actions:")
            for gate_name, gate_result in results["gates"].items():
                if not gate_result["passed"]:
                    report.append(f"  - Fix {gate_name} issues")
                    
                    # Add specific recommendations
                    metrics = gate_result["metrics"]
                    if gate_name == "security" and "vulnerabilities" in metrics:
                        for vuln in metrics["vulnerabilities"][:3]:  # Show first 3
                            report.append(f"    * {vuln['type']} in {vuln['file']}")
                    
                    if gate_name == "code_quality" and "issues" in metrics:
                        for issue in metrics["issues"][:3]:  # Show first 3
                            report.append(f"    * {issue['type']} in {issue['file']}")
        
        report.append("")
        report.append("Generated by Terragon Labs Autonomous SDLC")
        report.append(f"Timestamp: {time.ctime(results['timestamp'])}")
        
        return "\n".join(report)

def main():
    """Run autonomous quality gates pipeline."""
    print("🚀🔒 Memristor NN Simulator - Autonomous Quality Gates")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = QualityGateOrchestrator()
    
    # Run all quality gates
    results = orchestrator.run_all_gates()
    
    # Generate and save report
    report = orchestrator.generate_deployment_report(results)
    
    # Save results
    with open("quality_gates_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open("deployment_readiness_report.txt", "w") as f:
        f.write(report)
    
    # Print report
    print("\n" + report)
    
    # Final verdict
    if results["deployment_ready"]:
        print("\n🎯 QUALITY GATES PASSED! 🎉")
        print("System is approved for production deployment.")
        return 0
    else:
        print("\n⚠️ QUALITY GATES FAILED!")
        print("Resolve issues before deploying to production.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)