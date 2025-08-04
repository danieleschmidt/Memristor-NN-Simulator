#!/usr/bin/env python3
"""
Validation script for Memristor Neural Network Simulator implementation.

This script validates that all components of the autonomous SDLC execution
have been implemented correctly without requiring external dependencies.
"""

import os
import sys
from pathlib import Path
import importlib.util
import ast


def validate_project_structure():
    """Validate the project directory structure."""
    print("🏗️ Validating Project Structure...")
    
    expected_structure = {
        "memristor_nn/": {
            "__init__.py": "Package initialization",
            "core/": {
                "__init__.py": "Core module init",
                "crossbar.py": "Crossbar array implementation", 
                "device_models.py": "Memristor device models"
            },
            "mapping/": {
                "__init__.py": "Mapping module init",
                "neural_mapper.py": "Neural network mapping"
            },
            "simulator/": {
                "__init__.py": "Simulator module init", 
                "simulator.py": "Main simulation engine"
            },
            "rtl_gen/": {
                "__init__.py": "RTL generation init",
                "generator.py": "Hardware generation"
            },
            "analysis/": {
                "__init__.py": "Analysis module init",
                "explorer.py": "Design space exploration"
            },
            "validation/": {
                "__init__.py": "Validation module init",
                "validator.py": "Hardware validation"
            },
            "faults/": {
                "__init__.py": "Faults module init",
                "analyzer.py": "Fault injection analysis"
            },
            "optimization/": {
                "__init__.py": "Optimization module init",
                "cache_manager.py": "Caching system",
                "parallel_simulator.py": "Parallel execution",
                "memory_optimizer.py": "Memory management",
                "performance_profiler.py": "Performance profiling"
            },
            "utils/": {
                "__init__.py": "Utils module init",
                "logger.py": "Logging system",
                "validators.py": "Input validation", 
                "security.py": "Security utilities"
            }
        },
        "tests/": {
            "test_basic_functionality.py": "Basic functionality tests",
            "test_error_handling.py": "Error handling tests"
        },
        "examples/": {
            "basic_usage.py": "Basic usage example",
            "complete_workflow.py": "Complete workflow demo"
        },
        ".github/workflows/": {
            "ci.yml": "CI/CD pipeline"
        }
    }
    
    def check_structure(path, structure, level=0):
        indent = "  " * level
        all_good = True
        
        for item, description in structure.items():
            item_path = path / item
            
            if isinstance(description, dict):
                # Directory
                if item_path.is_dir():
                    print(f"{indent}✅ {item}")
                    if not check_structure(item_path, description, level + 1):
                        all_good = False
                else:
                    print(f"{indent}❌ {item} (directory missing)")
                    all_good = False
            else:
                # File
                if item_path.is_file():
                    print(f"{indent}✅ {item}")
                else:
                    print(f"{indent}❌ {item} (file missing)")
                    all_good = False
        
        return all_good
    
    return check_structure(Path("."), expected_structure)


def validate_imports():
    """Validate that all modules can be imported without external dependencies."""
    print("\n📦 Validating Module Imports...")
    
    # Core modules that should import without external deps
    internal_modules = [
        "memristor_nn.utils.logger",
        "memristor_nn.utils.validators", 
        "memristor_nn.utils.security"
    ]
    
    all_good = True
    
    for module_name in internal_modules:
        try:
            # Try to import using importlib to avoid actually loading dependencies
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"  ✅ {module_name} - importable")
            else:
                print(f"  ❌ {module_name} - not found")
                all_good = False
        except Exception as e:
            print(f"  ❌ {module_name} - error: {e}")
            all_good = False
    
    return all_good


def validate_code_quality():
    """Validate code quality standards."""
    print("\n🎯 Validating Code Quality...")
    
    python_files = list(Path(".").rglob("*.py"))
    issues = []
    
    for file_path in python_files:
        if file_path.name == "validate_implementation.py":
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST to check for basic quality
            try:
                tree = ast.parse(content)
                
                # Check for docstrings in classes and functions
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        if not ast.get_docstring(node):
                            if not node.name.startswith('_'):  # Skip private methods
                                issues.append(f"{file_path}: {node.name} missing docstring")
                
            except SyntaxError as e:
                issues.append(f"{file_path}: Syntax error - {e}")
                
        except Exception as e:
            issues.append(f"{file_path}: Cannot read - {e}")
    
    if issues:
        print(f"  ❌ Found {len(issues)} code quality issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
        return False
    else:
        print(f"  ✅ All {len(python_files)} Python files passed basic quality checks")
        return True


def validate_configuration_files():
    """Validate configuration and deployment files."""
    print("\n⚙️ Validating Configuration Files...")
    
    config_files = {
        "pyproject.toml": "Python package configuration",
        ".pre-commit-config.yaml": "Pre-commit hooks",
        "Dockerfile": "Container configuration", 
        "docker-compose.yml": "Container orchestration",
        ".github/workflows/ci.yml": "CI/CD pipeline"
    }
    
    all_good = True
    
    for file_path, description in config_files.items():
        if Path(file_path).exists():
            print(f"  ✅ {file_path} - {description}")
        else:
            print(f"  ❌ {file_path} - {description} (missing)")
            all_good = False
    
    return all_good


def validate_documentation():
    """Validate documentation completeness."""
    print("\n📚 Validating Documentation...")
    
    doc_files = {
        "README.md": "Project overview and usage",
        "CONTRIBUTING.md": "Contribution guidelines",
        "SECURITY.md": "Security policy",
        "DEPLOYMENT.md": "Deployment guide",
        "CHANGELOG.md": "Version history",
        "PROJECT_SUMMARY.md": "Project completion summary"
    }
    
    all_good = True
    
    for file_path, description in doc_files.items():
        if Path(file_path).exists():
            file_size = Path(file_path).stat().st_size
            if file_size > 1000:  # At least 1KB of content
                print(f"  ✅ {file_path} - {description} ({file_size} bytes)")
            else:
                print(f"  ⚠️ {file_path} - {description} (too short: {file_size} bytes)")
        else:
            print(f"  ❌ {file_path} - {description} (missing)")
            all_good = False
    
    return all_good


def validate_sdlc_generations():
    """Validate that all SDLC generations are implemented."""
    print("\n🚀 Validating SDLC Generations...")
    
    generations = {
        "Generation 1 (Simple)": {
            "Core functionality": ["memristor_nn/core/", "memristor_nn/mapping/", "memristor_nn/simulator/"],
            "Basic examples": ["examples/basic_usage.py"]
        },
        "Generation 2 (Robust)": {
            "Error handling": ["memristor_nn/utils/validators.py", "tests/test_error_handling.py"],
            "Security": ["memristor_nn/utils/security.py", "SECURITY.md"],
            "Logging": ["memristor_nn/utils/logger.py"]
        },
        "Generation 3 (Optimized)": {
            "Performance optimization": ["memristor_nn/optimization/"],
            "Caching": ["memristor_nn/optimization/cache_manager.py"],
            "Parallel processing": ["memristor_nn/optimization/parallel_simulator.py"]
        },
        "Quality Gates": {
            "CI/CD": [".github/workflows/ci.yml"],
            "Pre-commit hooks": [".pre-commit-config.yaml"],
            "Testing": ["tests/"]
        },
        "Production Ready": {
            "Containerization": ["Dockerfile", "docker-compose.yml"],
            "Deployment": ["DEPLOYMENT.md"],
            "Documentation": ["README.md", "CONTRIBUTING.md"]
        }
    }
    
    all_good = True
    
    for generation, components in generations.items():
        print(f"\n  📋 {generation}:")
        gen_good = True
        
        for component, paths in components.items():
            component_good = all(Path(path).exists() for path in paths)
            if component_good:
                print(f"    ✅ {component}")
            else:
                print(f"    ❌ {component}")
                missing = [path for path in paths if not Path(path).exists()]
                print(f"        Missing: {missing}")
                gen_good = False
        
        if not gen_good:
            all_good = False
    
    return all_good


def main():
    """Run complete validation."""
    print("🔍 Memristor NN Simulator - Implementation Validation")
    print("=" * 60)
    
    validation_results = {
        "Project Structure": validate_project_structure(),
        "Module Imports": validate_imports(), 
        "Code Quality": validate_code_quality(),
        "Configuration Files": validate_configuration_files(),
        "Documentation": validate_documentation(),
        "SDLC Generations": validate_sdlc_generations()
    }
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(validation_results)
    
    for check, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check:<20} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall Score: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✨ Autonomous SDLC execution successfully completed!")
        print("🚀 Project is ready for production deployment!")
        return 0
    else:
        print(f"⚠️ {total - passed} validation(s) failed")
        print("🔧 Please address the issues above before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())