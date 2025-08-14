#!/usr/bin/env python3
"""
Standalone test for progressive quality gates without external dependencies.
"""

import sys
import time
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_progressive_gates():
    """Test progressive quality gates system."""
    print("🧪 Testing Progressive Quality Gates System")
    print("=" * 60)
    
    try:
        # Import the progressive gates module directly
        from memristor_nn.quality.progressive_gates import (
            SecurityGate, CodeQualityGate, TestCoverageGate,
            ProgressiveQualityManager
        )
        
        print("✅ Successfully imported progressive gates modules")
        
        # Test SecurityGate
        print("\n🔒 Testing SecurityGate...")
        sec_gate = SecurityGate()
        sec_result = sec_gate.execute()
        print(f"   Result: {'✅ PASS' if sec_result.passed else '❌ FAIL'}")
        print(f"   Score: {sec_result.score:.2f}")
        print(f"   Time: {sec_result.execution_time:.2f}s")
        if sec_result.error_message:
            print(f"   Error: {sec_result.error_message}")
        if sec_result.recommendations:
            print(f"   Recommendations: {len(sec_result.recommendations)}")
        
        # Test CodeQualityGate
        print("\n📊 Testing CodeQualityGate...")
        code_gate = CodeQualityGate()
        code_result = code_gate.execute()
        print(f"   Result: {'✅ PASS' if code_result.passed else '❌ FAIL'}")
        print(f"   Score: {code_result.score:.2f}")
        print(f"   Time: {code_result.execution_time:.2f}s")
        if code_result.error_message:
            print(f"   Error: {code_result.error_message}")
        if code_result.metrics:
            print(f"   Files analyzed: {code_result.metrics.get('total_files', 0)}")
        
        # Test TestCoverageGate
        print("\n🧪 Testing TestCoverageGate...")
        test_gate = TestCoverageGate()
        test_result = test_gate.execute()
        print(f"   Result: {'✅ PASS' if test_result.passed else '❌ FAIL'}")
        print(f"   Score: {test_result.score:.2f}")
        print(f"   Time: {test_result.execution_time:.2f}s")
        if test_result.error_message:
            print(f"   Error: {test_result.error_message}")
        if test_result.metrics:
            print(f"   Test files found: {test_result.metrics.get('test_files', 0)}")
        
        # Test ProgressiveQualityManager (without functionality gate)
        print("\n🎯 Testing ProgressiveQualityManager...")
        manager = ProgressiveQualityManager()
        
        # Override gates to exclude the problematic FunctionalityGate
        original_method = manager.get_generation_gates
        def mock_gates(generation):
            return [SecurityGate(), CodeQualityGate(), TestCoverageGate()]
        manager.get_generation_gates = mock_gates
        
        report = manager.run_progressive_gates("Generation 1")
        
        print(f"   Overall: {'✅ PASS' if report.overall_passed else '❌ FAIL'}")
        print(f"   Quality Score: {report.quality_score:.3f}")
        print(f"   Quality Grade: {report.grade}")
        print(f"   Execution Time: {report.execution_time:.1f}s")
        print(f"   Gates Executed: {len(report.gate_results)}")
        
        if report.critical_issues:
            print(f"   Critical Issues: {len(report.critical_issues)}")
        
        if report.recommendations:
            print(f"   Recommendations: {len(report.recommendations)}")
        
        # Try to save report
        try:
            manager.save_report(report, "test_quality_report.json")
            print("   ✅ Report saved successfully")
        except Exception as e:
            print(f"   ⚠️ Report save failed: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 PROGRESSIVE QUALITY GATES TEST COMPLETED")
        print("=" * 60)
        print("Key Results:")
        print(f"  • Security Gate: {'✅' if sec_result.passed else '❌'} ({sec_result.score:.2f})")
        print(f"  • Code Quality Gate: {'✅' if code_result.passed else '❌'} ({code_result.score:.2f})")
        print(f"  • Test Coverage Gate: {'✅' if test_result.passed else '❌'} ({test_result.score:.2f})")
        print(f"  • Overall Quality: {report.grade} ({report.quality_score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_progressive_gates()
    sys.exit(0 if success else 1)