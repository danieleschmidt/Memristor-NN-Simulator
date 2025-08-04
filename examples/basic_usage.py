"""
Basic usage example for Memristor Neural Network Simulator.

This example demonstrates the core functionality of the simulator:
1. Creating crossbar arrays with different device models
2. Mapping neural networks to crossbar hardware
3. Running simulations with device variations
4. Design space exploration
5. Fault injection analysis
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import memristor_nn as mn


def create_simple_mlp():
    """Create a simple Multi-Layer Perceptron for demonstration."""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model


def demonstrate_basic_simulation():
    """Demonstrate basic crossbar simulation."""
    print("=== Basic Simulation Demo ===")
    
    # Create neural network
    model = create_simple_mlp()
    print(f"Created MLP with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create crossbar array with TaOx devices
    crossbar = mn.CrossbarArray(
        rows=128,
        cols=128, 
        device_model='IEDM2024_TaOx',
        tile_size=128
    )
    print(f"Created {crossbar.rows}x{crossbar.cols} crossbar with {crossbar.device_model.name}")
    
    # Map neural network to crossbar
    mapped_model = mn.map_to_crossbar(model, crossbar)
    hw_stats = mapped_model.get_hardware_stats()
    print(f"Mapped model uses {hw_stats['total_devices']} devices across {hw_stats['crossbar_count']} crossbars")
    
    # Generate synthetic test data
    test_data = torch.randn(1000, 784)
    
    # Run simulation with device variations
    print("Running simulation with device variations...")
    results = mn.simulate(
        mapped_model,
        test_data,
        include_noise=True,
        temperature=300,  # Room temperature
        max_batches=10
    )
    
    print(f"Results:")
    print(f"  Accuracy: {results.accuracy:.3f}")
    print(f"  Power: {results.power_mw:.2f} mW")
    print(f"  Latency: {results.latency_us:.2f} μs")  
    print(f"  Energy: {results.energy_pj:.2f} pJ/inference")
    print(f"  Area: {results.area_mm2:.3f} mm²")
    print()


def demonstrate_device_comparison():
    """Compare different memristor device technologies."""
    print("=== Device Technology Comparison ===")
    
    model = create_simple_mlp()
    test_data = torch.randn(500, 784)
    
    device_models = ['IEDM2024_TaOx', 'IEDM2024_HfOx']
    results_comparison = {}
    
    for device_model in device_models:
        print(f"Testing {device_model}...")
        
        crossbar = mn.CrossbarArray(
            rows=128,
            cols=128,
            device_model=device_model
        )
        
        mapped_model = mn.map_to_crossbar(model, crossbar)
        results = mn.simulate(mapped_model, test_data, max_batches=5)
        
        results_comparison[device_model] = {
            'power_mw': results.power_mw,
            'latency_us': results.latency_us,
            'accuracy': results.accuracy,
            'area_mm2': results.area_mm2
        }
    
    # Print comparison
    print("\nComparison Results:")
    print(f"{'Metric':<15} {'TaOx':<12} {'HfOx':<12}")
    print("-" * 40)
    
    for metric in ['power_mw', 'latency_us', 'accuracy', 'area_mm2']:
        taox_val = results_comparison['IEDM2024_TaOx'][metric]
        hfox_val = results_comparison['IEDM2024_HfOx'][metric]
        print(f"{metric:<15} {taox_val:<12.3f} {hfox_val:<12.3f}")
    
    print()


def demonstrate_design_space_exploration():
    """Demonstrate design space exploration capabilities."""
    print("=== Design Space Exploration ===")
    
    model = create_simple_mlp()
    
    # Create explorer
    explorer = mn.DesignSpaceExplorer(
        model=model,
        dataset=None,  # Will use synthetic data
        metrics=['power', 'latency', 'accuracy', 'area']
    )
    
    # Define parameter space to explore
    param_space = {
        'tile_size': [64, 128, 256],
        'device_technology': ['IEDM2024_TaOx', 'IEDM2024_HfOx'],
        'adc_precision': [6, 8],
        'peripheral_optimization': ['baseline', 'low_power']
    }
    
    print("Exploring design space...")
    results_df = explorer.explore(param_space, n_samples=20, parallel=False)
    
    print(f"Explored {len(results_df)} design points")
    print("\nBest designs per metric:")
    
    valid_results = results_df[results_df['power_mw'] != float('inf')]
    if not valid_results.empty:
        best_power = valid_results.loc[valid_results['power_mw'].idxmin()]
        best_latency = valid_results.loc[valid_results['latency_us'].idxmin()]
        best_accuracy = valid_results.loc[valid_results['accuracy'].idxmax()]
        
        print(f"  Lowest Power: {best_power['power_mw']:.2f} mW ({best_power['device_technology']}, {best_power['tile_size']})")
        print(f"  Lowest Latency: {best_latency['latency_us']:.2f} μs ({best_latency['device_technology']}, {best_latency['tile_size']})")
        print(f"  Highest Accuracy: {best_accuracy['accuracy']:.3f} ({best_accuracy['device_technology']}, {best_accuracy['tile_size']})")
    
    # Find Pareto frontier
    pareto_df = explorer.find_pareto_frontier(['power_mw', 'latency_us', 'accuracy'])
    print(f"\nFound {len(pareto_df)} Pareto optimal designs")
    
    print()


def demonstrate_fault_injection():
    """Demonstrate fault injection and reliability analysis.""" 
    print("=== Fault Injection Analysis ===")
    
    model = create_simple_mlp()
    crossbar = mn.CrossbarArray(rows=128, cols=128, device_model='IEDM2024_TaOx')
    mapped_model = mn.map_to_crossbar(model, crossbar)
    
    # Create fault analyzer
    fault_analyzer = mn.FaultAnalyzer(mapped_model)
    
    # Define fault injection campaign
    fault_types = ['stuck_at_on', 'stuck_at_off', 'drift']
    fault_rates = np.logspace(-4, -2, 5)  # 0.01% to 1%
    
    print("Running fault injection campaign...")
    fault_results = fault_analyzer.inject_faults(
        fault_types=fault_types,
        fault_rates=fault_rates,
        n_trials=3  # Reduced for demo
    )
    
    print(f"Completed {len(fault_results)} fault injection experiments")
    
    # Calculate MTBF
    mtbf_estimates = fault_analyzer.calculate_mtbf()
    print("\nMean Time Between Failures (MTBF):")
    
    for fault_type, mtbf_hours in mtbf_estimates.items():
        if mtbf_hours != float('inf'):
            years = mtbf_hours / 8760
            print(f"  {fault_type.replace('_', ' ').title()}: {years:.1f} years")
        else:
            print(f"  {fault_type.replace('_', ' ').title()}: >1000 years")
    
    print()


def demonstrate_rtl_generation():
    """Demonstrate RTL generation capabilities."""
    print("=== RTL Generation Demo ===")
    
    model = create_simple_mlp()
    crossbar = mn.CrossbarArray(rows=64, cols=64, device_model='IEDM2024_TaOx')  
    mapped_model = mn.map_to_crossbar(model, crossbar)
    
    # Create RTL generator
    rtl_gen = mn.RTLGenerator(
        target='ASIC',
        technology='28nm',
        frequency=1000  # 1 GHz
    )
    
    print("Generating Verilog RTL...")
    try:
        verilog_files = rtl_gen.generate_verilog(
            mapped_model,
            output_dir='./rtl_output',
            include_testbench=True
        )
        
        print(f"Generated {len(verilog_files)} RTL files:")
        for file_path in verilog_files:
            print(f"  {file_path}")
        
        # Generate constraints
        constraints = rtl_gen.generate_constraints(
            power_budget=50,  # 50 mW
            area_budget=1.0   # 1 mm²
        )
        
        print(f"\nSynthesis constraints:")
        print(f"  Clock period: {constraints['clock_period_ns']:.2f} ns")
        print(f"  Max power: {constraints['max_power_mw']} mW")
        print(f"  Max area: {constraints['max_area_mm2']} mm²")
        
    except Exception as e:
        print(f"RTL generation error: {e}")
    
    print()


def main():
    """Run all demonstration examples."""
    print("Memristor Neural Network Simulator - Examples\n")
    
    # Run demonstrations
    demonstrate_basic_simulation()
    demonstrate_device_comparison()
    demonstrate_design_space_exploration()
    demonstrate_fault_injection()
    demonstrate_rtl_generation()
    
    print("=== Demo Complete ===")
    print("All examples completed successfully!")
    
    # Show package info
    hw_stats_example = {
        'total_devices': 16384,
        'total_power_mw': 45.2,
        'total_area_mm2': 2.1
    }
    
    print(f"\nExample system specs:")
    print(f"  Devices: {hw_stats_example['total_devices']:,}")
    print(f"  Power: {hw_stats_example['total_power_mw']:.1f} mW")
    print(f"  Area: {hw_stats_example['total_area_mm2']:.2f} mm²")
    print(f"  Efficiency: {hw_stats_example['total_devices']/hw_stats_example['total_power_mw']:.0f} devices/mW")


if __name__ == "__main__":
    main()