"""
Simple standalone version of Memristor NN Simulator for testing core functionality.
This version works without PyTorch to demonstrate the device physics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import time

# Import core device models
from memristor_nn.core.device_models import IEDM2024_TaOx, IEDM2024_HfOx, DeviceConfig

class SimpleCrossbarArray:
    """Simple crossbar array implementation without PyTorch dependencies."""
    
    def __init__(self, rows: int, cols: int, device_model: str = "IEDM2024_TaOx"):
        self.rows = rows
        self.cols = cols
        self.device_model_name = device_model
        
        # Create device model
        if device_model == "IEDM2024_TaOx":
            self.device_model = IEDM2024_TaOx()
        elif device_model == "IEDM2024_HfOx":  
            self.device_model = IEDM2024_HfOx()
        else:
            raise ValueError(f"Unknown device model: {device_model}")
        
        # Initialize device states (random initial states)
        self.device_states = np.random.uniform(0.0, 1.0, (rows, cols))
        
        # Weight matrix representation
        self.weights = np.random.randn(rows, cols) * 0.1
        
    def matrix_vector_multiply(self, input_vector: np.ndarray) -> np.ndarray:
        """Perform matrix-vector multiplication through crossbar physics."""
        if len(input_vector) != self.rows:
            raise ValueError(f"Input vector size {len(input_vector)} doesn't match crossbar rows {self.rows}")
        
        # Convert weights to conductances through device physics
        conductance_matrix = np.zeros((self.rows, self.cols))
        
        for i in range(self.rows):
            for j in range(self.cols):
                # Use device model to get conductance
                voltage = input_vector[i]  # Input voltage
                state = self.device_states[i, j]
                conductance = self.device_model.conductance(voltage, state)
                conductance_matrix[i, j] = conductance
        
        # Analog multiplication: I = G * V
        currents = np.zeros(self.cols)
        for j in range(self.cols):
            for i in range(self.rows):
                currents[j] += conductance_matrix[i, j] * input_vector[i]
        
        return currents
    
    def program_weights(self, target_weights: np.ndarray):
        """Program the crossbar to implement target weights."""
        self.weights = target_weights.copy()
        
        # Convert weights to device states (simplified mapping)
        # Normalize weights to [0, 1] range for device states
        weight_min, weight_max = target_weights.min(), target_weights.max()
        if weight_max > weight_min:
            normalized_weights = (target_weights - weight_min) / (weight_max - weight_min)
        else:
            normalized_weights = np.ones_like(target_weights) * 0.5
        
        self.device_states = normalized_weights
        
        # Simulate programming pulses to update device states
        for i in range(self.rows):
            for j in range(self.cols):
                target_state = normalized_weights[i, j]
                current_state = self.device_states[i, j]
                
                # Apply programming voltage to move toward target state
                if target_state > current_state:
                    voltage = 1.5  # Positive programming
                else:
                    voltage = -1.5  # Negative programming
                
                # Update state using device dynamics
                new_state = self.device_model.update_state(voltage, 1e-6, current_state)  # 1 microsecond pulse
                self.device_states[i, j] = new_state

def create_simple_mlp_weights() -> Dict[str, np.ndarray]:
    """Create weights for a simple 3-layer MLP: 784 -> 128 -> 64 -> 10."""
    weights = {
        'layer1': np.random.randn(784, 128) * 0.1,  # Input to hidden1
        'layer2': np.random.randn(128, 64) * 0.1,   # Hidden1 to hidden2  
        'layer3': np.random.randn(64, 10) * 0.1     # Hidden2 to output
    }
    return weights

def demonstrate_crossbar_operation():
    """Demonstrate basic crossbar array operation."""
    print("🔌 Demonstrating Crossbar Array Operation")
    print("=" * 50)
    
    # Create a small crossbar array
    crossbar = SimpleCrossbarArray(784, 128, "IEDM2024_TaOx")
    print(f"Created {crossbar.rows}x{crossbar.cols} crossbar with {crossbar.device_model_name}")
    
    # Create some test input (simulating image pixels)
    input_vector = np.random.uniform(0.0, 1.0, 784)  # Simulated image
    print(f"Input vector size: {len(input_vector)}")
    
    # Perform computation
    start_time = time.time()
    output = crossbar.matrix_vector_multiply(input_vector)
    computation_time = time.time() - start_time
    
    print(f"Output vector size: {len(output)}")
    print(f"Computation time: {computation_time*1e6:.2f} μs")
    
    # Display some statistics
    print(f"Input range: [{input_vector.min():.3f}, {input_vector.max():.3f}]")
    print(f"Output range: [{output.min():.3e}, {output.max():.3e}] A")
    print(f"Device states range: [{crossbar.device_states.min():.3f}, {crossbar.device_states.max():.3f}]")
    
    return crossbar, input_vector, output

def demonstrate_device_variations():
    """Demonstrate the effect of device variations on computation."""
    print("\n🎲 Demonstrating Device Variations")
    print("=" * 50)
    
    # Create device config with different variation levels
    configs = [
        ("No variations", DeviceConfig(read_noise_sigma=0.0, ron_variation=0.0, roff_variation=0.0)),
        ("Low variations", DeviceConfig(read_noise_sigma=0.02, ron_variation=0.05, roff_variation=0.05)),
        ("High variations", DeviceConfig(read_noise_sigma=0.1, ron_variation=0.2, roff_variation=0.25))
    ]
    
    input_vector = np.random.uniform(0.0, 1.0, 100)
    results = {}
    
    for config_name, config in configs:
        # Create device with specific config
        device = IEDM2024_TaOx(config)
        
        # Test multiple trials
        outputs = []
        for trial in range(10):
            # Single device simulation
            voltage = 1.0
            state = 0.5
            base_conductance = device.conductance(voltage, state)
            varied_conductance = device.add_variations(base_conductance)
            outputs.append(varied_conductance)
        
        outputs = np.array(outputs)
        results[config_name] = {
            'mean': outputs.mean(),
            'std': outputs.std(),
            'cv': outputs.std() / outputs.mean() * 100  # Coefficient of variation
        }
        
        print(f"{config_name}:")
        print(f"  Mean conductance: {results[config_name]['mean']:.2e} S")
        print(f"  Std deviation: {results[config_name]['std']:.2e} S") 
        print(f"  Coefficient of variation: {results[config_name]['cv']:.1f}%")
    
    return results

def demonstrate_neural_mapping():
    """Demonstrate mapping a neural network to crossbar arrays."""
    print("\n🧠 Demonstrating Neural Network Mapping")
    print("=" * 50)
    
    # Create MLP weights
    mlp_weights = create_simple_mlp_weights()
    
    # Create crossbar arrays for each layer
    crossbars = {}
    for layer_name, weights in mlp_weights.items():
        rows, cols = weights.shape
        crossbar = SimpleCrossbarArray(rows, cols, "IEDM2024_TaOx")
        crossbar.program_weights(weights)
        crossbars[layer_name] = crossbar
        print(f"Mapped {layer_name}: {rows}x{cols} -> {crossbar.device_model_name}")
    
    # Simulate forward pass
    input_data = np.random.uniform(0.0, 1.0, 784)  # Simulated MNIST image
    
    # Layer 1: 784 -> 128
    hidden1 = crossbars['layer1'].matrix_vector_multiply(input_data)
    hidden1 = np.maximum(0, hidden1)  # ReLU activation
    
    # Layer 2: 128 -> 64  
    hidden2 = crossbars['layer2'].matrix_vector_multiply(hidden1)
    hidden2 = np.maximum(0, hidden2)  # ReLU activation
    
    # Layer 3: 64 -> 10
    output = crossbars['layer3'].matrix_vector_multiply(hidden2)
    
    print(f"Forward pass completed:")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Hidden1 shape: {hidden1.shape}, active neurons: {(hidden1 > 0).sum()}")
    print(f"  Hidden2 shape: {hidden2.shape}, active neurons: {(hidden2 > 0).sum()}")
    print(f"  Output shape: {output.shape}")
    print(f"  Predicted class: {np.argmax(output)}")
    
    return crossbars, output

def demonstrate_design_space_exploration():
    """Demonstrate design space exploration for different device technologies."""
    print("\n📊 Demonstrating Design Space Exploration")
    print("=" * 50)
    
    device_types = ["IEDM2024_TaOx", "IEDM2024_HfOx"]
    crossbar_sizes = [(64, 64), (128, 128), (256, 256)]
    
    results = {}
    
    for device_type in device_types:
        for size in crossbar_sizes:
            rows, cols = size
            
            # Create crossbar
            crossbar = SimpleCrossbarArray(rows, cols, device_type)
            
            # Benchmark performance
            input_vector = np.random.uniform(0.0, 1.0, rows)
            
            # Time multiple operations
            times = []
            for _ in range(5):
                start = time.time()
                output = crossbar.matrix_vector_multiply(input_vector)
                times.append(time.time() - start)
            
            avg_time = np.mean(times) * 1e6  # Convert to microseconds
            
            # Estimate power (simplified model)
            total_devices = rows * cols
            avg_conductance = 1.0 / ((crossbar.device_model.ron + crossbar.device_model.roff) / 2)
            power_per_device = 1.0 * 1.0 * avg_conductance  # V * I = V * (G * V) 
            total_power_mw = power_per_device * total_devices * 1000  # Convert to mW
            
            config_name = f"{device_type}_{rows}x{cols}"
            results[config_name] = {
                'latency_us': avg_time,
                'power_mw': total_power_mw,
                'area_mm2': rows * cols * 0.001,  # Simplified area model
                'device_count': total_devices
            }
            
            print(f"{config_name}:")
            print(f"  Latency: {avg_time:.2f} μs")
            print(f"  Power: {total_power_mw:.2f} mW")
            print(f"  Area: {results[config_name]['area_mm2']:.2f} mm²")
            print(f"  Devices: {total_devices}")
    
    return results

def create_visualization(results: Dict):
    """Create visualization of design space exploration results."""
    print("\n📈 Creating Design Space Visualization")
    print("=" * 50)
    
    try:
        # Extract data for plotting
        configs = list(results.keys())
        latencies = [results[config]['latency_us'] for config in configs]
        powers = [results[config]['power_mw'] for config in configs]
        areas = [results[config]['area_mm2'] for config in configs]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latency comparison
        ax1.bar(range(len(configs)), latencies)
        ax1.set_title('Computation Latency')
        ax1.set_ylabel('Latency (μs)')
        ax1.set_xticks(range(len(configs)))
        ax1.set_xticklabels(configs, rotation=45, ha='right')
        
        # Power comparison
        ax2.bar(range(len(configs)), powers)
        ax2.set_title('Power Consumption')
        ax2.set_ylabel('Power (mW)')
        ax2.set_xticks(range(len(configs)))
        ax2.set_xticklabels(configs, rotation=45, ha='right')
        
        # Area comparison
        ax3.bar(range(len(configs)), areas)
        ax3.set_title('Chip Area')
        ax3.set_ylabel('Area (mm²)')
        ax3.set_xticks(range(len(configs)))
        ax3.set_xticklabels(configs, rotation=45, ha='right')
        
        # Power vs Latency scatter
        colors = ['red' if 'TaOx' in config else 'blue' for config in configs]
        ax4.scatter(latencies, powers, c=colors, s=100, alpha=0.7)
        ax4.set_xlabel('Latency (μs)')
        ax4.set_ylabel('Power (mW)')
        ax4.set_title('Power vs Latency Trade-off')
        
        # Add legend
        import matplotlib.patches as mpatches
        taox_patch = mpatches.Patch(color='red', label='TaOx devices')
        hfox_patch = mpatches.Patch(color='blue', label='HfOx devices')
        ax4.legend(handles=[taox_patch, hfox_patch])
        
        plt.tight_layout()
        plt.savefig('/root/repo/design_space_exploration.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to design_space_exploration.png")
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Main demonstration function."""
    print("🔌🧠 Memristor Neural Network Simulator - Core Functionality Demo")
    print("=" * 70)
    
    # 1. Basic crossbar operation
    crossbar, input_vec, output = demonstrate_crossbar_operation()
    
    # 2. Device variations analysis
    variation_results = demonstrate_device_variations()
    
    # 3. Neural network mapping
    mapped_crossbars, nn_output = demonstrate_neural_mapping()
    
    # 4. Design space exploration
    design_results = demonstrate_design_space_exploration()
    
    # 5. Create visualization
    create_visualization(design_results)
    
    print("\n🎯 Demo Completed Successfully!")
    print("Key capabilities demonstrated:")
    print("✅ Device-accurate memristor modeling")
    print("✅ Crossbar array physics simulation") 
    print("✅ Neural network to hardware mapping")
    print("✅ Device variation analysis")
    print("✅ Design space exploration")
    print("✅ Power/latency/area trade-off analysis")

if __name__ == "__main__":
    main()