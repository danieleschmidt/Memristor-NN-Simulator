"""Map PyTorch neural networks to crossbar arrays."""

from typing import List, Union, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from ..core.crossbar import CrossbarArray
from ..core.device_models import DeviceConfig


class MappedLayer:
    """A neural network layer mapped to crossbar hardware."""
    
    def __init__(self, original_layer: nn.Module, crossbars: List[CrossbarArray]):
        self.original_layer = original_layer
        self.crossbars = crossbars
        self.layer_type = type(original_layer).__name__
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through mapped hardware."""
        if self.layer_type == "Linear":
            return self._forward_linear(x)
        else:
            # For non-linear layers, use original implementation
            return self.original_layer(x)
    
    def _forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for linear layers using crossbar analog computation."""
        batch_size = x.size(0)
        input_size = x.size(-1)
        
        # Convert to numpy for crossbar computation
        x_np = x.detach().cpu().numpy()
        
        outputs = []
        for batch_idx in range(batch_size):
            input_vec = x_np[batch_idx]
            
            # Use first crossbar (could extend to multiple for large layers)
            crossbar = self.crossbars[0]
            output_currents = crossbar.analog_matmul(input_vec)
            outputs.append(output_currents)
        
        # Convert back to torch tensor
        output_tensor = torch.tensor(np.array(outputs), dtype=x.dtype, device=x.device)
        
        # Add bias if present in original layer
        if hasattr(self.original_layer, 'bias') and self.original_layer.bias is not None:
            output_tensor += self.original_layer.bias
            
        return output_tensor


class MappedModel:
    """Complete neural network mapped to crossbar arrays."""
    
    def __init__(self, original_model: nn.Module, mapped_layers: List[MappedLayer]):
        self.original_model = original_model
        self.mapped_layers = mapped_layers
        self.device_count = sum(len(layer.crossbars) for layer in mapped_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through mapped model."""
        current_input = x
        
        # Process through each mapped layer
        for mapped_layer in self.mapped_layers:
            current_input = mapped_layer.forward(current_input)
            
        return current_input
    
    def get_hardware_stats(self) -> Dict[str, Any]:
        """Get hardware utilization statistics."""
        total_devices = 0
        total_power = 0.0
        total_area = 0.0
        
        for layer in self.mapped_layers:
            for crossbar in layer.crossbars:
                total_devices += crossbar.rows * crossbar.cols
                power_stats = crossbar.get_power_consumption()
                area_stats = crossbar.get_area_estimate()
                total_power += power_stats["total_power_mw"]
                total_area += area_stats["total_area_mm2"]
        
        return {
            "total_devices": total_devices,
            "total_power_mw": total_power,
            "total_area_mm2": total_area,
            "crossbar_count": len([cb for layer in self.mapped_layers for cb in layer.crossbars])
        }


def map_to_crossbar(
    model: nn.Module,
    crossbar_template: CrossbarArray,
    tile_size: int = 128,
    device_config: Union[DeviceConfig, None] = None
) -> MappedModel:
    """
    Map a PyTorch model to crossbar arrays.
    
    Args:
        model: PyTorch neural network model
        crossbar_template: Template crossbar for configuration
        tile_size: Maximum tile size for large layers
        device_config: Device configuration for all crossbars
        
    Returns:
        MappedModel with hardware mapping
    """
    mapped_layers = []
    
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            mapped_layer = _map_linear_layer(layer, crossbar_template, tile_size, device_config)
            mapped_layers.append(mapped_layer)
        elif isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
            # Non-linear activations use original implementation
            mapped_layer = MappedLayer(layer, [])
            mapped_layers.append(mapped_layer)
    
    return MappedModel(model, mapped_layers)


def _map_linear_layer(
    layer: nn.Linear,
    template: CrossbarArray,
    tile_size: int,
    device_config: Union[DeviceConfig, None]
) -> MappedLayer:
    """Map a linear layer to one or more crossbar arrays."""
    weight_matrix = layer.weight.detach().cpu().numpy()
    input_size, output_size = weight_matrix.shape[1], weight_matrix.shape[0]
    
    # Determine tiling strategy
    tiles = _calculate_tiling(input_size, output_size, tile_size)
    crossbars = []
    
    for tile_info in tiles:
        row_start, row_end, col_start, col_end = tile_info
        tile_rows = row_end - row_start
        tile_cols = col_end - col_start
        
        # Create crossbar for this tile
        crossbar = CrossbarArray(
            rows=tile_rows,
            cols=tile_cols,
            device_model=template.device_model,
            tile_size=tile_size,
            config=device_config or template.config
        )
        
        # Program weights for this tile
        tile_weights = weight_matrix[row_start:row_end, col_start:col_end]
        crossbar.program_weights(tile_weights.T)  # Transpose for crossbar orientation
        
        crossbars.append(crossbar)
    
    return MappedLayer(layer, crossbars)


def _calculate_tiling(input_size: int, output_size: int, tile_size: int) -> List[tuple]:
    """Calculate optimal tiling strategy for large layers."""
    tiles = []
    
    # Simple row-major tiling
    for row_start in range(0, output_size, tile_size):
        row_end = min(row_start + tile_size, output_size)
        
        for col_start in range(0, input_size, tile_size):
            col_end = min(col_start + tile_size, input_size)
            
            tiles.append((row_start, row_end, col_start, col_end))
    
    return tiles


def estimate_mapping_efficiency(model: nn.Module, crossbar_size: int = 128) -> Dict[str, float]:
    """Estimate hardware mapping efficiency for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    
    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_params += module.weight.numel()
            if module.bias is not None:
                linear_params += module.bias.numel()
    
    # Estimate crossbar utilization
    crossbar_devices = crossbar_size ** 2
    estimated_crossbars = np.ceil(linear_params / crossbar_devices)
    utilization = linear_params / (estimated_crossbars * crossbar_devices)
    
    return {
        "total_parameters": total_params,
        "mappable_parameters": linear_params,
        "mapping_coverage": linear_params / total_params,
        "estimated_crossbars": int(estimated_crossbars),
        "crossbar_utilization": utilization
    }