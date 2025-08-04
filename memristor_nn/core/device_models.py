"""Memristor device models with physics-based implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field


class DeviceConfig(BaseModel):
    """Configuration for memristor device variations and noise models."""
    
    read_noise_sigma: float = Field(0.05, description="Read noise standard deviation")
    ron_variation: float = Field(0.15, description="Ron device-to-device variation")
    roff_variation: float = Field(0.20, description="Roff device-to-device variation")
    drift_coefficient: float = Field(0.1, description="Temporal drift coefficient")
    stuck_at_rate: float = Field(0.001, description="Stuck-at fault rate")
    temp_coefficient: float = Field(0.002, description="Temperature coefficient per Kelvin")
    temperature: float = Field(300.0, description="Operating temperature in Kelvin")


class DeviceModel(ABC):
    """Abstract base class for memristor device models."""
    
    def __init__(self, config: Optional[DeviceConfig] = None):
        self.config = config or DeviceConfig()
        self.ron = 1e4  # Low resistance state (Ohms)
        self.roff = 1e6  # High resistance state (Ohms)
        self.name = "GenericMemristor"
    
    @abstractmethod
    def conductance(self, voltage: float, state: float) -> float:
        """Calculate conductance given voltage and internal state."""
        pass
    
    @abstractmethod
    def update_state(self, voltage: float, time: float, current_state: float) -> float:
        """Update internal state based on applied voltage and time."""
        pass
    
    def add_variations(self, base_conductance: float) -> float:
        """Add device-to-device and cycle-to-cycle variations."""
        # Device-to-device variations
        ron_var = np.random.normal(1.0, self.config.ron_variation)
        roff_var = np.random.normal(1.0, self.config.roff_variation) 
        
        # Cycle-to-cycle read noise
        read_noise = np.random.normal(1.0, self.config.read_noise_sigma)
        
        # Temperature effects
        temp_factor = 1 + self.config.temp_coefficient * (self.config.temperature - 300)
        
        return base_conductance * ron_var * roff_var * read_noise * temp_factor
    
    def get_resistance(self, voltage: float, state: float) -> float:
        """Get resistance with variations included."""
        conductance = self.conductance(voltage, state)
        varied_conductance = self.add_variations(conductance)
        return 1.0 / max(varied_conductance, 1e-12)  # Avoid division by zero


class IEDM2024_TaOx(DeviceModel):
    """TaOx memristor model calibrated with IEDM 2024 data."""
    
    def __init__(self, config: Optional[DeviceConfig] = None):
        super().__init__(config)
        self.ron = 1e4  # 10 kOhm
        self.roff = 1e6  # 1 MOhm  
        self.name = "IEDM2024_TaOx"
        self.switching_voltage = 1.2  # Volts
        self.nonlinearity = 2.5
    
    def conductance(self, voltage: float, state: float) -> float:
        """TaOx I-V characteristics with nonlinear switching."""
        g_on = 1.0 / self.ron
        g_off = 1.0 / self.roff
        
        # Sigmoid switching function
        switching_factor = 1.0 / (1.0 + np.exp(-self.nonlinearity * (abs(voltage) - self.switching_voltage)))
        
        # State-dependent conductance (state between 0 and 1)
        base_conductance = g_off + state * (g_on - g_off)
        
        # Voltage-dependent modulation
        return base_conductance * (1.0 + 0.1 * switching_factor * np.sign(voltage))
    
    def update_state(self, voltage: float, time: float, current_state: float) -> float:
        """Update state based on voltage-driven switching dynamics."""
        if abs(voltage) < 0.1:  # No switching for small voltages
            return current_state
            
        # Switching rate depends on voltage magnitude
        switching_rate = 1e6 * (abs(voltage) / self.switching_voltage) ** 2
        
        # Target state based on voltage polarity
        target_state = 1.0 if voltage > 0 else 0.0
        
        # Exponential approach to target state
        time_constant = 1.0 / switching_rate
        new_state = target_state + (current_state - target_state) * np.exp(-time / time_constant)
        
        return np.clip(new_state, 0.0, 1.0)


class IEDM2024_HfOx(DeviceModel):
    """HfOx memristor model calibrated with IEDM 2024 data."""
    
    def __init__(self, config: Optional[DeviceConfig] = None):
        super().__init__(config)
        self.ron = 1e3  # 1 kOhm
        self.roff = 1e5  # 100 kOhm
        self.name = "IEDM2024_HfOx"
        self.switching_voltage = 0.8  # Volts
        self.nonlinearity = 3.0
    
    def conductance(self, voltage: float, state: float) -> float:
        """HfOx I-V characteristics with fast switching."""
        g_on = 1.0 / self.ron
        g_off = 1.0 / self.roff
        
        # Sharp switching characteristics
        switching_factor = np.tanh(self.nonlinearity * (abs(voltage) - self.switching_voltage))
        switching_factor = max(0, switching_factor)
        
        base_conductance = g_off + state * (g_on - g_off)
        return base_conductance * (1.0 + 0.2 * switching_factor)
    
    def update_state(self, voltage: float, time: float, current_state: float) -> float:
        """Fast HfOx switching dynamics."""
        if abs(voltage) < 0.05:
            return current_state
            
        # Faster switching than TaOx
        switching_rate = 5e6 * (abs(voltage) / self.switching_voltage) ** 1.5
        target_state = 1.0 if voltage > 0 else 0.0
        
        time_constant = 1.0 / switching_rate
        new_state = target_state + (current_state - target_state) * np.exp(-time / time_constant)
        
        return np.clip(new_state, 0.0, 1.0)


# Device registry for easy access
DEVICE_REGISTRY: Dict[str, type] = {
    "IEDM2024_TaOx": IEDM2024_TaOx,
    "IEDM2024_HfOx": IEDM2024_HfOx,
}


def register_device(name: str, device_class: type) -> None:
    """Register a custom device model."""
    DEVICE_REGISTRY[name] = device_class


def create_device(name: str, config: Optional[DeviceConfig] = None) -> DeviceModel:
    """Create a device instance from the registry."""
    if name not in DEVICE_REGISTRY:
        raise ValueError(f"Unknown device model: {name}. Available: {list(DEVICE_REGISTRY.keys())}")
    
    return DEVICE_REGISTRY[name](config)