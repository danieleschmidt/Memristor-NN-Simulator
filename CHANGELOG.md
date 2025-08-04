# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of memristor neural network simulator
- IEDM 2024 calibrated device models (TaOx, HfOx)
- Neural network to crossbar mapping functionality
- Cycle-accurate simulation engine
- RTL generation for Verilog and Chisel
- Design space exploration tools
- Fault injection and reliability analysis
- Hardware validation against measured silicon data
- Comprehensive error handling and security features
- Performance optimization and caching system
- Parallel simulation capabilities
- Memory optimization utilities
- Docker containerization support
- CI/CD pipeline with GitHub Actions

### Security
- Input validation and sanitization utilities
- Secure file path handling
- Memory usage monitoring and limits  
- Rate limiting for API operations
- Configuration parameter validation

## [0.1.0] - 2025-01-XX

### Added
- Core crossbar array implementation with device models
- Basic neural network mapping capabilities
- Simulation engine with device variations
- RTL generation framework
- Design space exploration tools
- Fault injection analysis
- Comprehensive test suite
- Documentation and examples

### Features
- **Device Models**: Physics-based memristor models calibrated with IEDM 2024 datasets
- **Neural Mapping**: Automatic mapping of PyTorch/TensorFlow models to crossbar arrays
- **Simulation**: Cycle-accurate simulation with noise and variation models
- **RTL Generation**: Automated Verilog/Chisel generation for ASIC/FPGA deployment
- **Design Exploration**: Power/latency/accuracy Pareto frontier analysis
- **Fault Analysis**: Comprehensive stuck-at-fault and drift models
- **Validation**: Hardware validation against measured silicon data

### Supported Devices
- TaOx memristors (IEDM 2024 calibrated)
- HfOx memristors (IEDM 2024 calibrated)
- Generic memristor model framework

### Performance
- Parallel simulation support
- Intelligent caching system
- Memory optimization utilities
- GPU acceleration ready

### Developer Experience
- Comprehensive documentation
- Example notebooks and scripts
- Docker containerization
- Automated testing and CI/CD
- Type hints and code quality tools

[Unreleased]: https://github.com/danieleschmidt/photonic-mlir-synth-bridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/danieleschmidt/photonic-mlir-synth-bridge/releases/tag/v0.1.0