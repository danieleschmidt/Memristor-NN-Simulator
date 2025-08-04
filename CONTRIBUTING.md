# Contributing to Memristor Neural Network Simulator

We welcome contributions from the community! This guide will help you get started with contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Guidelines](#contributing-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Areas of Interest](#areas-of-interest)
7. [Recognition](#recognition)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Basic understanding of neural networks and hardware simulation

### First Time Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/photonic-mlir-synth-bridge.git
   cd photonic-mlir-synth-bridge
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/danieleschmidt/photonic-mlir-synth-bridge.git
   ```

## Development Setup

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,rtl]"

# Install pre-commit hooks
pre-commit install
```

### Using Docker

```bash
# Development environment
docker-compose up memristor-dev

# Run tests
docker-compose run memristor-dev pytest tests/
```

### Verify Installation

```bash
# Run basic tests
python -c "import memristor_nn; print('✓ Package imports successfully')"

# Run example
python examples/basic_usage.py
```

## Contributing Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Automated checks

Run all checks locally:
```bash
# Format code
black memristor_nn tests examples
isort memristor_nn tests examples

# Check style
flake8 memristor_nn tests examples
mypy memristor_nn

# Run pre-commit on all files
pre-commit run --all-files
```

### Testing

We maintain high test coverage (>85%). Before submitting a PR:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=memristor_nn --cov-report=html

# Run specific test categories
pytest tests/test_device_models.py -v
```

### Documentation

- Use Google-style docstrings
- Include type hints for all public functions
- Add examples for complex functionality
- Update README.md if needed

Example docstring:
```python
def simulate(
    mapped_model: MappedModel,
    test_data: torch.Tensor,
    temperature: float = 300.0
) -> SimulationResults:
    """
    Run cycle-accurate simulation of mapped neural network.
    
    Args:
        mapped_model: Neural network mapped to crossbar arrays
        test_data: Test dataset tensor  
        temperature: Operating temperature in Kelvin
        
    Returns:
        Comprehensive simulation results
        
    Raises:
        ValidationError: If inputs are invalid
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> crossbar = CrossbarArray(10, 5)
        >>> mapped = map_to_crossbar(model, crossbar)
        >>> results = simulate(mapped, torch.randn(100, 10))
        >>> print(f"Accuracy: {results.accuracy:.3f}")
    """
```

## Pull Request Process

### Before You Start

1. Check existing issues and PRs to avoid duplicates
2. Create an issue to discuss major changes
3. Ensure you understand the project architecture

### Creating a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our guidelines
3. Add tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass:
   ```bash
   pytest tests/
   pre-commit run --all-files
   ```

6. Commit with descriptive messages:
   ```bash
   git commit -m "feat(core): add new device model for PCMO memristors

   - Implement PCMO device physics model
   - Add calibration data from Nature 2023 paper  
   - Include comprehensive unit tests
   - Update device registry and documentation
   
   Closes #123"
   ```

7. Push and create PR:
   ```bash
   git push origin feature/your-feature-name
   ```

### PR Requirements

- [ ] Descriptive title and detailed description
- [ ] All tests pass
- [ ] Code coverage remains >85%
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts
- [ ] Linked to relevant issue(s)

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all feedback
4. Squash commits if requested
5. Maintainer will merge when ready

## Areas of Interest

We especially welcome contributions in these areas:

### 🔬 Device Models and Calibration
- New memristor device models (PCMO, Ag/Si, etc.)
- Calibration with experimental data
- Noise and variation modeling
- Temperature and aging effects

### 🧮 Mapping Algorithms
- Novel neural network to crossbar mapping strategies
- Optimization for different network architectures
- Fault-tolerant mapping techniques
- Multi-crossbar coordination

### ⚡ RTL Generation and Hardware
- Verilog/SystemVerilog generation improvements
- Chisel backend development
- FPGA-specific optimizations
- ASIC design flow integration

### 🛡️ Fault Tolerance and Reliability
- Advanced fault injection models
- Error correction techniques
- Reliability analysis tools
- Yield optimization strategies

### 📊 Analysis and Visualization
- Enhanced design space exploration
- Interactive visualization tools
- Performance analysis dashboards
- Pareto frontier optimization

### 🚀 Performance and Scalability
- Parallel simulation optimization
- Distributed computing support
- Memory usage optimization
- GPU acceleration

## Recognition

### Contributors

All contributors are recognized in:
- GitHub contributors list
- CONTRIBUTORS.md file
- Annual acknowledgments in releases
- Conference paper acknowledgments (when applicable)

### Types of Contributions

We value all types of contributions:
- 🐛 **Bug fixes**: Help improve reliability
- ✨ **New features**: Expand functionality
- 📚 **Documentation**: Improve usability
- 🧪 **Testing**: Increase coverage and reliability
- 🎨 **Examples**: Help users learn
- 💡 **Ideas**: Suggest improvements
- 🔍 **Issue reporting**: Help identify problems

### Maintainer Recognition

Outstanding contributors may be invited to become maintainers with:
- Commit access
- Issue triage permissions
- PR review responsibilities
- Input on project direction

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Email**: daniel@terragonlabs.com for sensitive topics

### Development Questions

If you're stuck:
1. Check existing documentation and examples
2. Search closed issues and PRs
3. Ask in GitHub Discussions
4. Reach out to maintainers

### Research Collaboration

For academic collaborations or research partnerships:
- Email: daniel@terragonlabs.com
- Include: affiliation, research interests, proposed collaboration

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project. See [LICENSE](LICENSE) for details.

## Thank You!

Thank you for your interest in contributing to memristor-nn-simulator! Your contributions help advance the field of memristive computing and neuromorphic hardware.

Together, we're building the future of neural hardware acceleration! 🚀