# Development Guide

This guide covers development setup, contributing guidelines, and technical details for Pipeline Combat.

## Development Environment

### Quick Setup

The fastest way to start developing is using GitHub Codespaces:

1. Fork the repository on GitHub
2. Open your fork in Codespaces
3. Everything is pre-configured and ready!

### Local Development Setup

#### Prerequisites
- Python 3.12+
- Git
- [uv](https://docs.astral.sh/uv/) package manager

#### Installation
```bash
# Clone your fork
git clone https://github.com/yourusername/pipelineCombat.git
cd pipelineCombat

# Install development dependencies
uv sync --group dev

# Activate environment
source .venv/bin/activate
```

#### Verify Setup
```bash
# Run tests (when available)
python -m pytest

# Run demo
python examples/demo.py

# Run code formatting
./scripts/format_code.sh
```

## Project Structure

```
pipelineCombat/
├── .devcontainer/              # GitHub Codespaces config
├── .github/                    # GitHub workflows and configs
├── .vscode/                    # VS Code settings and tasks
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── scripts/                    # Utility scripts
├── src/pipelinecombat/        # Main package source
│   ├── __init__.py            # Package initialization
│   ├── harmonization.py       # Data harmonization
│   ├── diffusion.py           # Diffusion MRI processing
│   └── statistics.py          # Statistical analysis
├── pyproject.toml             # Project configuration
└── uv.lock                    # Locked dependencies
```

## Code Quality Standards

### Formatting and Linting

The project uses a comprehensive code formatting pipeline:

- **autoflake**: Removes unused imports and variables
- **isort**: Organizes imports
- **pyupgrade**: Upgrades syntax for Python 3.12+
- **black**: Code formatting

#### Automatic Formatting

**VS Code Users:**
- `Ctrl+Shift+Alt+F` - Format all Python code
- `Ctrl+Alt+F` - Format current file

**Command Line:**
```bash
# Format all code
./scripts/format_code.sh

# Format specific file
./scripts/format_file.sh path/to/file.py

# Individual tools
uv run --group dev black src/ examples/
uv run --group dev isort src/ examples/
uv run --group dev autoflake --remove-all-unused-imports --in-place --recursive src/
uv run --group dev pyupgrade --py312-plus src/**/*.py
```

### Code Style Guidelines

- **Line length**: 88 characters (Black default)
- **Python version**: 3.12+ features encouraged
- **Type hints**: Use where helpful
- **Docstrings**: Required for all public functions and classes
- **Import organization**: Follow isort with black profile

### Example Code Style

```python
"""Module docstring describing the module purpose."""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from third_party import some_function


class ExampleClass:
    """Class docstring explaining the class purpose.
    
    Args:
        param1: Description of parameter
        param2: Description of optional parameter
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None) -> None:
        self.param1 = param1
        self.param2 = param2
    
    def public_method(self, data: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Public method with clear documentation.
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of processed data and metadata
            
        Raises:
            ValueError: If data has wrong shape
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D array")
            
        # Processing logic here
        result = self._private_method(data)
        metadata = {"shape": data.shape, "processed": True}
        
        return result, metadata
    
    def _private_method(self, data: np.ndarray) -> np.ndarray:
        """Private method for internal use."""
        return data * 2


def utility_function(input_path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Utility function with type hints and documentation."""
    path = Path(input_path)
    if not path.exists():
        return None
    return pd.read_csv(path)
```

## Testing Guidelines

### Test Structure (Future)

When adding tests, follow this structure:

```
tests/
├── test_harmonization.py
├── test_diffusion.py
├── test_statistics.py
└── test_integration.py
```

### Testing Best Practices

```python
import pytest
import numpy as np
from pipelinecombat.statistics import NeuroStatAnalyzer


def test_correlation_analysis():
    """Test correlation analysis with known data."""
    analyzer = NeuroStatAnalyzer()
    
    # Create test data with known correlation
    n_subjects = 100
    brain_data = np.random.randn(n_subjects, 10)
    behavioral_data = brain_data[:, 0] + np.random.randn(n_subjects) * 0.1
    
    results = analyzer.correlation_analysis(brain_data, behavioral_data)
    
    # Check structure
    assert 'correlations' in results
    assert 'pvalues' in results
    assert len(results['correlations']) == 10
    
    # Check first correlation is high (known relationship)
    assert results['correlations'][0] > 0.8


def test_invalid_input():
    """Test proper error handling."""
    analyzer = NeuroStatAnalyzer()
    
    # Mismatched shapes should raise ValueError
    with pytest.raises(ValueError):
        analyzer.correlation_analysis(
            np.random.randn(100, 10),  # 100 subjects
            np.random.randn(50)        # 50 subjects - mismatch!
        )
```

## Contributing Workflow

### 1. Setup Development Environment

```bash
# Fork and clone
git clone https://github.com/yourusername/pipelineCombat.git
cd pipelineCombat

# Install development dependencies
uv sync --group dev

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow code style guidelines
- Add appropriate documentation
- Include examples if adding new features
- Update relevant documentation files

### 3. Test Changes

```bash
# Format code
./scripts/format_code.sh

# Run demo to verify nothing breaks
python examples/demo.py

# Add tests for new functionality (future)
```

### 4. Commit and Push

```bash
# Commit with descriptive message
git add .
git commit -m "feat: add new harmonization method

- Implement robust harmonization for small samples
- Add validation for input data
- Update documentation with examples"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create Pull Request

- Create PR from your fork to the main repository
- Include clear description of changes
- Reference any related issues
- Ensure CI passes

## Architecture Overview

### Module Dependencies

```
pipelinecombat/
├── __init__.py           # Package entry point
├── harmonization.py      # Uses: neuroHarmonize, pandas, numpy
├── diffusion.py         # Uses: dipy, nibabel, numpy
└── statistics.py        # Uses: statsmodels, scipy, numpy
```

### Design Principles

1. **Modular Design**: Each module handles a specific domain
2. **Minimal Dependencies**: Only include necessary packages
3. **Clear APIs**: Simple, consistent function signatures
4. **Error Handling**: Graceful handling of common errors
5. **Documentation**: Comprehensive docs for users and developers

### Adding New Features

#### New Analysis Method

1. Add to appropriate module (harmonization, diffusion, statistics)
2. Follow existing patterns for function signatures
3. Include comprehensive docstring
4. Add example to demo.py
5. Update user guide documentation

#### New Module

1. Create new Python file in `src/pipelinecombat/`
2. Add imports to `__init__.py`
3. Follow naming conventions
4. Include module docstring
5. Add documentation section

## VS Code Development

### Recommended Extensions

The project includes these pre-configured extensions:

- **Python**: Core Python support
- **Black Formatter**: Code formatting
- **isort**: Import organization
- **Ruff**: Fast linting
- **GitHub Copilot**: AI assistance

### Useful Tasks

- **Format Python Code**: `Ctrl+Shift+Alt+F`
- **Format Current File**: `Ctrl+Alt+F`
- **Run Demo**: Use Command Palette → "Tasks: Run Task" → "Run Pipeline Combat Demo"

### Debugging

VS Code is configured for Python debugging:

1. Set breakpoints in code
2. Press `F5` to start debugging
3. Use the integrated terminal for testing

## Documentation

### Building Documentation

The project uses **pdoc** for automatic API documentation generation from docstrings:

#### Generate Documentation

**VS Code:**
- Use Command Palette: "Tasks: Run Task" → "Generate API Documentation"

**Command Line:**
```bash
# Generate HTML and markdown documentation
./scripts/generate_docs.sh
```

This creates:
- **HTML documentation**: `docs/api-html/` (for local viewing)
- **Markdown index**: `docs/api-reference.md` (for GitHub)

#### Viewing Documentation

```bash
# Open HTML documentation in browser
open docs/api-html/index.html  # macOS
xdg-open docs/api-html/index.html  # Linux
```

### Documentation Workflow

The project uses a hybrid approach:

1. **Manual documentation**: User guides, getting started, development guides
2. **Auto-generated API docs**: Generated from Python docstrings using pdoc

#### Updating API Documentation

API documentation is automatically generated from docstrings. To update:

1. **Improve docstrings** in the source code
2. **Run documentation generation**: `./scripts/generate_docs.sh`
3. **Review HTML output** for accuracy
4. **Commit changes** (generated files are gitignored)

#### Example Good Docstring

```python
def harmonize_data(
    data: np.ndarray, 
    covars: pd.DataFrame, 
    batch_col: str = "SITE"
) -> tuple[np.ndarray, dict]:
    """Harmonize neuroimaging data using neuroCombat.
    
    This function removes scanner/site effects from neuroimaging data
    while preserving biological effects of interest.
    
    Args:
        data: Neuroimaging data matrix (n_subjects × n_features)
        covars: DataFrame with covariates including batch information
        batch_col: Column name indicating batches/sites
        
    Returns:
        Tuple of (harmonized_data, harmonization_model)
        
    Raises:
        ValueError: If data and covars have mismatched subjects
        ImportError: If neuroHarmonize is not installed
        
    Example:
        ```python
        harmonized_data, model = harmonize_data(
            data, covars, batch_col='site'
        )
        ```
    """
    # Implementation here
    pass
```

### Documentation Guidelines

- **User-focused**: Write for researchers, not developers
- **Examples**: Include practical code examples
- **Clear structure**: Use consistent headings and formatting
- **Cross-references**: Link between related sections

### Adding Documentation

1. **User Guide**: Add to `docs/user-guide.md`
2. **API Reference**: Add to `docs/api-reference.md`
3. **Getting Started**: Update `docs/getting-started.md` if needed
4. **Changelog**: Add technical details to `CHANGELOG.md`

## Release Process (Future)

### Version Management

The project follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Update documentation
5. Create GitHub release
6. Publish to PyPI (future)

## Getting Help

### Development Questions

- Check existing documentation first
- Review similar patterns in codebase
- Ask in GitHub Discussions
- Open an issue for bugs

### Useful Resources

- **neuroCombat**: [GitHub](https://github.com/Jfortin1/ComBatHarmonization)
- **DIPY**: [Documentation](https://dipy.org/)
- **pgmpy**: [Documentation](https://pgmpy.org/)
- **statsmodels**: [Documentation](https://www.statsmodels.org/)
- **uv**: [Documentation](https://docs.astral.sh/uv/)

---

**Ready to contribute?** Start by running `python examples/demo.py` and exploring the codebase!
