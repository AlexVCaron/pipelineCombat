# Getting Started

This guide will help you set up Pipeline Combat and run your first analysis.

## Prerequisites

- Python 3.12 or higher
- Git (for cloning the repository)

## Installation Options

### Option 1: GitHub Codespaces (Recommended)

The easiest way to get started is using GitHub Codespaces, which provides a pre-configured development environment in the cloud.

1. Navigate to the [Pipeline Combat repository](https://github.com/AlexVCaron/pipelineCombat)
2. Click the green **"Code"** button
3. Select **"Codespaces"**
4. Click **"Create codespace on main"**

The environment will automatically set up with all dependencies installed. This takes about 2-3 minutes.

Once ready, you can immediately run:
```bash
python examples/demo.py
```

### Option 2: Local Development

If you prefer to work locally, follow these steps:

#### Step 1: Install uv (Python Package Manager)

Pipeline Combat uses `uv` for fast dependency management:

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.sh | iex"
```

#### Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/AlexVCaron/pipelineCombat.git
cd pipelineCombat

# Install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step 3: Verify Installation

```bash
# Run the demo to verify everything works
python examples/demo.py
```

## First Steps

### 1. Run the Demo

The demo showcases all major features of Pipeline Combat:

```bash
python examples/demo.py
```

You should see output like:
```
🧠 Pipeline Combat Example
==================================================
✅ All modules imported successfully!

📊 Example 1: Data Harmonization (Concept Demo)
Created data: 50 subjects, 20 features
Sites: ['Site_0' 'Site_1' 'Site_2']
...
```

### 2. Understand the Project Structure

```
pipelineCombat/
├── src/pipelinecombat/          # Main package
│   ├── harmonization.py         # Data harmonization tools
│   ├── diffusion.py             # Diffusion MRI processing
│   └── statistics.py            # Statistical analysis
├── examples/
│   └── demo.py                  # Complete example
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
└── pyproject.toml              # Project configuration
```

### 3. Try the Interactive Python Session

Open a Python interpreter and try:

```python
# Import the main modules
from pipelinecombat.statistics import NeuroStatAnalyzer
from pipelinecombat.harmonization import create_example_data
from pipelinecombat.diffusion import create_example_dwi_data

# Create some example data
data, covars = create_example_data(n_subjects=30, n_features=10, n_sites=2)
print(f"Created data with shape: {data.shape}")

# Initialize statistical analyzer
analyzer = NeuroStatAnalyzer()
print("Statistical analyzer ready!")
```

## Available Commands

Once installed, you have access to these commands:

```bash
# Run the package directly
uv run pipelinecombat

# Run the demo
uv run python examples/demo.py

# Add new dependencies (if needed)
uv add package_name

# Sync/update dependencies
uv sync

# Format code (for developers)
./scripts/format_code.sh
```

## VS Code Integration

If you're using VS Code, the project includes pre-configured settings for optimal development:

- Python language support with type checking
- Jupyter notebook support
- GitHub Copilot integration
- Code formatting tools (Black, isort, autoflake, pyupgrade)
- Keyboard shortcuts:
  - `Ctrl+Shift+Alt+F` - Format all Python code
  - `Ctrl+Alt+F` - Format current file

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# If you see import errors, make sure dependencies are installed
uv sync
```

**Python Version Issues**
```bash
# Check your Python version
python --version
# Should be 3.12 or higher
```

**Permission Issues (macOS/Linux)**
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Getting Help

- Check the [User Guide](user-guide.md) for detailed examples
- Review the [API Reference](api-reference.md) for technical details
- Run `python examples/demo.py` to see if everything works
- The [CHANGELOG.md](../CHANGELOG.md) contains detailed technical information

## Next Steps

Now that you have Pipeline Combat installed and running:

1. **Read the [User Guide](user-guide.md)** to learn about the main features
2. **Explore the [API Reference](api-reference.md)** for detailed usage
3. **Check the [Development Guide](development-guide.md)** if you want to contribute

---

**Ready to dive deeper?** Continue to the [User Guide](user-guide.md)!
