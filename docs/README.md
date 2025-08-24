# Pipeline Combat Documentation

Welcome to Pipeline Combat! This toolkit helps researchers process and harmonize neuroimaging data across different scanners and acquisition sites.

## 📚 Documentation

- **[📖 Getting Started](getting-started.md)** - Installation and first steps
- **[👥 User Guide](user-guide.md)** - Complete usage guide with examples
- **[🔧 API Reference](api-reference.md)** - Auto-generated API documentation
- **[💻 Development Guide](development-guide.md)** - Contributing and development setup

## 🚀 Quick Start

Choose your preferred way to get started:

**Option 1: GitHub Codespaces (Recommended)**
- Click "Code" → "Codespaces" → "Create codespace on main"
- Everything is pre-configured and ready to use
- Run: `python examples/demo.py`

**Option 2: Local Development**
- Install [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Clone the repository and run: `uv sync`
- Run: `python examples/demo.py`

## 🧠 What Can You Do?

### Data Harmonization
Remove scanner and site effects from your neuroimaging data using statistical harmonization methods.

### Diffusion Imaging
Process and analyze diffusion MRI data to compute DTI metrics like FA, MD, AD, and RD.

### Statistical Analysis
Perform comprehensive statistical analysis including correlations, group comparisons, and multiple comparisons correction.

### Probabilistic Models
Build and analyze Bayesian networks and probabilistic graphical models for causal inference.

## 🔧 Key Features

- **Multi-scanner harmonization** using neuroCombat
- **Diffusion imaging processing** with DIPY
- **Statistical modeling** with statsmodels
- **Probabilistic models** with pgmpy
- **GitHub Codespaces ready** for instant development
- **Comprehensive examples** and tutorials

## 🆘 Need Help?

- Check the [User Guide](user-guide.md) for detailed examples
- Run `python examples/demo.py` to see everything in action
- Visit the [API Reference](api-reference.md) for technical details
- See [Development Guide](development-guide.md) to contribute

## 🏗️ Project Structure

```
pipelineCombat/
├── src/pipelinecombat/          # Main package
│   ├── harmonization.py         # neuroCombat harmonization
│   ├── diffusion.py             # DIPY diffusion processing
│   └── statistics.py            # Statistical analysis
├── examples/
│   └── demo.py                  # Complete usage example
├── docs/                        # Documentation (you are here!)
└── scripts/                     # Utility scripts
```

---

**Ready to start?** Head to [Getting Started](getting-started.md)!
