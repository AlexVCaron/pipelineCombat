# Pipeline Combat 🧠

A comprehensive Python toolkit for neuroimaging data processing and statistical harmonization across different scanners and acquisition sites.

## ✨ Features

- **🔗 Data Harmonization**: Remove scanner/site effects using neuroCombat
- **🧠 Diffusion Imaging**: Process and analyze diffusion MRI data with DIPY  
- **📊 Statistical Analysis**: Comprehensive statistical modeling with statsmodels
- **🎲 Probabilistic Models**: Bayesian networks and causal inference with pgmpy
- **☁️ GitHub Codespaces Ready**: Pre-configured development environment

## 🚀 Quick Start

### GitHub Codespaces (Recommended)
1. Click **"Code"** → **"Codespaces"** → **"Create codespace on main"**
2. Wait for automatic setup (2-3 minutes)
3. Run: `python examples/demo.py`

### Local Installation
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd pipelineCombat
uv sync

# Run example
python examples/demo.py
```

## 💡 Quick Example

```python
from pipelinecombat.harmonization import harmonize_data
from pipelinecombat.statistics import NeuroStatAnalyzer
import pandas as pd
import numpy as np

# Harmonize multi-site neuroimaging data
data = np.random.randn(100, 68)  # 100 subjects, 68 brain regions
covars = pd.DataFrame({
    'site': ['site1'] * 50 + ['site2'] * 50,
    'age': np.random.uniform(20, 70, 100)
})

harmonized_data, model = harmonize_data(data, covars, batch_col='site')

# Statistical analysis
analyzer = NeuroStatAnalyzer()
results = analyzer.correlation_analysis(harmonized_data, covars['age'])
```

## 📚 Documentation

**Complete documentation is available in the [`docs/`](docs/) directory:**

- **[📖 Getting Started](docs/getting-started.md)** - Installation and first steps
- **[👥 User Guide](docs/user-guide.md)** - Complete usage guide with examples  
- **[🔧 API Reference](docs/api-reference.md)** - Detailed API documentation
- **[💻 Development Guide](docs/development-guide.md)** - Contributing and development

## 🛠️ Core Dependencies

- **[dipy](https://dipy.org/)** - Diffusion imaging in Python
- **[neuroCombat](https://github.com/Jfortin1/neuroCombat_python)** - Statistical harmonization
- **[pgmpy](https://pgmpy.org/)** - Probabilistic graphical models  
- **[statsmodels](https://www.statsmodels.org/)** - Statistical modeling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following our [Development Guide](docs/development-guide.md)
4. Submit a pull request

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 🙏 Acknowledgments

Pipeline Combat builds on excellent open-source neuroimaging tools. Please cite the underlying packages when using this toolkit in research.

---

**📖 [Start with the Getting Started guide](docs/getting-started.md)** | **💬 Questions? Open an issue!**
