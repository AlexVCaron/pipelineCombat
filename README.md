# Pipeline Combat 🧠

A comprehensive Python toolkit for neuroimaging data processing and statistical harmonization across different scanners and acquisition sites.

## Features

- **Data Harmonization**: Statistical harmonization using neuroCombat to remove scanner/site effects
- **Diffusion Imaging**: Processing and analysis of diffusion MRI data using DIPY
- **Statistical Analysis**: Comprehensive statistical modeling with statsmodels
- **Probabilistic Models**: Bayesian networks and causal inference with pgmpy
- **GitHub Codespaces Ready**: Pre-configured development environment

## Dependencies

This project uses the following key libraries:

- **[dipy](https://dipy.org/)**: Diffusion imaging in Python for processing diffusion MRI
- **[neuroCombat](https://github.com/Jfortin1/neuroCombat_python)**: Statistical harmonization of neuroimaging data
- **[pgmpy](https://pgmpy.org/)**: Probabilistic graphical models for Bayesian networks
- **[statsmodels](https://www.statsmodels.org/)**: Statistical modeling and econometrics

## Quick Start

### Option 1: GitHub Codespaces (Recommended)

1. Click "Code" → "Codespaces" → "Create codespace on main"
2. The environment will be automatically set up with all dependencies
3. Run the example: `python examples/demo.py`

### Option 2: Local Development

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd pipelineCombat
   uv sync
   ```

3. **Activate environment and run example**:
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   python examples/demo.py
   ```

## Usage Examples

### Data Harmonization

```python
from pipelinecombat.harmonization import harmonize_data
import pandas as pd

# Load your data
data = ...  # Your neuroimaging data (subjects x features)
covars = pd.DataFrame({
    'site': ['site1', 'site2', ...],
    'age': [25, 30, ...],
    'sex': ['M', 'F', ...]
})

# Harmonize across sites
harmonized_data, model = harmonize_data(
    data, covars, 
    batch_col='site', 
    smooth_terms=['age']
)
```

### Diffusion Imaging Analysis

```python
from pipelinecombat.diffusion import DiffusionProcessor

# Initialize processor
processor = DiffusionProcessor()

# Load DWI data
processor.load_data('dwi.nii.gz', 'bvals', 'bvecs')

# Create brain mask
mask = processor.create_mask()

# Fit DTI model and compute metrics
fa, md, ad, rd = processor.fit_dti()

# Save results
metrics = {'fa': fa, 'md': md, 'ad': ad, 'rd': rd}
processor.save_metrics(metrics, 'output_dir')
```

### Statistical Analysis

```python
from pipelinecombat.statistics import NeuroStatAnalyzer

analyzer = NeuroStatAnalyzer()

# Correlation with behavioral scores
corr_results = analyzer.correlation_analysis(
    brain_data, behavioral_scores, method='pearson'
)

# Group comparisons
anova_results = analyzer.anova_analysis(brain_data, groups)

# Multiple comparisons correction
reject, pvals_corrected, _, _ = analyzer.multiple_comparisons_correction(
    anova_results['pvalues'], method='fdr_bh'
)
```

## Project Structure

```
pipelineCombat/
├── src/pipelinecombat/          # Main package
│   ├── __init__.py              # Package initialization
│   ├── harmonization.py         # neuroCombat harmonization
│   ├── diffusion.py             # DIPY diffusion processing
│   └── statistics.py            # statsmodels statistical analysis
├── examples/
│   └── demo.py                  # Complete usage example
├── .devcontainer/               # GitHub Codespaces configuration
├── .github/
│   └── copilot-instructions.md  # Copilot customization
└── pyproject.toml               # Project configuration
```

## Development

This project is configured for development in GitHub Codespaces or VS Code with the following extensions:

- Python
- Jupyter
- GitHub Copilot
- Black Formatter

The development environment includes:
- Python 3.12+
- uv package manager
- Pre-configured debugging
- Linting and formatting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest` (when tests are added)
5. Submit a pull request

## License

This project is open source. Please check the LICENSE file for details.

## Acknowledgments

- **DIPY**: For diffusion MRI processing capabilities
- **neuroCombat**: For statistical harmonization methods
- **pgmpy**: For probabilistic graphical models
- **statsmodels**: For statistical analysis tools

## Citation

If you use this toolkit in your research, please cite the relevant underlying packages:

- DIPY: Garyfallidis et al., "Dipy, a library for the analysis of diffusion MRI data", Frontiers in Neuroinformatics, 2014
- neuroCombat: Fortin et al., "Harmonization of cortical thickness measurements across scanners and sites", NeuroImage, 2018
- pgmpy: Ankan & Panda, "pgmpy: Probabilistic Graphical Models using Python", JMLR, 2015
- statsmodels: Seabold & Perktold, "statsmodels: Econometric and statistical modeling with python", 2010