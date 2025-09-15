# Pipeline Combat - AI Coding Agent Instructions

## 🧠 Project Overview
Neuroimaging data processing and harmonization toolkit with modular architecture for multi-site DTI analysis workflows.

**Core Libraries**: `dipy` (diffusion MRI), `neuroHarmonize` (scanner harmonization), `statsmodels` (statistics), `pgmpy` (probabilistic models)

## 🏗️ Architecture & Integration Patterns

### Three-Layer Modular Design
```
pipelinecombat/
├── harmonization.py    # neuroCombat multi-site harmonization
├── diffusion.py       # DIPY diffusion MRI processing
└── statistics.py      # statsmodels statistical analysis
```

**Integration Workflow**: Raw DTI → `DiffusionProcessor` → site harmonization → `NeuroStatAnalyzer` → results

### Key Integration Points
- **Graceful degradation**: All modules check dependency availability (`DIPY_AVAILABLE`, `NEUROCOMBAT_AVAILABLE`) and provide synthetic data alternatives
- **Class-based state management**: `DiffusionProcessor` maintains data/affine/gradient state; `NeuroStatAnalyzer` stores results
- **Consistent data flow**: numpy arrays between modules, pandas DataFrames for covariates
- **Example data generators**: Each module provides `create_example_*` functions for testing/demos

## 🛠️ Critical Developer Workflows

### Package Management (uv-based)
- **Install/sync**: `uv sync` (not pip install)
- **Run scripts**: `uv run python examples/demo.py`
- **Add packages**: `uv add package_name`
- **Dev dependencies**: `uv add --group dev package_name`

### PEP 8 Workflow (Automated)
- **Format current file**: VS Code Task "PEP 8: Format Current File"
- **Format codebase**: `./scripts/format_code.sh`
- **Quick lint**: `./scripts/lint_check.sh`
- **Line length**: 79 chars (strict PEP 8), numpy docstrings, double quotes

### Testing Patterns
- **Run with coverage**: VS Code Task "Run Tests with Coverage"
- **Test structure**: `test_*.py` files, class-based (`TestClassName`)
- **Integration tests**: Use synthetic data generators, not real neuroimaging files
- **Coverage target**: 80% minimum

## 📊 Neuroimaging Data Patterns & Advanced Models

### DTI Processing Chain
```python
# Standard workflow in diffusion.py
processor = DiffusionProcessor()
processor.load_data(dwi_path, bvals_path, bvecs_path)  # NIfTI + gradient files
fa, md, ad, rd = processor.fit_dti()  # Extract DTI metrics
processor.save_metrics({'fa': fa, 'md': md}, 'output_dir/')
```

### Design Matrix Architecture
```python
# Statistical modeling core in model/design.py
design = DesignMatrix(categorical_data, numerical_data, batch_col_index='batch')
model = design.generate()  # Returns Model with X (design matrix) and I (pseudoinverse)

# Key methods for neuroimaging harmonization:
beta, residuals = model.fit(y)  # Least squares fitting
y_standard = model.standard(y, beta, sigma)  # Standardization
y_combat = model.combat(y, beta, gamma, delta, sigma)  # Combat correction
```

### PCA Integration Pattern
```python
# Dimensionality reduction in model/pca.py
pca_model = PCA(numerical_data, n_components=20)
model = pca_model.generate()  # Includes variance-aware standardization
# Component 0 is data mean, components 1+ are PCA directions
transformed_data = []
for sample in original_data:
    beta, _ = model.fit(sample)  # Transform via least squares
    transformed_data.append(beta)
```

### Pipeline Combat Workflow
```python
# Advanced multi-batch harmonization in pipelineCombat.py
designs, models, gamma_star, delta_var_star = pipeline_combat(
    biased_data, covariates,
    batch_col_index='batch', modality_col_index='modality',
    create_pca_block=True, pca_n_components=70,
    batch_links=dependency_matrix  # Bayesian network structure
)
```

### Data Conventions
- **Subject data**: `(n_subjects, n_features)` numpy arrays
- **Brain regions**: Default to 68 (Desikan-Killiany atlas)
- **Site coding**: String identifiers (`'Site_0'`, `'Site_1'`) in pandas DataFrames
- **DTI metrics**: FA/MD/AD/RD as separate numpy arrays, NaN→0 cleanup
- **Design matrices**: Categorical→one-hot encoding, numerical data appended
- **PCA components**: Index 0 = data mean, indices 1+ = principal components

## 🎯 Project-Specific Conventions

### Error Handling Pattern
```python
try:
    from dipy.core.gradients import gradient_table
    DIPY_AVAILABLE = True
except ImportError:
    logger.warning("DIPY not available. Install with: uv add dipy")
    DIPY_AVAILABLE = False
```

### Logging Configuration
- Module-level loggers: `logger = logging.getLogger(__name__)`
- Info-level progress tracking for neuroimaging operations
- Warning for missing optional dependencies

### File Naming & Structure
- **Module imports**: `from pipelinecombat.diffusion import DiffusionProcessor`
- **Scripts location**: `scripts/` for utility scripts, `examples/` for user demos
- **Test isolation**: Mirror `src/pipelinecombat/` structure in `tests/`

## 🧪 Advanced Demo Patterns

### Comprehensive Visualization Architecture (`examples/`)
- **pca_demo.py**: 4-panel PCA analysis with loadings, variance, reconstruction quality
- **pipeline_combat_demo.py**: Full workflow simulation with Bayesian network dependencies
- **dwi_real_data_demo.py**: Real Stanford HARDI dataset analysis with slice-based PCA
- **demo.ipynb**: Interactive notebook combining all modules

### Demo Data Generation Conventions
```python
# Synthetic neuroimaging data with realistic biases
np.random.seed(42)  # Always set for reproducibility
n_regions = 68  # Standard atlas size
site_bias = np.random.normal(0, 0.05, n_regions)  # Scanner effects
fa_values = np.random.beta(2, 3, (n_subjects, n_regions)) * 0.8  # Realistic FA range
```

### Visualization Patterns
```python
# Check availability, degrade gracefully
try:
    import matplotlib.pyplot as plt
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Install with: uv add matplotlib")

# Multi-panel figure standard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Analysis Results", fontsize=16, fontweight="bold")
```

## 🚀 Quick Start Commands
```bash
# Setup & run demo
uv sync && uv run python examples/demo.py

# Development workflow
./scripts/format_code.sh        # Format before commit
uv run pytest --cov            # Run tests
./scripts/lint_check.sh         # Verify PEP 8
```
