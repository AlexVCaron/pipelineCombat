# User Guide

This guide covers the main features of Pipeline Combat with practical examples and use cases.

## Overview

Pipeline Combat provides four main areas of functionality:

1. **[Data Harmonization](#data-harmonization)** - Remove scanner/site effects
2. **[Diffusion Imaging](#diffusion-imaging)** - Process diffusion MRI data
3. **[Statistical Analysis](#statistical-analysis)** - Comprehensive statistical tools
4. **[Probabilistic Models](#probabilistic-models)** - Bayesian networks and causal inference

## Data Harmonization

Harmonization removes systematic differences between scanning sites or scanners while preserving biological effects.

### Basic Usage

```python
from pipelinecombat.harmonization import harmonize_data
import pandas as pd
import numpy as np

# Your neuroimaging data (subjects × features)
# This could be cortical thickness, FA values, etc.
data = np.random.randn(100, 68)  # 100 subjects, 68 brain regions

# Covariates including site/scanner information
covars = pd.DataFrame({
    'site': ['site1'] * 50 + ['site2'] * 50,  # Scanner sites
    'age': np.random.uniform(20, 70, 100),    # Age
    'sex': np.random.choice(['M', 'F'], 100)  # Sex
})

# Harmonize across sites
harmonized_data, model = harmonize_data(
    data, 
    covars, 
    batch_col='site',           # Column indicating batches (sites)
    smooth_terms=['age']        # Variables to preserve as smooth terms
)
```

### What Gets Harmonized?

- **Cortical thickness** measurements
- **DTI metrics** (FA, MD, AD, RD)
- **Volumetric** measurements
- **Connectivity** measures
- Any **continuous neuroimaging** metric

### Important Notes

- Harmonization should be done **before** statistical analysis
- Include relevant **covariates** (age, sex, etc.)
- The **same harmonization model** should be applied to new data

## Diffusion Imaging

Process diffusion MRI data to extract meaningful metrics like fractional anisotropy (FA).

### Processing Pipeline

```python
from pipelinecombat.diffusion import DiffusionProcessor

# Initialize the processor
processor = DiffusionProcessor()

# Load your diffusion data
processor.load_data(
    dwi_path='dwi.nii.gz',      # 4D diffusion-weighted images
    bvals_path='dwi.bval',      # b-values file
    bvecs_path='dwi.bvec'       # gradient directions file
)

# Create brain mask (optional, but recommended)
mask = processor.create_mask()

# Fit DTI model and extract metrics
fa, md, ad, rd = processor.fit_dti()

# Save results
metrics = {
    'fa': fa,    # Fractional Anisotropy
    'md': md,    # Mean Diffusivity  
    'ad': ad,    # Axial Diffusivity
    'rd': rd     # Radial Diffusivity
}
processor.save_metrics(metrics, 'output_directory/')
```

### DTI Metrics Explained

- **FA (Fractional Anisotropy)**: Measure of white matter integrity (0-1)
- **MD (Mean Diffusivity)**: Average rate of diffusion
- **AD (Axial Diffusivity)**: Diffusion along main fiber direction
- **RD (Radial Diffusivity)**: Diffusion perpendicular to main fiber direction

### Working with Synthetic Data

For testing and development:

```python
from pipelinecombat.diffusion import create_example_dwi_data

# Create synthetic diffusion data
dwi_data, bvals, bvecs = create_example_dwi_data(
    shape=(64, 64, 30),     # Image dimensions
    n_directions=32         # Number of gradient directions
)
```

## Statistical Analysis

Comprehensive statistical tools for neuroimaging data analysis.

### Initialization

```python
from pipelinecombat.statistics import NeuroStatAnalyzer

analyzer = NeuroStatAnalyzer()
```

### Correlation Analysis

Find relationships between brain measures and behavioral scores:

```python
# Brain data: subjects × regions
brain_data = np.random.randn(80, 68)  # 80 subjects, 68 regions

# Behavioral measure (e.g., cognitive score)
behavioral_scores = np.random.randn(80)

# Perform correlation analysis
results = analyzer.correlation_analysis(
    brain_data, 
    behavioral_scores, 
    method='pearson'  # or 'spearman'
)

# Results contain:
# - correlations: correlation coefficients
# - pvalues: p-values for each correlation
```

### Group Comparisons

Compare brain measures between groups:

```python
# Group labels (e.g., patients vs controls)
groups = ['control'] * 40 + ['patient'] * 40

# Perform ANOVA
anova_results = analyzer.anova_analysis(brain_data, groups)

# Apply multiple comparisons correction
reject, pvals_corrected, _, _ = analyzer.multiple_comparisons_correction(
    anova_results['pvalues'], 
    method='fdr_bh',    # False Discovery Rate correction
    alpha=0.05
)

# Find significant regions
significant_regions = np.where(reject)[0]
print(f"Found {len(significant_regions)} significant regions")
```

### Multiple Comparisons Correction

When testing many brain regions, correct for multiple comparisons:

```python
# Available methods:
# - 'bonferroni': Conservative correction
# - 'fdr_bh': False Discovery Rate (Benjamini-Hochberg)
# - 'fdr_by': False Discovery Rate (Benjamini-Yekutieli)

reject, pvals_corrected, alpha_sidak, alpha_bonf = analyzer.multiple_comparisons_correction(
    pvalues, 
    method='fdr_bh', 
    alpha=0.05
)
```

### Creating Example Data

For testing and learning:

```python
from pipelinecombat.statistics import create_example_behavioral_data

brain_data, behavioral_scores, covariates = create_example_behavioral_data(
    n_subjects=100
)
```

## Probabilistic Models

Build Bayesian networks and perform causal inference (advanced feature).

### Basic Bayesian Network

```python
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# Define network structure
model = BayesianNetwork([('Age', 'CognitiveScore'), 
                        ('Education', 'CognitiveScore')])

# This functionality uses pgmpy - see pgmpy documentation for details
```

## Practical Workflows

### Complete Multi-Site DTI Analysis

Here's a typical workflow for analyzing DTI data from multiple sites:

```python
import numpy as np
import pandas as pd
from pipelinecombat.diffusion import DiffusionProcessor
from pipelinecombat.harmonization import harmonize_data
from pipelinecombat.statistics import NeuroStatAnalyzer

# 1. Process DTI data from each site
sites = ['site1', 'site2', 'site3']
all_fa_data = []
all_subjects = []

for site in sites:
    processor = DiffusionProcessor()
    
    # Process each subject's DTI data
    for subject in get_subjects_for_site(site):  # Your function
        processor.load_data(f'{site}/{subject}/dwi.nii.gz', ...)
        fa, _, _, _ = processor.fit_dti()
        
        # Extract FA values for regions of interest
        fa_values = extract_roi_values(fa)  # Your function
        
        all_fa_data.append(fa_values)
        all_subjects.append({'subject': subject, 'site': site})

# 2. Combine data
fa_matrix = np.array(all_fa_data)
subject_info = pd.DataFrame(all_subjects)

# 3. Add additional covariates
subject_info['age'] = get_ages(subject_info['subject'])  # Your function
subject_info['sex'] = get_sex(subject_info['subject'])   # Your function

# 4. Harmonize across sites
harmonized_fa, _ = harmonize_data(
    fa_matrix, 
    subject_info, 
    batch_col='site',
    smooth_terms=['age']
)

# 5. Statistical analysis
analyzer = NeuroStatAnalyzer()

# Correlate with age
age_corr = analyzer.correlation_analysis(
    harmonized_fa, 
    subject_info['age']
)

# Multiple comparisons correction
reject, _, _, _ = analyzer.multiple_comparisons_correction(
    age_corr['pvalues'], 
    method='fdr_bh'
)

print(f"Found {np.sum(reject)} regions significantly correlated with age")
```

## Data Formats

### Expected Input Formats

**Neuroimaging Data Matrix:**
- Shape: `(n_subjects, n_features)`
- Features can be: brain regions, voxels, vertices, etc.
- Data type: `numpy.ndarray` or `pandas.DataFrame`

**Covariates DataFrame:**
- Required columns depend on analysis
- For harmonization: batch/site column
- Common columns: age, sex, education, diagnosis
- Data type: `pandas.DataFrame`

**Diffusion Data:**
- DWI images: NIfTI format (`.nii` or `.nii.gz`)
- b-values: Text file with b-values
- b-vectors: Text file with gradient directions

## Best Practices

### Data Quality

1. **Visual inspection** of raw data before processing
2. **Motion correction** for diffusion data
3. **Outlier detection** before harmonization
4. **Check for missing data** and handle appropriately

### Statistical Analysis

1. **Always correct for multiple comparisons** when testing many regions
2. **Include relevant covariates** (age, sex, etc.)
3. **Check assumptions** (normality, equal variance)
4. **Use appropriate effect size measures**

### Harmonization

1. **Apply harmonization before statistical analysis**
2. **Include all relevant covariates**
3. **Validate harmonization effectiveness**
4. **Apply same model to new data**

## Troubleshooting

### Common Issues

**Memory Errors:**
- Process data in smaller batches
- Use memory-efficient data formats
- Consider dimensionality reduction

**Convergence Issues in Harmonization:**
- Check for outliers in data
- Ensure sufficient sample size per site
- Verify covariate specifications

**Statistical Analysis:**
- Check data distributions
- Ensure proper alignment of data arrays
- Verify group sizes are adequate

---

**Next:** Check the [API Reference](api-reference.md) for detailed technical documentation.
