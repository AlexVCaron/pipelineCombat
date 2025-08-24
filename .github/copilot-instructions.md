<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Pipeline Combat Project Instructions

This is a Python project focused on neuroimaging data processing and harmonization using the following key libraries:

- **dipy**: Diffusion imaging in Python for processing and analyzing diffusion MRI data
- **neuroCombat**: Statistical harmonization of neuroimaging data across different scanners/sites
- **pgmpy**: Probabilistic graphical models for Bayesian networks and causal inference
- **statsmodels**: Statistical modeling and econometrics

## Development Guidelines

- Use `uv` as the package manager for dependency management
- Follow PEP 8 style guidelines for Python code
- Include proper docstrings for all functions and classes
- Use type hints where appropriate
- Write unit tests for critical functionality
- Focus on neuroimaging data processing workflows
- Implement robust error handling for file I/O operations
- Consider memory efficiency when working with large neuroimaging datasets

## Project Structure

- Main package code should be in the `src/pipelinecombat/` directory
- Tests should be in the `tests/` directory
- Documentation and examples in appropriate subdirectories
- Use meaningful variable names related to neuroimaging concepts (e.g., `dwi_data`, `fa_map`, `harmonized_values`)

## Neuroimaging Context

When generating code, consider:
- Common neuroimaging file formats (NIfTI, DICOM, etc.)
- Typical preprocessing pipelines
- Statistical harmonization workflows
- Quality control procedures
- Batch processing of multiple subjects/sessions
