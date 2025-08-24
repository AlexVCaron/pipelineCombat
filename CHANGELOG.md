# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - 2025-08-24

## Documentation System Overhaul

#### Comprehensive Documentation Structure
- **docs/ Directory**: Complete restructuring of project documentation
  - `docs/README.md` - Documentation hub with clear navigation
  - `docs/getting-started.md` - Installation and first steps guide
  - `docs/user-guide.md` - Complete usage guide with examples
  - `docs/api-reference.md` - Auto-generated API documentation
  - `docs/development-guide.md` - Contributing and development setup

#### Automated API Documentation Generation
- **pdoc Integration**: Lightweight, automated API documentation system
  - Added `pdoc` to development dependencies in `pyproject.toml`
  - `scripts/generate_docs.sh` - Automated documentation generation script
  - `scripts/extract_api_docs.py` - Markdown index generation for GitHub compatibility
  - HTML documentation output in `docs/api-html/` for rich local viewing
  - Markdown index at `docs/api-reference.md` for GitHub integration

#### Documentation Automation Features
- **Dual Format Output**: Both HTML and markdown documentation generation
  - HTML docs with search functionality and rich formatting
  - GitHub-compatible markdown with proper linking
  - Automatic docstring extraction from source code
  - Zero-maintenance documentation that stays in sync with code

- **GitHub Integration**: Optimized for repository viewing
  - Markdown files display properly on GitHub
  - Cross-references between documentation files
  - Professional documentation structure for open-source projects
  - Automatic exclusion of generated files via `.gitignore`

#### Documentation Content Migration
- **README.md Simplification**: Streamlined main README to focus on quick start
  - Removed technical details in favor of documentation references
  - Clear navigation to comprehensive docs
  - Emphasis on getting started quickly
  
- **Comprehensive User Guides**: Detailed documentation for all user types
  - Progressive disclosure from quick start to advanced usage
  - Real-world examples and use cases
  - Complete API coverage with examples

## Core Application Development

#### Package Architecture
- **Modular Design**: Organized codebase with clear separation of concerns
  - `harmonization.py` - neuroCombat/neuroHarmonize integration modules
  - `diffusion.py` - DIPY diffusion imaging processing workflows
  - `statistics.py` - statsmodels statistical analysis functions
  - `__init__.py` - Package entry point and public API definition

#### Core Dependencies and Environment
- **Primary Dependencies**: Installed neuroimaging and data science libraries
  - `dipy` - Diffusion imaging processing and analysis
  - `neuroHarmonize>=2.4.5` - Statistical harmonization (neuroCombat implementation)
  - `pgmpy` - Probabilistic graphical models and Bayesian networks
  - `statsmodels` - Statistical modeling and econometrics
  - `numpy`, `pandas`, `scikit-learn` - Core data science libraries

- **Package Manager**: Configured `uv` for modern Python dependency management
  - `pyproject.toml` - Project configuration with proper metadata
  - `uv.lock` - Locked dependencies for reproducible builds
  - Python 3.12+ requirement specification

## Development Environment & Tooling

#### Repository Structure
- **Project Structure**: Established complete Python project structure
  - `src/pipelinecombat/` - Main package directory with modular components
  - `examples/` - Demo scripts and usage examples
  - `.devcontainer/` - GitHub Codespaces configuration
  - `.github/` - GitHub-specific configurations and workflows
  - `.vscode/` - VS Code workspace settings and tasks

#### Development Environment Setup
- **GitHub Codespaces**: Pre-configured development container
  - `devcontainer.json` - Container configuration for cloud development
  - Automatic dependency installation and environment setup
  - VS Code extensions pre-installed

- **VS Code Integration**: Development environment optimization
  - Python language support with type checking
  - Jupyter notebook support for data exploration
  - GitHub Copilot integration for AI-assisted coding
  - Initial task configuration for running demos

#### Enhanced VS Code Integration
- **Extensions** (`.vscode/extensions.json`): Added formatting-specific extensions
  - `ms-python.black-formatter` - Black formatter integration
  - `ms-python.isort` - isort integration  
  - `charliermarsh.ruff` - Fast Python linter
  
- **Settings** (`.vscode/settings.json`): Enhanced Python development environment
  - Black formatter as default Python formatter
  - isort configured with black profile for compatibility
  - autoflake configured to remove unused imports and variables
  - Disabled format-on-save to allow manual control
  
- **Tasks** (`.vscode/tasks.json`): Added automated formatting tasks
  - `Format Python Code` - Formats all Python files in project
  - `Format Current Python File` - Formats only the currently open Python file
  - Both tasks use shell scripts for reliable execution
  
- **Keybindings** (`.vscode/keybindings.json`): Added keyboard shortcuts for quick access
  - `Ctrl+Shift+Alt+F` - Format all Python code in project
  - `Ctrl+Alt+F` - Format current Python file (only active when .py file is open)

#### GitHub Copilot Integration
- **Custom Instructions**: Tailored AI assistance for neuroimaging development
  - `.github/copilot-instructions.md` - Project-specific guidance
  - Neuroimaging domain knowledge integration
  - Development workflow recommendations
  - Code style and best practice guidelines

#### Development Workflow
- **Quick Start Commands**: Streamlined development experience
  - `uv run python examples/demo.py` - Run comprehensive demo
  - `uv sync` - Dependency synchronization
  - `uv add <package>` - Easy dependency management
  - Package execution via `uv run pipelinecombat`

## Code Quality & Standards

#### Code Formatting Pipeline Setup
- **Dependencies**: Added comprehensive Python code formatting tools as development dependencies
  - `autoflake` - Removes unused imports and variables  
  - `isort` - Sorts and organizes imports
  - `pyupgrade` - Upgrades syntax for newer Python versions (3.12+)
  - `black` - Code formatter
  - All tools added to `[dependency-groups].dev` in `pyproject.toml`

#### Tool Configuration
- **pyproject.toml**: Added comprehensive tool configurations for consistent behavior
  - `[tool.isort]` - Configured with black profile, 88 character line length
  - `[tool.black]` - 88 character line length, Python 3.12+ target, proper exclusions
  - Tool configurations ensure all formatters work together without conflicts

#### Automation Scripts
- **scripts/format_code.sh**: Comprehensive formatting script for entire codebase
  - Runs all four tools in optimal sequence: autoflake → isort → pyupgrade → black
  - Provides clear progress feedback with emojis and status messages
  - Executable permissions set automatically
  
- **scripts/format_file.sh**: Single file formatting script
  - Accepts file path as argument with validation
  - Runs same tool sequence on individual files
  - Error handling for non-Python files and missing files
  - Clear success/error reporting

## Quality Assurance & Testing

#### CI/CD Infrastructure
- **GitHub Actions**: Automated testing and deployment pipeline
  - `.github/workflows/ci.yml` - Continuous integration configuration
  - Automated testing on multiple Python versions
  - Code quality checks and validation

#### Testing and Validation
- **Functional Testing**: Verified core functionality across all modules
  - Package installation and import validation
  - Statistical analysis with real example data
  - Diffusion imaging workflows with synthetic data
  - Data harmonization concept verification
  - Cross-module integration testing

## Documentation & Examples

#### Documentation
- **README.md**: Comprehensive project documentation
  - Project overview and objectives
  - Installation and setup instructions
  - Usage examples and API documentation
  - Development guidelines and best practices

- **FORMATTING.md**: Complete usage documentation
  - Tool descriptions and purpose
  - VS Code command instructions with keyboard shortcuts
  - Command line usage examples
  - Configuration details
  - Installation instructions

#### Demo Implementation
- **examples/demo.py**: Complete working example
  - Integrated demonstration of all major components
  - Synthetic data generation for testing
  - Statistical analysis workflows
  - Diffusion imaging processing examples
  - Data harmonization concept demonstrations

## Developer Experience Enhancements

#### Developer Experience Improvements
- **One-command formatting**: Single keyboard shortcut or command formats entire codebase
- **Incremental formatting**: Option to format only current file during development
- **Progress feedback**: Clear visual feedback during formatting operations
- **Error handling**: Proper error reporting and validation in scripts
- **Cross-platform compatibility**: Scripts work in various shell environments
- **IDE integration**: Seamless integration with VS Code editor features

### Technical Specifications
- **Code Style**: Black-compatible configuration across all tools
- **Line Length**: Standardized to 88 characters (Black default)
- **Import Organization**: isort configured with black profile for consistency
- **Shell Compatibility**: Scripts designed for bash/zsh compatibility

## Project Maintenance & Cleanup

#### Documentation Consolidation
- **Single Source of Truth**: Eliminated redundant API documentation files
  - Removed manual `api-reference.md` in favor of auto-generated version
  - Updated all cross-references to point to single API documentation
  - Consolidated documentation generation workflow

#### File Organization & Cleanup
- **Temporary File Removal**: Cleaned up development artifacts
  - Removed `docs/temp/` directory created during testing
  - Eliminated duplicate and backup files
  - Streamlined directory structure for production

#### Configuration Updates
- **Updated Build Scripts**: Aligned all scripts with final file structure
  - Modified `scripts/generate_docs.sh` to output to correct filenames
  - Updated `.gitignore` to exclude proper generated documentation files
  - Synchronized development guide with actual workflow

#### Quality Assurance
- **Link Validation**: Verified all documentation cross-references
  - Tested all internal links between documentation files
  - Validated generated documentation functionality
  - Ensured consistent navigation throughout documentation system

### Enhanced Developer Experience
- **Streamlined Workflow**: Simplified documentation and code formatting processes
  - Single command documentation generation: `./scripts/generate_docs.sh`
  - Unified API documentation approach with automatic updates
  - Clear separation between generated and maintained content
  - Professional documentation structure ready for open-source collaboration
