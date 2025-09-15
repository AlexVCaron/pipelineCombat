# PEP 8 Development Workflow Guide

This project is configured with comprehensive PEP 8 compliance and modern Python development best practices.

## 🔧 Available Tools & Scripts

### Formatting Scripts

| Script | Purpose | Usage |
|--------|---------|--------|
| `scripts/format_code.sh` | Format entire codebase | `./scripts/format_code.sh` |
| `scripts/format_file_comprehensive.sh` | Format single file | `./scripts/format_file_comprehensive.sh <file>` |
| `scripts/lint_check.sh` | Quick PEP 8 lint check | `./scripts/lint_check.sh [file_or_dir]` |
| `scripts/validate_pep8_config.sh` | Validate tool configuration | `./scripts/validate_pep8_config.sh` |

### VS Code Tasks

Access via `Ctrl+Shift+P` → "Tasks: Run Task":

- **PEP 8: Format Current File** - Format the currently open file
- **PEP 8: Format Entire Codebase** - Format all Python files
- **PEP 8: Quick Lint Check** - Check code quality
- **PEP 8: Validate Configuration** - Verify tool setup
- **Run Tests with Coverage** - Execute tests with coverage report

## 📋 PEP 8 Configuration Details

### Tools & Settings

- **Line Length**: 79 characters (PEP 8 standard)
- **Indentation**: 4 spaces (no tabs)
- **Quote Style**: Double quotes preferred
- **Docstring Style**: NumPy conventions
- **Target Python**: 3.12+

### Comprehensive Rule Coverage

**Ruff Enabled Rules (27 categories):**
- `E` - pycodestyle errors (PEP 8 compliance)
- `W` - pycodestyle warnings
- `F` - pyflakes (unused imports, variables)
- `I` - isort (import organization)
- `N` - pep8-naming (naming conventions)
- `D` - pydocstyle (docstring quality)
- `UP` - pyupgrade (modern Python syntax)
- `B` - flake8-bugbear (common bugs)
- `C4` - flake8-comprehensions
- `C90` - mccabe (complexity)
- `PL` - pylint (code quality)
- Plus 16 additional specialized rule sets

### VS Code Integration

- **Rulers**: Visual guides at 72 (docstrings) and 79 (code) characters
- **Format on Save**: Automatic PEP 8 formatting
- **Real-time Linting**: Instant feedback via ruff language server
- **Import Organization**: Automatic import sorting
- **Type Checking**: Pylance with comprehensive analysis

## 🚀 Development Workflow

### 1. Quick Development Check
```bash
./scripts/lint_check.sh src/
```

### 2. Format Single File
```bash
./scripts/format_file_comprehensive.sh examples/demo.py
```

### 3. Format Entire Codebase
```bash
./scripts/format_code.sh
```

### 4. Run Tests with Coverage
```bash
uv run pytest --cov=src/pipelinecombat --cov-report=html
```

### 5. Validate Configuration
```bash
./scripts/validate_pep8_config.sh
```

## 🎯 What Gets Checked & Fixed

### Automatic Fixes
- Line length violations (wrapped appropriately)
- Import sorting and organization
- Unused imports/variables removal
- Modern Python syntax upgrades
- Quote style normalization
- Trailing whitespace removal
- Final newline insertion

### Manual Fix Required
- Complex docstring issues
- Boolean argument patterns (FBT rules)
- Code complexity violations
- Private member access patterns
- Some naming convention violations

## 📊 Quality Standards

- **Test Coverage**: 80% minimum (configured in pyproject.toml)
- **Line Length**: 79 characters maximum
- **Complexity**: McCabe complexity ≤ 15
- **Documentation**: NumPy-style docstrings required
- **Type Hints**: Encouraged for public APIs
- **Import Style**: Organized in PEP 8 order

## 🛠️ Manual Commands

For advanced usage:

```bash
# Run comprehensive check
uv run ruff check src/ tests/ examples/

# Auto-fix what's possible
uv run ruff check --fix src/ tests/ examples/

# Format with ruff
uv run ruff format src/ tests/ examples/

# Run specific rule categories
uv run ruff check --select E,W,F,D src/

# Generate detailed report
uv run ruff check --output-format=json src/ > lint_report.json
```

This configuration ensures consistent, high-quality Python code that follows PEP 8 standards and modern development best practices.
