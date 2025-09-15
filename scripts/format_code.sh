#!/bin/bash
# Comprehensive PEP 8 formatting for entire codebase
# Formats all Python files in src/, tests/, examples/, and scripts/

set -e  # Exit on any error

echo "🧹 Comprehensive PEP 8 codebase formatting..."
echo "=============================================="

# Define target directories
DIRS="src/ tests/ examples/ scripts/"

# Step 1: Upgrade to modern Python syntax
echo "⬆️  Running pyupgrade (modern Python syntax)..."
find $DIRS -name '*.py' -exec uv run --group dev pyupgrade --py312-plus {} \; 2>/dev/null || true

# Step 2: Clean up unused imports and variables
echo "🗑️  Running autoflake (cleanup unused code)..."
uv run --group dev autoflake \
    --remove-all-unused-imports \
    --remove-unused-variables \
    --remove-duplicate-keys \
    --expand-star-imports \
    --ignore-init-module-imports \
    --in-place \
    --recursive $DIRS

# Step 3: Organize imports with PEP 8 compliance
echo "📦 Running isort (import organization)..."
uv run --group dev isort $DIRS

# Step 4: Apply comprehensive auto-fixes
echo "🔧 Running ruff check --fix (comprehensive auto-fixes)..."
uv run --group dev ruff check --fix $DIRS || echo "Some issues fixed, continuing..."

# Step 5: Format with ruff (primary formatter)
echo "📐 Running ruff format (PEP 8 formatting)..."
uv run --group dev ruff format $DIRS

# Step 6: Final black pass for any remaining formatting
echo "⚫ Running black (final formatting pass)..."
uv run --group dev black $DIRS

# Step 7: Final check and summary
echo "🔍 Running final quality check..."
echo ""
if uv run --group dev ruff check $DIRS --quiet; then
    echo "✅ All files pass PEP 8 compliance checks!"
else
    echo "⚠️  Some issues remain in the codebase"
    echo "📋 Run 'uv run ruff check $DIRS' for detailed report"
fi

echo ""
echo "✅ Comprehensive PEP 8 codebase formatting complete!"
echo "📊 Summary:"
echo "   • Code upgraded to modern Python 3.12+ syntax"
echo "   • Unused imports and variables removed"
echo "   • Imports organized by PEP 8 standards"
echo "   • Comprehensive linting fixes applied"
echo "   • 79-character line length enforced"
echo "   • NumPy docstring style maintained"
