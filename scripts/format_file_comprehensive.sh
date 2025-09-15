#!/bin/bash

# Comprehensive Python file formatter with enhanced PEP 8 compliance
# Usage: ./format_file_comprehensive.sh <file_path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <python_file>"
    exit 1
fi

file="$1"

if [[ ! "$file" == *.py ]] || [[ ! -f "$file" ]]; then
    echo "❌ Not a Python file or file does not exist: $file"
    exit 1
fi

echo "🧹 Comprehensive PEP 8 formatting of $file..."

# Step 1: Upgrade to modern Python syntax first
echo "⬆️  Running pyupgrade (modern Python syntax)..."
uv run --group dev pyupgrade --py312-plus "$file"

# Step 2: Clean up unused imports/variables
echo "🗑️  Running autoflake (unused imports/variables)..."
uv run --group dev autoflake \
    --remove-all-unused-imports \
    --remove-unused-variables \
    --remove-duplicate-keys \
    --expand-star-imports \
    --ignore-init-module-imports \
    --in-place "$file"

# Step 3: Organize imports with PEP 8 compliance
echo "📦 Running isort (import organization)..."
uv run --group dev isort "$file"

# Step 4: Apply comprehensive auto-fixes with ruff
echo "🔧 Running ruff check --fix (comprehensive auto-fixes)..."
uv run --group dev ruff check --fix "$file"

# Step 5: Format code with ruff (modern, fast formatter)
echo "📐 Running ruff format (PEP 8 formatting)..."
uv run --group dev ruff format "$file"

# Step 6: Final black pass for any remaining formatting
echo "⚫ Running black (final formatting pass)..."
uv run --group dev black "$file"

# Step 7: Final comprehensive check and report
echo "🔍 Running final ruff check..."
if uv run --group dev ruff check "$file"; then
    echo "✅ File passes all PEP 8 checks!"
else
    echo "⚠️  Some issues remain (manual fix may be needed)"
    echo "📋 Run 'uv run ruff check $file' for detailed report"
fi

echo "✅ Comprehensive PEP 8 formatting complete for $file"
