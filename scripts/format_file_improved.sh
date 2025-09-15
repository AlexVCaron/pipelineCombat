#!/bin/bash
# Format a single Python file with comprehensive linting and formatting tools

set -e  # Exit on any error

if [ -z "$1" ]; then
    echo "❌ Usage: $0 <python_file>"
    exit 1
fi

FILE="$1"

if [[ ! "$FILE" == *.py ]]; then
    echo "❌ Not a Python file: $FILE"
    exit 1
fi

if [[ ! -f "$FILE" ]]; then
    echo "❌ File does not exist: $FILE"
    exit 1
fi

echo "🔧 Comprehensive Python file formatting for: $FILE"
echo "=" * 60

# Step 1: Remove unused imports and variables
echo "🧹 Step 1: Running autoflake..."
uv run --group dev autoflake --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports --in-place "$FILE"

# Step 2: Sort imports
echo "📦 Step 2: Running isort..."
uv run --group dev isort --profile=black --line-length=79 "$FILE"

# Step 3: Upgrade Python syntax
echo "⬆️  Step 3: Running pyupgrade..."
uv run --group dev pyupgrade --py312-plus "$FILE"

# Step 4: Format with ruff (handles more edge cases than black alone)
echo "🚀 Step 4: Running ruff format..."
uv run --group dev ruff format --line-length=79 "$FILE"

# Step 5: Run ruff linting with auto-fixes
echo "🔍 Step 5: Running ruff check with auto-fixes..."
uv run --group dev ruff check --fix "$FILE" || true

# Step 6: Final black formatting pass
echo "⚫ Step 6: Final black formatting..."
uv run --group dev black --line-length=79 "$FILE"

# Step 7: Check remaining issues
echo "📋 Step 7: Checking remaining linting issues..."
echo "Remaining ruff issues (if any):"
uv run --group dev ruff check "$FILE" || echo "✅ No remaining ruff issues"

echo ""
echo "✅ Comprehensive formatting completed for $FILE"
echo "=" * 60