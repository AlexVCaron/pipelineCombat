#!/bin/bash
# Format Python code with autoflake, isort, pyupgrade, and black

set -e  # Exit on any error

echo "🧹 Running autoflake..."
uv run --group dev autoflake --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports --in-place --recursive src/ examples/

echo "📦 Running isort..."
uv run --group dev isort --profile=black src/ examples/

echo "⬆️  Running pyupgrade..."
find src/ examples/ -name '*.py' -exec uv run --group dev pyupgrade --py312-plus {} \;

echo "⚫ Running black..."
uv run --group dev black src/ examples/

echo "✅ Code formatting complete!"
