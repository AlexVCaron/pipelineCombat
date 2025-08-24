#!/bin/bash
# Format a single Python file with autoflake, isort, pyupgrade, and black

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

echo "🧹 Running autoflake on $FILE..."
uv run --group dev autoflake --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports --in-place "$FILE"

echo "📦 Running isort on $FILE..."
uv run --group dev isort --profile=black "$FILE"

echo "⬆️  Running pyupgrade on $FILE..."
uv run --group dev pyupgrade --py312-plus "$FILE"

echo "⚫ Running black on $FILE..."
uv run --group dev black "$FILE"

echo "✅ Formatted $FILE"
