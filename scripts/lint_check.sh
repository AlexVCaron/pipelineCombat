#!/bin/bash
# Quick PEP 8 lint check for development workflow
# Usage: ./lint_check.sh [file_or_directory]

TARGET=${1:-"src/ tests/ examples/"}

echo "🔍 PEP 8 Lint Check"
echo "==================="
echo "Target: $TARGET"
echo ""

# Run comprehensive ruff check
echo "🔧 Running ruff check (comprehensive PEP 8 analysis)..."
if uv run --group dev ruff check $TARGET; then
    echo ""
    echo "✅ All files pass PEP 8 compliance!"
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo "❌ PEP 8 violations found"
    echo ""
    echo "🛠️  Quick fixes available:"
    echo "   Format single file:    ./scripts/format_file_comprehensive.sh <file>"
    echo "   Format entire codebase: ./scripts/format_code.sh"
    echo ""
    echo "🔧 Auto-fix what's possible:"
    echo "   uv run ruff check --fix $TARGET"
    echo ""
    exit $EXIT_CODE
fi
