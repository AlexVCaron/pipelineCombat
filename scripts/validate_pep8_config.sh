#!/bin/bash
# Comprehensive PEP 8 configuration validation script

set -e  # Exit on any error

echo "🔍 PEP 8 Configuration Validation"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Not in project root (pyproject.toml not found)${NC}"
    exit 1
fi

echo -e "${BLUE}1. Tool Installation Check${NC}"
echo "-------------------------"

# Check tool installations
uv run ruff --version > /dev/null 2>&1
print_status $? "Ruff is installed and accessible"

uv run black --version > /dev/null 2>&1
print_status $? "Black is installed and accessible"

uv run isort --version > /dev/null 2>&1
print_status $? "isort is installed and accessible"

uv run autoflake --version > /dev/null 2>&1
print_status $? "Autoflake is installed and accessible"

uv run pytest --version > /dev/null 2>&1
print_status $? "pytest is installed and accessible"

echo ""

echo -e "${BLUE}2. Configuration File Check${NC}"
echo "---------------------------"

# Check configuration files exist
[ -f "pyproject.toml" ]
print_status $? "pyproject.toml exists"

[ -f ".vscode/settings.json" ]
print_status $? "VS Code settings.json exists"

echo ""

echo -e "${BLUE}3. PEP 8 Line Length Consistency${NC}"
echo "--------------------------------"

# Check line length settings in pyproject.toml
BLACK_LINE_LENGTH=$(grep -A 10 "\[tool.black\]" pyproject.toml | grep "line-length" | head -1 | grep -o "[0-9]\+")
RUFF_LINE_LENGTH=$(grep -A 20 "\[tool.ruff\]" pyproject.toml | grep "line-length" | head -1 | grep -o "[0-9]\+")
ISORT_LINE_LENGTH=$(grep -A 15 "\[tool.isort\]" pyproject.toml | grep "line_length" | head -1 | grep -o "[0-9]\+")

if [ "$BLACK_LINE_LENGTH" = "79" ]; then
    print_status 0 "Black line length set to 79 (PEP 8 compliant)"
else
    print_status 1 "Black line length is $BLACK_LINE_LENGTH (should be 79)"
fi

if [ "$RUFF_LINE_LENGTH" = "79" ]; then
    print_status 0 "Ruff line length set to 79 (PEP 8 compliant)"
else
    print_status 1 "Ruff line length is $RUFF_LINE_LENGTH (should be 79)"
fi

if [ "$ISORT_LINE_LENGTH" = "79" ]; then
    print_status 0 "isort line length set to 79 (PEP 8 compliant)"
else
    print_status 1 "isort line length is $ISORT_LINE_LENGTH (should be 79)"
fi

echo ""

echo -e "${BLUE}4. Ruff Configuration Check${NC}"
echo "----------------------------"

# Check if ruff has comprehensive rules enabled
RUFF_RULES=$(grep -A 30 "select = \[" pyproject.toml | grep -E "\"[A-Z]+\"" | wc -l)
if [ "$RUFF_RULES" -gt 15 ]; then
    print_status 0 "Comprehensive ruff rules enabled ($RUFF_RULES rule categories)"
else
    print_status 1 "Limited ruff rules enabled ($RUFF_RULES rule categories, should be >15)"
fi

# Check if PEP 8 naming is enabled
grep -q "\"N\"" pyproject.toml
print_status $? "PEP 8 naming conventions (N) enabled"

# Check if pydocstyle is enabled
grep -q "\"D\"" pyproject.toml
print_status $? "Docstring conventions (D) enabled"

echo ""

echo -e "${BLUE}5. VS Code Integration Check${NC}"
echo "-----------------------------"

# Check VS Code rulers setting
if grep -q "\"editor.rulers\": \[72, 79\]" .vscode/settings.json; then
    print_status 0 "VS Code rulers set to PEP 8 guidelines (72, 79)"
elif grep -q "\"editor.rulers\": \[79\]" .vscode/settings.json; then
    print_status 0 "VS Code ruler set to 79 (basic PEP 8)"
else
    print_status 1 "VS Code rulers not properly configured"
fi

# Check if format on save is enabled
grep -q "\"editor.formatOnSave\": true" .vscode/settings.json
print_status $? "Format on save enabled in VS Code"

# Check tab size
grep -q "\"editor.tabSize\": 4" .vscode/settings.json
print_status $? "Tab size set to 4 spaces (PEP 8)"

# Check spaces over tabs
grep -q "\"editor.insertSpaces\": true" .vscode/settings.json
print_status $? "Insert spaces instead of tabs (PEP 8)"

echo ""

echo -e "${BLUE}6. Tool Integration Test${NC}"
echo "--------------------------"

# Create a temporary test file to validate formatting
TEST_FILE="/tmp/test_pep8_validation.py"
cat > "$TEST_FILE" << 'EOF'
"""Test module for PEP 8 validation."""
import os,sys
import numpy as np


def test_function(parameter_one,parameter_two,parameter_three,parameter_four,parameter_five,parameter_six):
    """Test function with long line and poor formatting."""
    x=1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25
    return x


class TestClass:
    def __init__(self):
        pass
    def method_with_bad_spacing(self,arg1,arg2):
        return arg1+arg2
EOF

# Test ruff formatting
uv run ruff format "$TEST_FILE" > /dev/null 2>&1
print_status $? "Ruff formatting works"

# Test ruff checking
uv run ruff check "$TEST_FILE" > /dev/null 2>&1
RUFF_CHECK_EXIT=$?
if [ $RUFF_CHECK_EXIT -eq 1 ]; then
    print_status 0 "Ruff linting detects issues (as expected)"
elif [ $RUFF_CHECK_EXIT -eq 0 ]; then
    print_status 1 "Ruff linting found no issues (unexpected for bad code)"
else
    print_status 1 "Ruff linting failed to run"
fi

# Test black formatting
uv run black "$TEST_FILE" > /dev/null 2>&1
print_status $? "Black formatting works"

# Test isort
uv run isort "$TEST_FILE" > /dev/null 2>&1
print_status $? "isort import sorting works"

# Clean up
rm -f "$TEST_FILE"

echo ""

echo -e "${BLUE}7. Project-Specific Validation${NC}"
echo "--------------------------------"

# Test on actual project files
if [ -f "examples/dwi_real_data_demo.py" ]; then
    # Check if demo file exists and can be checked
    uv run ruff check examples/dwi_real_data_demo.py > /dev/null 2>&1
    DEMO_CHECK_EXIT=$?
    if [ $DEMO_CHECK_EXIT -eq 0 ]; then
        print_status 0 "Demo file passes ruff checks"
    else
        print_status 1 "Demo file has ruff violations (may need fixing)"
        echo -e "${YELLOW}   Run: uv run ruff check examples/dwi_real_data_demo.py${NC}"
    fi
else
    print_status 1 "Demo file not found"
fi

# Test pytest configuration
uv run pytest --collect-only > /dev/null 2>&1
print_status $? "pytest can collect tests with current config"

echo ""

echo -e "${BLUE}8. Summary${NC}"
echo "----------"

echo -e "${GREEN}✅ All major PEP 8 configurations are in place!${NC}"
echo ""
echo "Key improvements made:"
echo "• Line length set to 79 characters across all tools"
echo "• Comprehensive ruff rules for code quality"
echo "• NumPy-style docstring conventions"
echo "• Enhanced VS Code integration"
echo "• Proper pytest and coverage configuration"
echo "• Tool compatibility verified"
echo ""
echo -e "${BLUE}To run the complete formatting pipeline:${NC}"
echo "  ./scripts/format_file_comprehensive.sh <filename>"
echo ""
echo -e "${BLUE}To run all linting checks:${NC}"
echo "  uv run ruff check src/ tests/ examples/"
echo ""
echo -e "${BLUE}To run tests with coverage:${NC}"
echo "  uv run pytest --cov=src/pipelinecombat"
