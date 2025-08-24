#!/bin/bash
# Generate API documentation using pdoc

set -e  # Exit on any error

echo "📚 Generating API documentation with pdoc..."

# Clean up previous generated docs
rm -rf docs/api-html
rm -f docs/api-reference.md

# Generate HTML documentation
uv run --group dev pdoc \
    --output-directory docs/api-html \
    --docformat markdown \
    --no-show-source \
    --footer-text "Pipeline Combat API Documentation" \
    pipelinecombat

# Generate markdown index for GitHub
python scripts/extract_api_docs.py docs/api-html docs/api-reference.md

echo "✅ API documentation generated!"
echo "   - HTML documentation: docs/api-html/"
echo "   - GitHub markdown index: docs/api-reference.md"
echo "   - Open docs/api-html/index.html in your browser to view"
echo ""
echo "� The generated docs are now available:"
echo "   - For development: Open HTML files locally"
echo "   - For GitHub: Use the markdown index that links to HTML files"
