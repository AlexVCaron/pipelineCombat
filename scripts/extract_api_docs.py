#!/usr/bin/env python3
"""
Simple API documentation extractor.
Creates a basic markdown index from pdoc HTML output.
"""

import sys
from pathlib import Path


def create_api_index(api_html_dir: Path) -> str:
    """Create a simple markdown index for the HTML API docs."""
    markdown_lines = [
        "# API Reference",
        "",
        "*This documentation is automatically generated from docstrings using pdoc.*",
        "",
        "## 📖 Viewing the API Documentation",
        "",
        "The complete API docs are in HTML format for best browsing:",
        "",
        "**[Open API Documentation](api-html/index.html)** (HTML format)",
        "",
        "## 📋 Quick Module Overview",
        "",
    ]

    # Look for submodule files
    pipelinecombat_dir = api_html_dir / "pipelinecombat"
    if pipelinecombat_dir.exists():
        markdown_lines.append("### Available Modules")
        markdown_lines.append("")

        for html_file in sorted(pipelinecombat_dir.glob("*.html")):
            if html_file.name != "index.html":
                module_name = html_file.stem
                link = f"api-html/pipelinecombat/{html_file.name}"
                markdown_lines.append(
                    f"- **[{module_name}]({link})** - {module_name.title()}"
                )

        markdown_lines.append("")

    markdown_lines.extend(
        [
            "## 💡 Usage",
            "",
            "For the best experience viewing the API documentation:",
            "",
            "1. **Local**: Open `docs/api-html/index.html` in your browser",
            "2. **Dev**: Use the 'Generate API Documentation' VS Code task",
            "3. **Command line**: Run `./scripts/generate_docs.sh`",
            "",
            "## 🔄 Updating Documentation",
            "",
            "API docs are auto-generated from docstrings in source code.",
            "To update the documentation:",
            "",
            "```bash",
            "# Generate new documentation",
            "./scripts/generate_docs.sh",
            "```",
            "",
            "VS Code: `Ctrl+Shift+P` → `Tasks: Run Task` → `Generate API`",
            "",
            "---",
            "",
            "*For detailed usage examples and guides, see the [User Guide](user-guide.md).*",
        ]
    )

    return "\n".join(markdown_lines)


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print(
            "Usage: python extract_api_docs.py <api_html_dir> <output_md_file>"
        )
        sys.exit(1)

    api_html_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not api_html_dir.exists():
        print(f"HTML documentation directory not found: {api_html_dir}")
        sys.exit(1)

    # Create index markdown
    markdown_content = create_api_index(api_html_dir)

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"✅ Generated API documentation index: {output_file}")


if __name__ == "__main__":
    main()
