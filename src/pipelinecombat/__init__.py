"""
Pipeline Combat: Neuroimaging Data Processing and Harmonization

A Python package for processing and harmonizing neuroimaging data across
different scanners and acquisition sites using statistical methods.
"""

import logging
from pathlib import Path
from typing import Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

__version__ = "0.1.0"
__author__ = "AlexVCaron"


def main() -> None:
    """Main entry point for the pipelinecombat package."""
    print("🧠 Welcome to Pipeline Combat!")
    print("Neuroimaging Data Processing and Harmonization Toolkit")
    print(f"Version: {__version__}")
    print("\nAvailable modules:")
    print("- neuroCombat: Statistical harmonization")
    print("- dipy: Diffusion imaging processing")
    print("- pgmpy: Probabilistic graphical models")
    print("- statsmodels: Statistical modeling")

    # Basic dependency check
    try:
        import dipy
        import neuroHarmonize  # neuroCombat is imported as neuroHarmonize
        import pgmpy
        import statsmodels

        print("\n✅ All dependencies are properly installed!")
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Run 'uv sync' to install all dependencies.")


def get_version() -> str:
    """Get the package version."""
    return __version__


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.getLogger().setLevel(numeric_level)
    logging.info(f"Logging level set to {level}")


if __name__ == "__main__":
    main()
