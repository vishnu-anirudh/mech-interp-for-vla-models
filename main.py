#!/usr/bin/env python3
"""
Entry point for the mechanistic interpretability pipeline.

This script allows running the main pipeline from the project root.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run main
from main import main

if __name__ == "__main__":
    main()
