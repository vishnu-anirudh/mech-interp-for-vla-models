#!/usr/bin/env python3
"""
Entry point for graph generation.

This script allows generating graphs from the project root.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import and run main
from generate_graphs_from_results import main

if __name__ == "__main__":
    main()
