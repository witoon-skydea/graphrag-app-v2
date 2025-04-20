#!/usr/bin/env python3
"""
GraphRAG Main Script

This script provides a command line interface for running the GraphRAG system.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import CLI module
from src.cli_api.cli import graphrag

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('graphrag.log')
        ]
    )
    
    # Run CLI
    graphrag()
