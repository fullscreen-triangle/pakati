#!/usr/bin/env python3
"""
Simple script to start the Pakati server.

This script provides a convenient way to start the Pakati server
with default settings.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add the parent directory to the path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import Pakati
from pakati.__main__ import serve


def main():
    """Run the Pakati server."""
    # Call the serve function from the CLI
    serve()


if __name__ == "__main__":
    main() 