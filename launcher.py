#!/usr/bin/env python3
"""
This is the main entry point for the AFAGNetV3 project.
Run this script to access all project management features.

Usage:
    python launcher.py              # Interactive mode
    python launcher.py --setup-only # Setup only
    python launcher.py --help       # Show help
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    """Launch the project"""
    try:
        from project import main as pm_main
        pm_main()
    except ImportError as e:
        print("Error: Could not import project.py")
        print(f"Details: {e}")
        print("\nMake sure you're running this from the project root directory.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f" Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()