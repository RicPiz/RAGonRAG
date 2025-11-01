#!/usr/bin/env python3
"""
RAGonRAG Streamlit App Launcher

This script provides a clean way to run the RAGonRAG Streamlit application
with proper module path configuration.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app with proper configuration."""
    # Add src directory to Python path
    src_dir = Path(__file__).parent / "src"
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Set environment variables for better Streamlit experience
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "localhost")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "8501")

    # Import and run streamlit
    try:
        import streamlit.web.cli as st_cli
        import sys

        # Run the app
        sys.argv = ["streamlit", "run", str(src_dir / "app.py")]
        st_cli.main()

    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
