"""conftest.py — ensures the project root is on sys.path for tests."""
import sys
from pathlib import Path

# Add the repository root to the Python path so `from src.xxx import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent))
