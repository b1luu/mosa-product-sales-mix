"""Load Square export data from local disk.

This module provides a placeholder entry point for reading raw exports into
pandas DataFrames. It should be adapted to the exact Square export schema.
"""
from __future__ import annotations

from pathlib import Path


def load_square_exports(raw_dir: Path) -> None:
    """Load Square export files from a local directory.

    Args:
        raw_dir: Path to the directory containing Square export files.

    Returns:
        None. This placeholder should be updated to return parsed data.
    """
    raise NotImplementedError("Implement Square export loading logic.")
