"""Load Square export data from local disk."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _select_export_csv(raw_dir: Path) -> Path:
    """Pick a single CSV export from a directory."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    preferred = raw_dir / "orders.csv"
    if preferred.exists():
        return preferred

    csv_candidates = sorted(raw_dir.glob("*.csv"))
    if len(csv_candidates) == 1:
        return csv_candidates[0]
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV exports found in {raw_dir}")
    raise FileNotFoundError(
        "Multiple CSV exports found. Keep one file or name it orders.csv."
    )


def load_square_exports(raw_dir: Path) -> pd.DataFrame:
    """Load Square export files from a local directory.

    Args:
        raw_dir: Path to the directory containing Square export files.

    Returns:
        DataFrame containing the raw export data.
    """
    export_path = _select_export_csv(raw_dir)
    return pd.read_csv(export_path)
