"""Load Square export data from local disk."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _validate_detailed_export_schema(df: pd.DataFrame) -> None:
    """Validate that a detailed line-item export schema is present."""
    normalized = {col.strip().lower(): col for col in df.columns}

    required = {
        "order_id": ["order id", "order", "transaction id"],
        "order_datetime": ["order datetime", "created at"],
        "date": ["date"],
        "time": ["time"],
        "category_name": ["category", "category name"],
        "item_name": ["item", "item name"],
        "quantity": ["quantity", "qty"],
        "item_gross_sales": ["item gross sales", "gross sales", "gross_sales", "total", "total sales", "sales"],
    }

    missing = []
    def has_any(variants: list[str]) -> bool:
        return any(variant in normalized for variant in variants)

    if not has_any(required["order_id"]):
        missing.append("order_id")
    if not (has_any(required["order_datetime"]) or (has_any(required["date"]) and has_any(required["time"]))):
        missing.append("order_datetime or Date+Time")
    if not has_any(required["category_name"]):
        missing.append("category_name")
    if not has_any(required["item_name"]):
        missing.append("item_name")
    if not has_any(required["quantity"]):
        missing.append("quantity")
    if not has_any(required["item_gross_sales"]):
        missing.append("item_gross_sales")

    if missing:
        raise ValueError(
            "Detailed line-item export schema missing required fields: "
            f"{', '.join(missing)}"
        )


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
    df = pd.read_csv(export_path)
    _validate_detailed_export_schema(df)
    return df
