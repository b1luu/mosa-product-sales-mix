"""Compute sales mix by category and product from Square export data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from src.load_data import load_square_exports

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected schema."""
    df = df.copy()
    rename_map = {}

    column_map = {
        "order_id": ["order_id", "order id", "order", "transaction id"],
        "order_datetime": ["order_datetime", "order datetime", "created at"],
        "category_name": ["category_name", "category", "category name"],
        "item_name": ["item_name", "item", "item name"],
        "quantity": ["quantity", "qty"],
        "item_gross_sales": [
            "item_gross_sales",
            "gross sales",
            "gross_sales",
            "total",
            "total sales",
            "sales",
        ],
    }

    for target, variants in column_map.items():
        for variant in variants:
            for col in df.columns:
                if col.strip().lower() == variant:
                    rename_map[col] = target
                    break

    df = df.rename(columns=rename_map)
    return df


def _build_order_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create order_datetime from Date and Time columns when available."""
    df = df.copy()
    if "order_datetime" in df.columns:
        return df

    normalized = {col.strip().lower(): col for col in df.columns}
    has_date = "date" in normalized
    has_time = "time" in normalized
    if has_date and has_time:
        date_col = normalized["date"]
        time_col = normalized["time"]
        df["order_datetime"] = pd.to_datetime(
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    return df


def _coerce_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure item_gross_sales is numeric and handle missing values."""
    df = df.copy()
    if "item_gross_sales" not in df.columns:
        raise ValueError("Missing required column: item_gross_sales")

    if df["item_gross_sales"].dtype == object:
        df["item_gross_sales"] = (
            df["item_gross_sales"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

    missing_sales = df["item_gross_sales"].isna().sum()
    if missing_sales > 0:
        print(
            f"Warning: {missing_sales} rows missing item_gross_sales; treating as 0."
        )
    df["item_gross_sales"] = (
        pd.to_numeric(df["item_gross_sales"], errors="coerce").fillna(0)
    )
    return df


def _compute_category_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute category-level sales mix for a given window."""
    if df.empty:
        return pd.DataFrame(
            columns=["category_name", "total_sales", "category_sales_pct_of_total"]
        )

    category_mix = (
        df.groupby("category_name", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = category_mix["total_sales"].sum()
    if total_sales == 0:
        category_mix["category_sales_pct_of_total"] = 0.0
    else:
        category_mix["category_sales_pct_of_total"] = (
            category_mix["total_sales"] / total_sales
        )
    category_mix = category_mix.sort_values("total_sales", ascending=False)
    return category_mix


def _compute_product_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute product-level sales mix for a given window."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "category_name",
                "item_name",
                "total_sales",
                "product_sales_pct_of_category",
                "product_sales_pct_of_total",
            ]
        )

    product_mix = (
        df.groupby(["category_name", "item_name"], dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )

    total_sales = product_mix["total_sales"].sum()
    if total_sales == 0:
        product_mix["product_sales_pct_of_total"] = 0.0
    else:
        product_mix["product_sales_pct_of_total"] = (
            product_mix["total_sales"] / total_sales
        )

    category_totals = (
        product_mix.groupby("category_name", dropna=False)["total_sales"]
        .sum()
        .reset_index()
        .rename(columns={"total_sales": "category_total_sales"})
    )
    product_mix = product_mix.merge(category_totals, on="category_name", how="left")
    product_mix["product_sales_pct_of_category"] = product_mix.apply(
        lambda row: 0.0
        if row["category_total_sales"] == 0
        else row["total_sales"] / row["category_total_sales"],
        axis=1,
    )
    product_mix = product_mix.drop(columns=["category_total_sales"])

    product_mix = product_mix.sort_values(
        ["category_name", "total_sales"], ascending=[True, False]
    )
    return product_mix


def _print_summary(
    category_mix: pd.DataFrame,
    product_mix: pd.DataFrame,
    label: str,
) -> None:
    """Print a brief console summary for a window."""
    print(f"\nSummary for {label}:")
    if category_mix.empty:
        print("Warning: category mix is empty.")
    else:
        top_categories = category_mix.head(5)
        print("Top categories:")
        for _, row in top_categories.iterrows():
            pct = row["category_sales_pct_of_total"] * 100
            print(f"  {row['category_name']}: {row['total_sales']:.2f} ({pct:.1f}%)")

    if product_mix.empty:
        print("Warning: product mix is empty.")
    else:
        top_products = product_mix.sort_values("total_sales", ascending=False).head(10)
        print("Top products:")
        for _, row in top_products.iterrows():
            pct_total = row["product_sales_pct_of_total"] * 100
            print(
                f"  {row['category_name']} - {row['item_name']}: "
                f"{row['total_sales']:.2f} ({pct_total:.1f}%)"
            )


def main() -> None:
    """Run sales mix computation for last month and last 3 months."""
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and standardize raw export data.
    df = load_square_exports(raw_dir)
    df = _normalize_columns(df)
    df = _build_order_datetime(df)

    required_cols = {
        "order_id",
        "order_datetime",
        "category_name",
        "item_name",
        "quantity",
        "item_gross_sales",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Expected a detailed line-item export; missing required columns after "
            f"normalization: {sorted(missing_cols)}"
        )

    # Parse dates and handle basic data issues.
    df["order_datetime"] = pd.to_datetime(df["order_datetime"], errors="coerce")
    invalid_dates = df["order_datetime"].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: {invalid_dates} rows have invalid order_datetime values.")
    df = df.dropna(subset=["order_datetime"])

    df = _coerce_sales(df)

    if df.empty:
        print("Warning: no valid rows after cleaning; exiting.")
        return

    # Define calendar windows based on the latest order date.
    max_date = df["order_datetime"].max()
    last_month_start = max_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (last_month_start + pd.offsets.MonthBegin(1)).to_pydatetime()
    last_month_end = next_month - pd.Timedelta(seconds=1)
    last_3_start = (last_month_start - pd.offsets.MonthBegin(2)).to_pydatetime()

    print("Max order date:", max_date)
    print(
        "Last full month window:",
        last_month_start,
        "to",
        last_month_end,
    )
    print(
        "Last 3 full months window:",
        last_3_start,
        "to",
        last_month_end,
    )

    # Filter to the desired windows.
    df_last_month = df[
        (df["order_datetime"] >= last_month_start)
        & (df["order_datetime"] <= last_month_end)
    ]
    df_last_3_months = df[
        (df["order_datetime"] >= last_3_start)
        & (df["order_datetime"] <= last_month_end)
    ]

    if df_last_month.empty:
        print("Warning: last month window has no rows.")
    if df_last_3_months.empty:
        print("Warning: last 3 months window has no rows.")

    last_month_category = _compute_category_mix(df_last_month)
    last_month_product = _compute_product_mix(df_last_month)
    last_3_category = _compute_category_mix(df_last_3_months)
    last_3_product = _compute_product_mix(df_last_3_months)
    global_category = _compute_category_mix(df)
    global_product = _compute_product_mix(df)

    if last_month_category.empty:
        print("Warning: last month category mix is empty.")
    if last_month_product.empty:
        print("Warning: last month product mix is empty.")
    if last_3_category.empty:
        print("Warning: last 3 months category mix is empty.")
    if last_3_product.empty:
        print("Warning: last 3 months product mix is empty.")
    if global_category.empty:
        print("Warning: global category mix is empty.")
    if global_product.empty:
        print("Warning: global product mix is empty.")

    last_month_category.to_csv(
        processed_dir / "last_month_category_mix.csv", index=False
    )
    last_month_product.to_csv(
        processed_dir / "last_month_product_mix.csv", index=False
    )
    last_3_category.to_csv(
        processed_dir / "last_3_months_category_mix.csv", index=False
    )
    last_3_product.to_csv(
        processed_dir / "last_3_months_product_mix.csv", index=False
    )
    global_category.to_csv(processed_dir / "global_category_mix.csv", index=False)
    global_product.to_csv(processed_dir / "global_product_mix.csv", index=False)

    _print_summary(last_month_category, last_month_product, "Last Month")
    _print_summary(global_category, global_product, "All Data")


if __name__ == "__main__":
    main()
