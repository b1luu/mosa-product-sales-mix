"""Compute sales mix by category and product from Square export data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    from load_data import load_square_exports
    from config import EXCLUDE_ITEM_PATTERNS, KEEP_REFUND_PATTERNS
except ImportError:  # pragma: no cover - fallback for package-style imports
    from src.load_data import load_square_exports
    from src.config import EXCLUDE_ITEM_PATTERNS, KEEP_REFUND_PATTERNS

# --- Data normalization and cleaning helpers ---
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to expected schema."""
    df = df.copy()
    rename_map = {}

    # Map common Square export headers to a normalized schema.
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
                # Case-insensitive, whitespace-tolerant header matching.
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

    # If order_datetime isn't provided, combine Date + Time into a timestamp.
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


def _filter_refunds(df: pd.DataFrame) -> pd.DataFrame:
    """Filter refunds while keeping Panda-noted rows and removing cancellations."""
    df = df.copy()
    columns = {col.strip().lower(): col for col in df.columns}
    event_col = columns.get("event type") or columns.get("event_type")
    notes_col = columns.get("notes")

    if notes_col:
        # Drop explicit cancellations regardless of event type.
        canceled_mask = (
            df[notes_col]
            .astype(str)
            .str.contains("canceled order", case=False, na=False)
        )
        df = df[~canceled_mask]

    if event_col:
        refund_mask = (
            df[event_col].astype(str).str.strip().str.lower() == "refund"
        )
        if notes_col:
            keep_refund_mask = (
                df[notes_col]
                .astype(str)
                .str.contains("|".join(KEEP_REFUND_PATTERNS), case=False, na=False)
            )
            if "item_gross_sales" in df.columns:
                # Treat valid Hungry Panda refunds as positive sales.
                panda_refunds = refund_mask & keep_refund_mask
                df.loc[panda_refunds, "item_gross_sales"] = df.loc[
                    panda_refunds, "item_gross_sales"
                ].abs()
            df = df[~refund_mask | keep_refund_mask]
        else:
            df = df[~refund_mask]

    return df


def _filter_non_product_items(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-product items like tips or placeholders."""
    df = df.copy()
    if "item_name" not in df.columns:
        return df

    # Remove non-sales line items from mix calculations.
    mask = (
        df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(EXCLUDE_ITEM_PATTERNS), regex=True, na=False)
    )
    return df[~mask]


# --- Aggregations ---
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


def _assign_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Assign channel and in-person subchannel labels."""
    df = df.copy()
    columns = {col.strip().lower(): col for col in df.columns}
    channel_col = columns.get("channel")
    notes_col = columns.get("notes")

    notes = df[notes_col].astype(str) if notes_col else pd.Series("", index=df.index)
    channel = (
        df[channel_col].astype(str) if channel_col else pd.Series("", index=df.index)
    )

    # Priority: HP tags first, then delivery platforms, otherwise in-person.
    hungry_panda = notes.str.contains("|".join(KEEP_REFUND_PATTERNS), case=False, na=False)
    doordash = channel.str.contains("doordash", case=False, na=False)
    ubereats = channel.str.contains("uber", case=False, na=False)
    square_online = channel.str.contains("square online|online", case=False, na=False)

    channel_group = pd.Series("In Person", index=df.index)
    channel_group = channel_group.mask(hungry_panda, "Hungry Panda")
    channel_group = channel_group.mask(doordash, "DoorDash")
    channel_group = channel_group.mask(ubereats, "Uber Eats")
    channel_group = channel_group.mask(square_online, "Square Online")

    other_mask = (
        ~hungry_panda & ~doordash & ~ubereats & ~square_online
        & channel.str.strip().ne("")
        & ~channel.str.contains("mosa tea|kiosk", case=False, na=False)
    )
    channel_group = channel_group.mask(other_mask, "Other")

    # Split in-person orders into kiosk vs counter.
    in_person_channel = pd.Series("Counter", index=df.index)
    kiosk_mask = channel.str.contains("kiosk", case=False, na=False)
    in_person_channel = in_person_channel.mask(kiosk_mask, "Kiosk")
    in_person_channel = in_person_channel.where(channel_group == "In Person", "")

    df["channel_group"] = channel_group
    df["in_person_channel"] = in_person_channel
    return df


def _compute_channel_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute channel-level sales mix for a given window."""
    if df.empty:
        return pd.DataFrame(
            columns=["channel_group", "total_sales", "channel_sales_pct_of_total"]
        )

    channel_mix = (
        df.groupby("channel_group", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = channel_mix["total_sales"].sum()
    if total_sales == 0:
        channel_mix["channel_sales_pct_of_total"] = 0.0
    else:
        channel_mix["channel_sales_pct_of_total"] = (
            channel_mix["total_sales"] / total_sales
        )
    channel_mix = channel_mix.sort_values("total_sales", ascending=False)
    return channel_mix


def _compute_in_person_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute in-person subchannel mix for a given window."""
    if df.empty:
        return pd.DataFrame(
            columns=["in_person_channel", "total_sales", "in_person_sales_pct_of_total"]
        )

    in_person = df[df["channel_group"] == "In Person"]
    if in_person.empty:
        return pd.DataFrame(
            columns=["in_person_channel", "total_sales", "in_person_sales_pct_of_total"]
        )

    in_person_mix = (
        in_person.groupby("in_person_channel", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = in_person_mix["total_sales"].sum()
    if total_sales == 0:
        in_person_mix["in_person_sales_pct_of_total"] = 0.0
    else:
        in_person_mix["in_person_sales_pct_of_total"] = (
            in_person_mix["total_sales"] / total_sales
        )
    in_person_mix = in_person_mix.sort_values("total_sales", ascending=False)
    return in_person_mix


def _build_channel_summary(
    channel_mix: pd.DataFrame,
    in_person_mix: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Build a single summary table for channel + in-person splits."""
    channel_mix = channel_mix.copy()
    channel_mix["scope"] = label
    channel_mix["metric"] = "channel"
    channel_mix = channel_mix.rename(columns={"channel_group": "segment"})
    channel_mix = channel_mix[
        ["scope", "metric", "segment", "total_sales", "channel_sales_pct_of_total"]
    ].rename(columns={"channel_sales_pct_of_total": "sales_pct_of_total"})

    in_person_mix = in_person_mix.copy()
    if in_person_mix.empty:
        in_person_mix = pd.DataFrame(
            columns=["segment", "total_sales", "sales_pct_of_total"]
        )
    else:
        in_person_mix = in_person_mix.rename(
            columns={
                "in_person_channel": "segment",
                "in_person_sales_pct_of_total": "sales_pct_of_total",
            }
        )
    in_person_mix["scope"] = label
    in_person_mix["metric"] = "in_person"
    in_person_mix = in_person_mix[
        ["scope", "metric", "segment", "total_sales", "sales_pct_of_total"]
    ]

    summary = pd.concat([channel_mix, in_person_mix], ignore_index=True)
    return summary


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


def _compute_hourly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly sales distribution for peak-hour analysis."""
    if df.empty:
        return pd.DataFrame(columns=["hour", "total_sales", "sales_pct_of_total"])

    if "order_datetime" not in df.columns:
        raise ValueError("Missing required column: order_datetime")

    hourly = (
        df.assign(hour=df["order_datetime"].dt.hour)
        .groupby("hour", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = hourly["total_sales"].sum()
    if total_sales == 0:
        hourly["sales_pct_of_total"] = 0.0
    else:
        hourly["sales_pct_of_total"] = hourly["total_sales"] / total_sales
    hourly = hourly.sort_values("hour")
    return hourly


# --- Reporting helpers ---
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


# --- Entry point ---
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
    df = _filter_refunds(df)
    df = _filter_non_product_items(df)
    df = _assign_channel(df)

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

    if not df_last_3_months.empty:
        min_3_months = df_last_3_months["order_datetime"].min()
        if min_3_months >= last_month_start:
            print(
                "Warning: last 3 months window has no data before last month; "
                "results may match last month."
            )

    if df_last_month.empty:
        print("Warning: last month window has no rows.")
    if df_last_3_months.empty:
        print("Warning: last 3 months window has no rows.")

    last_month_category = _compute_category_mix(df_last_month)
    last_month_product = _compute_product_mix(df_last_month)
    last_month_channel = _compute_channel_mix(df_last_month)
    last_month_in_person = _compute_in_person_mix(df_last_month)
    last_month_hourly = _compute_hourly_sales(df_last_month)
    last_3_category = _compute_category_mix(df_last_3_months)
    last_3_product = _compute_product_mix(df_last_3_months)
    last_3_channel = _compute_channel_mix(df_last_3_months)
    last_3_in_person = _compute_in_person_mix(df_last_3_months)
    last_3_hourly = _compute_hourly_sales(df_last_3_months)
    global_category = _compute_category_mix(df)
    global_product = _compute_product_mix(df)
    global_channel = _compute_channel_mix(df)
    global_in_person = _compute_in_person_mix(df)
    global_hourly = _compute_hourly_sales(df)

    channel_summary = pd.concat(
        [
            _build_channel_summary(last_month_channel, last_month_in_person, "Last Month"),
            _build_channel_summary(last_3_channel, last_3_in_person, "Last 3 Months"),
            _build_channel_summary(global_channel, global_in_person, "All Data"),
        ],
        ignore_index=True,
    )

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
    if last_month_channel.empty:
        print("Warning: last month channel mix is empty.")
    if last_3_channel.empty:
        print("Warning: last 3 months channel mix is empty.")
    if global_channel.empty:
        print("Warning: global channel mix is empty.")

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
    last_month_channel.to_csv(
        processed_dir / "last_month_channel_mix.csv", index=False
    )
    last_3_channel.to_csv(
        processed_dir / "last_3_months_channel_mix.csv", index=False
    )
    global_channel.to_csv(processed_dir / "global_channel_mix.csv", index=False)
    last_month_in_person.to_csv(
        processed_dir / "last_month_in_person_mix.csv", index=False
    )
    last_3_in_person.to_csv(
        processed_dir / "last_3_months_in_person_mix.csv", index=False
    )
    global_in_person.to_csv(
        processed_dir / "global_in_person_mix.csv", index=False
    )
    last_month_hourly.to_csv(
        processed_dir / "last_month_hourly_sales.csv", index=False
    )
    last_3_hourly.to_csv(
        processed_dir / "last_3_months_hourly_sales.csv", index=False
    )
    global_hourly.to_csv(
        processed_dir / "global_hourly_sales.csv", index=False
    )
    channel_summary.to_csv(
        processed_dir / "channel_summary.csv", index=False
    )

    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    channel_summary_md = channel_summary.copy()
    channel_summary_md["total_sales"] = channel_summary_md["total_sales"].map(
        lambda value: f"{value:,.2f}"
    )
    channel_summary_md["sales_pct_of_total"] = channel_summary_md[
        "sales_pct_of_total"
    ].map(lambda value: f"{value * 100:.1f}%")
    try:
        markdown = channel_summary_md.to_markdown(index=False)
    except ImportError:
        headers = channel_summary_md.columns.tolist()
        rows = channel_summary_md.astype(str).values.tolist()
        markdown_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        markdown_lines += ["| " + " | ".join(row) + " |" for row in rows]
        markdown = "\n".join(markdown_lines)
    (reports_dir / "channel_summary.md").write_text(markdown)

    _print_summary(last_month_category, last_month_product, "Last Month")
    _print_summary(global_category, global_product, "All Data")


if __name__ == "__main__":
    main()
