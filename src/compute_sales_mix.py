"""Compute sales mix by category and product from Square export data."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

try:
    from load_data import load_square_exports
    from config import EXCLUDE_ITEM_PATTERNS, KEEP_REFUND_PATTERNS, FEATURED_ITEM_QUERY
except ImportError:  # pragma: no cover - fallback for package-style imports
    from src.load_data import load_square_exports
    from src.config import EXCLUDE_ITEM_PATTERNS, KEEP_REFUND_PATTERNS, FEATURED_ITEM_QUERY

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
        "modifiers_applied": ["modifiers applied", "modifier", "modifiers"],
        "source": ["source", "order source", "fulfillment source"],
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
    source_col = columns.get("source")
    notes_col = columns.get("notes")
    item_col = columns.get("item_name")

    notes = df[notes_col].astype(str) if notes_col else pd.Series("", index=df.index)
    channel = df[channel_col].astype(str) if channel_col else pd.Series("", index=df.index)
    source = df[source_col].astype(str) if source_col else pd.Series("", index=df.index)
    items = df[item_col].astype(str) if item_col else pd.Series("", index=df.index)

    # Priority: HP tags first, then delivery platforms, otherwise in-person.
    panda_pattern = "|".join(KEEP_REFUND_PATTERNS)
    hungry_panda = notes.str.contains(panda_pattern, case=False, na=False) | items.str.contains(
        panda_pattern, case=False, na=False
    )
    doordash = channel.str.contains("doordash", case=False, na=False) | source.str.contains(
        "doordash", case=False, na=False
    )
    ubereats = channel.str.contains("uber", case=False, na=False) | source.str.contains(
        "uber", case=False, na=False
    )
    square_online = channel.str.contains(
        "square online|online", case=False, na=False
    ) | source.str.contains("square online|online", case=False, na=False)
    source_kiosk = source.str.contains("kiosk", case=False, na=False)
    source_register = source.str.contains("register", case=False, na=False)

    channel_group = pd.Series("In Person", index=df.index)
    channel_group = channel_group.mask(hungry_panda, "Hungry Panda")
    channel_group = channel_group.mask(doordash, "DoorDash")
    channel_group = channel_group.mask(ubereats, "Uber Eats")
    channel_group = channel_group.mask(square_online, "Square Online")
    channel_group = channel_group.mask(source_kiosk | source_register, "In Person")

    other_mask = (
        ~hungry_panda & ~doordash & ~ubereats & ~square_online
        & channel.str.strip().ne("")
        & ~channel.str.contains("mosa tea|kiosk", case=False, na=False)
    )
    channel_group = channel_group.mask(other_mask, "Other")

    # Split in-person orders into kiosk vs counter.
    in_person_channel = pd.Series("Counter", index=df.index)
    kiosk_mask = channel.str.contains("kiosk", case=False, na=False) | source_kiosk
    register_mask = source_register
    in_person_channel = in_person_channel.mask(kiosk_mask, "Kiosk")
    in_person_channel = in_person_channel.mask(register_mask, "Counter")
    in_person_channel = in_person_channel.where(channel_group == "In Person", "")

    df["channel_group"] = channel_group
    df["in_person_channel"] = in_person_channel
    return df


def _assign_tea_base(df: pd.DataFrame) -> pd.DataFrame:
    """Assign tea base labels based on item names and modifiers."""
    df = df.copy()
    columns = {col.strip().lower(): col for col in df.columns}
    item_col = columns.get("item_name")
    modifiers_col = columns.get("modifiers_applied")
    category_col = columns.get("category_name")

    if not item_col:
        df["tea_base"] = "Unknown"
        return df

    item_text = df[item_col].astype(str)
    modifiers_text = (
        df[modifiers_col].astype(str) if modifiers_col else pd.Series("", index=df.index)
    )
    category_text = (
        df[category_col].astype(str) if category_col else pd.Series("", index=df.index)
    )
    combined = (item_text + " " + modifiers_text + " " + category_text).str.lower()

    tea_base = pd.Series("Unknown", index=df.index)

    # Matcha dominates any blend.
    matcha_mask = combined.str.contains("matcha|抹茶", na=False)
    tea_base = tea_base.mask(matcha_mask, "Matcha")

    # Explicit signature overrides.
    item_lower = item_text.str.lower()
    tea_base = tea_base.mask(
        item_lower.str.contains("taiwanese retro", na=False) & (tea_base == "Unknown"),
        "Black",
    )
    tea_base = tea_base.mask(
        item_lower.str.contains("pistachio mist", na=False) & (tea_base == "Unknown"),
        "Genmai Green",
    )
    tea_base = tea_base.mask(
        item_lower.str.contains("brown sugar mist", na=False) & (tea_base == "Unknown"),
        "TGY Oolong",
    )
    tea_base = tea_base.mask(
        item_lower.str.contains("grapefruit bloom", na=False) & (tea_base == "Unknown"),
        "Four Seasons",
    )
    tea_base = tea_base.mask(
        item_lower.str.contains("hot spice", na=False) & (tea_base == "Unknown"),
        "Black",
    )

    genmai_mask = combined.str.contains("genmai|玄米", na=False)
    green_mask = combined.str.contains("green|綠茶|綠", na=False)
    tea_base = tea_base.mask(
        genmai_mask & green_mask & (tea_base == "Unknown"),
        "Genmai Green",
    )
    tea_base = tea_base.mask(
        genmai_mask & ~green_mask & (tea_base == "Unknown"),
        "Genmai Green",
    )

    tea_base = tea_base.mask(
        combined.str.contains("tgy|oolong|tie guan yin|鐵觀音", na=False)
        & (tea_base == "Unknown"),
        "TGY Oolong",
    )
    tea_base = tea_base.mask(
        combined.str.contains("buckwheat|barley|蕎麥", na=False)
        & (tea_base == "Unknown"),
        "Buckwheat Barley",
    )
    tea_base = tea_base.mask(
        combined.str.contains(r"\bblack tea\b|\bblack\b|紅茶|熟成", regex=True, na=False)
        & (tea_base == "Unknown"),
        "Black",
    )
    tea_base = tea_base.mask(green_mask & (tea_base == "Unknown"), "Green")
    tea_base = tea_base.mask(
        combined.str.contains("four seasons|four season|四季", na=False)
        & (tea_base == "Unknown"),
        "Four Seasons",
    )

    df["tea_base"] = tea_base
    return df


def _assign_milk_type(df: pd.DataFrame) -> pd.DataFrame:
    """Assign milk type labels (Milk Tea vs Au Lait) from category names."""
    df = df.copy()
    columns = {col.strip().lower(): col for col in df.columns}
    category_col = columns.get("category_name")
    if not category_col:
        df["milk_type"] = "Unknown"
        return df

    category_text = df[category_col].astype(str).str.lower()
    milk_type = pd.Series("Unknown", index=df.index)
    milk_type = milk_type.mask(
        category_text.str.contains("milk tea", na=False), "Milk Tea"
    )
    milk_type = milk_type.mask(
        category_text.str.contains("au lait", na=False), "Au Lait"
    )
    df["milk_type"] = milk_type
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


def _compute_tea_base_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute tea base sales mix for a given window."""
    if df.empty:
        return pd.DataFrame(
            columns=["tea_base", "total_sales", "tea_base_sales_pct_of_total"]
        )

    if "tea_base" not in df.columns:
        raise ValueError("Missing required column: tea_base")

    base_df = df.copy()
    if "category_name" in base_df.columns:
        base_df = base_df[
            ~base_df["category_name"]
            .astype(str)
            .str.contains("merchandise|周邊小物", case=False, na=False)
        ]

    base_mix = (
        base_df.groupby("tea_base", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = base_mix["total_sales"].sum()
    if total_sales == 0:
        base_mix["tea_base_sales_pct_of_total"] = 0.0
    else:
        base_mix["tea_base_sales_pct_of_total"] = (
            base_mix["total_sales"] / total_sales
        )
    base_mix = base_mix.sort_values("total_sales", ascending=False)
    return base_mix


def _compute_milk_type_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Milk Tea vs Au Lait sales mix for a given window."""
    if df.empty:
        return pd.DataFrame(
            columns=["milk_type", "total_sales", "milk_type_sales_pct_of_total"]
        )

    if "milk_type" not in df.columns:
        raise ValueError("Missing required column: milk_type")

    milk_df = df[df["milk_type"].isin(["Milk Tea", "Au Lait"])]
    if milk_df.empty:
        return pd.DataFrame(
            columns=["milk_type", "total_sales", "milk_type_sales_pct_of_total"]
        )

    mix = (
        milk_df.groupby("milk_type", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = mix["total_sales"].sum()
    if total_sales == 0:
        mix["milk_type_sales_pct_of_total"] = 0.0
    else:
        mix["milk_type_sales_pct_of_total"] = mix["total_sales"] / total_sales
    mix = mix.sort_values("total_sales", ascending=False)
    return mix


def _compute_fresh_fruit_tea_base_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Green vs Four Seasons mix within Fresh Fruit Tea items."""
    if df.empty:
        return pd.DataFrame(
            columns=["tea_base", "total_sales", "tea_base_sales_pct_of_total"]
        )

    if "tea_base" not in df.columns or "item_name" not in df.columns:
        raise ValueError("Missing required column: tea_base or item_name")

    fruit_mask = (
        df["item_name"]
        .astype(str)
        .str.contains("fresh fruit tea", case=False, na=False)
    )
    fruit_df = df[fruit_mask & df["tea_base"].isin(["Green", "Four Seasons"])]
    if fruit_df.empty:
        return pd.DataFrame(
            columns=["tea_base", "total_sales", "tea_base_sales_pct_of_total"]
        )

    mix = (
        fruit_df.groupby("tea_base", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    total_sales = mix["total_sales"].sum()
    if total_sales == 0:
        mix["tea_base_sales_pct_of_total"] = 0.0
    else:
        mix["tea_base_sales_pct_of_total"] = mix["total_sales"] / total_sales
    mix = mix.sort_values("total_sales", ascending=False)
    return mix


def _compute_top_item_by_tea_base(df: pd.DataFrame) -> pd.DataFrame:
    """Compute top-selling item within each tea base."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "tea_base",
                "item_name",
                "total_sales",
                "item_sales_pct_of_base",
            ]
        )

    if "tea_base" not in df.columns or "item_name" not in df.columns:
        raise ValueError("Missing required column: tea_base or item_name")

    grouped = (
        df.groupby(["tea_base", "item_name"], dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    base_totals = (
        grouped.groupby("tea_base", dropna=False)["total_sales"]
        .sum()
        .reset_index()
        .rename(columns={"total_sales": "base_total_sales"})
    )
    grouped = grouped.merge(base_totals, on="tea_base", how="left")
    grouped["item_sales_pct_of_base"] = grouped.apply(
        lambda row: 0.0
        if row["base_total_sales"] == 0
        else row["total_sales"] / row["base_total_sales"],
        axis=1,
    )
    grouped = grouped.sort_values(
        ["tea_base", "total_sales"], ascending=[True, False]
    )
    top_items = grouped.groupby("tea_base", dropna=False).head(1)
    top_items = top_items.drop(columns=["base_total_sales"])
    return top_items


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


def _compute_top_products_with_other(
    df: pd.DataFrame, top_n: int = 24
) -> pd.DataFrame:
    """Compute top-N products with an 'Other' bucket for the remainder."""
    if df.empty:
        return pd.DataFrame(
            columns=["item_name", "total_sales", "product_sales_pct_of_total"]
        )

    if "item_name" not in df.columns:
        raise ValueError("Missing required column: item_name")

    totals = (
        df.groupby("item_name", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    totals = totals.sort_values("total_sales", ascending=False)
    top = totals.head(top_n).copy()
    other_total = totals["total_sales"].sum() - top["total_sales"].sum()
    if other_total > 0:
        top = pd.concat(
            [
                top,
                pd.DataFrame(
                    [{"item_name": "Other", "total_sales": other_total}]
                ),
            ],
            ignore_index=True,
        )
    total_sales = top["total_sales"].sum()
    top["product_sales_pct_of_total"] = (
        top["total_sales"] / total_sales if total_sales else 0.0
    )
    return top


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


def _compute_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily total sales."""
    if df.empty:
        return pd.DataFrame(columns=["date", "total_sales"])

    if "order_datetime" not in df.columns:
        raise ValueError("Missing required column: order_datetime")

    daily = (
        df.assign(date=df["order_datetime"].dt.date)
        .groupby("date", dropna=False)["item_gross_sales"]
        .sum()
        .reset_index()
        .rename(columns={"item_gross_sales": "total_sales"})
    )
    daily = daily.sort_values("date")
    return daily


def _compute_daily_sales_zscore(daily_sales: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores for daily sales against weekday baselines."""
    if daily_sales.empty:
        return pd.DataFrame(
            columns=["date", "total_sales", "weekday", "baseline_mean", "baseline_std", "z_score"]
        )

    daily = daily_sales.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["weekday"] = daily["date"].dt.day_name()

    stats = (
        daily.groupby("weekday", dropna=False)["total_sales"]
        .agg(baseline_mean="mean", baseline_std="std")
        .reset_index()
    )
    daily = daily.merge(stats, on="weekday", how="left")
    daily["z_score"] = daily.apply(
        lambda row: 0.0
        if row["baseline_std"] == 0 or pd.isna(row["baseline_std"])
        else (row["total_sales"] - row["baseline_mean"]) / row["baseline_std"],
        axis=1,
    )
    return daily


def _compute_daily_anomalies_by_threshold(
    daily_sales_zscore: pd.DataFrame, threshold: float = 2.25
) -> pd.DataFrame:
    """Select daily anomalies where abs(z-score) exceeds threshold."""
    if daily_sales_zscore.empty:
        return pd.DataFrame(
            columns=["date", "total_sales", "weekday", "baseline_mean", "baseline_std", "z_score"]
        )

    anomalies = daily_sales_zscore.copy()
    anomalies = anomalies[anomalies["z_score"].abs() >= threshold]
    anomalies = anomalies.sort_values("z_score", ascending=False)
    return anomalies


def _compute_top_daily_anomalies_by_abs_zscore(
    daily_sales_zscore: pd.DataFrame, top_n: int = 10
) -> pd.DataFrame:
    """Select top anomalies by absolute z-score."""
    if daily_sales_zscore.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "total_sales",
                "weekday",
                "baseline_mean",
                "baseline_std",
                "z_score",
                "abs_z_score",
            ]
        )

    ranked = daily_sales_zscore.copy()
    ranked["abs_z_score"] = ranked["z_score"].abs()
    ranked = ranked.sort_values("abs_z_score", ascending=False).head(top_n)
    return ranked


def _compute_daily_sales_rolling_zscore(
    daily_sales: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """Compute rolling z-scores for daily sales."""
    if daily_sales.empty:
        return pd.DataFrame(columns=["date", "total_sales", "rolling_mean", "rolling_std", "z_score"])

    daily = daily_sales.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["rolling_mean"] = daily["total_sales"].rolling(window=window, min_periods=3).mean()
    daily["rolling_std"] = daily["total_sales"].rolling(window=window, min_periods=3).std()
    daily["z_score"] = daily.apply(
        lambda row: 0.0
        if row["rolling_std"] == 0 or pd.isna(row["rolling_std"])
        else (row["total_sales"] - row["rolling_mean"]) / row["rolling_std"],
        axis=1,
    )
    return daily


def _compute_daily_sales_robust_zscore(
    daily_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Compute robust z-scores using median and MAD."""
    if daily_sales.empty:
        return pd.DataFrame(columns=["date", "total_sales", "median", "mad", "z_score"])

    daily = daily_sales.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    median = daily["total_sales"].median()
    mad = (daily["total_sales"] - median).abs().median()
    daily["median"] = median
    daily["mad"] = mad
    daily["z_score"] = daily.apply(
        lambda row: 0.0 if mad == 0 or pd.isna(mad) else (row["total_sales"] - median) / mad,
        axis=1,
    )
    return daily


def _extract_pct(modifiers: pd.Series, label: str) -> pd.Series:
    """Extract percent value for a modifier label (e.g., 'Sugar' or 'Ice')."""
    pattern = rf"(\d+)%\s*{label}"
    pct = modifiers.str.extract(pattern, expand=False)
    pct = pct.fillna("")
    no_label = modifiers.str.contains(rf"\bNo\s*{label}\b", case=False, na=False)
    pct = pct.mask(no_label, "0")
    return pct.replace("", pd.NA)


def _compute_modifier_pct_mix(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compute percent mix for a modifier label."""
    if df.empty:
        return pd.DataFrame(columns=[f"{label.lower()}_pct", "count", "share"])

    if "modifiers_applied" not in df.columns:
        raise ValueError("Missing required column: modifiers_applied")

    modifiers = df["modifiers_applied"].astype(str)
    pct = _extract_pct(modifiers, label)
    pct = pct.dropna()
    if pct.empty:
        return pd.DataFrame(columns=[f"{label.lower()}_pct", "count", "share"])

    mix = (
        pct.value_counts()
        .rename_axis(f"{label.lower()}_pct")
        .reset_index(name="count")
    )
    mix["share"] = mix["count"] / mix["count"].sum()
    mix[f"{label.lower()}_pct"] = mix[f"{label.lower()}_pct"].astype(int)
    mix = mix.sort_values(f"{label.lower()}_pct")
    return mix


def _extract_toppings(modifiers: pd.Series, item_names: pd.Series) -> pd.Series:
    """Extract topping names from modifiers applied and item defaults."""
    if modifiers.empty:
        return pd.Series(dtype=str)

    tokens = modifiers.fillna("").astype(str).str.split(",")
    toppings = tokens.explode().str.strip()
    toppings = toppings[toppings.ne("")]

    lowered = toppings.str.lower()
    is_sugar = lowered.str.contains("sugar", na=False)
    is_ice = lowered.str.contains("ice", na=False)
    is_jelly = lowered.str.contains("jelly", na=False)
    is_foam = lowered.str.contains("foam", na=False)
    is_boba = lowered.str.contains("boba", na=False)
    is_tea = lowered.str.contains("tea", na=False) & ~is_jelly
    is_base = lowered.str.contains("genmai|oat milk|extra matcha|no chestnut", na=False)
    is_multiplier = lowered.str.contains(r"\bx\\d+\b|×", na=False)
    is_sugar_level = is_sugar & ~(is_jelly | is_foam | is_boba)

    toppings = toppings[~(is_sugar_level | is_ice | is_tea | is_base | is_multiplier)]

    normalized = toppings.copy()
    normalized = normalized.str.replace(
        "Osmanthus Tie Guan Yin Jelly", "Tea Jelly", regex=False
    )
    normalized = normalized.str.replace(
        "Tie Guan Yin Oolong Jelly", "Tea Jelly", regex=False
    )
    normalized = normalized.str.replace(
        r"Brown Sugar.*Tapioca Jelly", "HK Jelly", regex=True
    )
    normalized = normalized.str.replace(
        r"Brown Sugar.*Jelly", "HK Jelly", regex=True
    )
    normalized = normalized.str.replace(r"HK Jelly\)+", "HK Jelly", regex=True)
    normalized = normalized.str.replace(
        "Hún-Kué (Tapioca Jelly)", "HK Jelly", regex=False
    )
    normalized = normalized.str.replace("HK Jelly", "HK Jelly", regex=False)
    normalized = normalized.str.replace("Boba", "Boba", regex=False)
    normalized = normalized.str.replace("Cream Foam", "Cream Foam", regex=False)
    normalized = normalized.str.replace("Brown Sugar Cream Foam", "Cream Foam", regex=False)

    item_lower = item_names.fillna("").astype(str).str.lower()
    has_taiwanese_retro = item_lower.str.contains("taiwanese retro|珍珠", na=False)
    has_grapefruit_bloom = item_lower.str.contains("grapefruit bloom|柚香", na=False)
    has_tgy_special = item_lower.str.contains("tgy special|鐵觀音奶茶", na=False)
    extras = []
    if has_taiwanese_retro.any():
        extras.extend(["Boba"] * has_taiwanese_retro.sum())
    if has_grapefruit_bloom.any():
        extras.extend(["Tea Jelly"] * has_grapefruit_bloom.sum())
    if has_tgy_special.any():
        extras.extend(["HK Jelly"] * has_tgy_special.sum())
    has_genmai_matcha_jelly = item_lower.str.contains("genmai matcha", na=False)
    if has_genmai_matcha_jelly.any():
        extras.extend(["Daily Jelly (Matcha Jelly)"] * has_genmai_matcha_jelly.sum())
    has_pistachio_mist = item_lower.str.contains("pistachio mist", na=False)
    if has_pistachio_mist.any():
        extras.extend(["Pistachio Foam"] * has_pistachio_mist.sum())
    has_brown_sugar_mist = item_lower.str.contains("brown sugar mist", na=False)
    if has_brown_sugar_mist.any():
        extras.extend(["Cream Foam"] * has_brown_sugar_mist.sum())
    if extras:
        normalized = pd.concat([normalized, pd.Series(extras)], ignore_index=True)

    return normalized


def _compute_topping_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute topping popularity from modifiers."""
    if df.empty:
        return pd.DataFrame(columns=["topping", "count", "share_of_toppings"])

    if "modifiers_applied" not in df.columns or "item_name" not in df.columns:
        raise ValueError("Missing required column: modifiers_applied or item_name")

    toppings = _extract_toppings(df["modifiers_applied"], df["item_name"])
    if toppings.empty:
        return pd.DataFrame(columns=["topping", "count", "share_of_toppings"])

    counts = toppings.value_counts().rename_axis("topping").reset_index(name="count")
    total = counts["count"].sum()
    counts["share_of_toppings"] = counts["count"] / total if total else 0.0
    return counts


def _compute_item_pair_stats(
    df: pd.DataFrame,
    min_support: float = 0.005,
    min_lift: float = 1.5,
    max_basket_size: int = 6,
) -> pd.DataFrame:
    """Compute item pair co-purchase stats (support, confidence, lift)."""
    if df.empty:
        return pd.DataFrame(
            columns=["item_a", "item_b", "count", "support", "confidence", "lift"]
        )

    required_cols = {"order_id", "item_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    order_sales = df.groupby("order_id", dropna=False)["item_gross_sales"].sum().to_dict()
    basket = (
        df.groupby("order_id", dropna=False)["item_name"]
        .apply(lambda items: sorted(set(map(str, items))))
        .reset_index()
    )
    basket = basket[basket["item_name"].map(len) >= 2]
    basket = basket[basket["item_name"].map(len) <= max_basket_size]
    total_orders = len(basket)
    if total_orders == 0:
        return pd.DataFrame(
            columns=[
                "item_a",
                "item_b",
                "count",
                "support",
                "confidence",
                "lift",
                "pair_sales",
                "pair_sales_pct_of_total",
                "total_transactions",
            ]
        )

    item_counts = {}
    pair_counts = {}
    pair_sales = {}
    for order_id, items in basket.itertuples(index=False):
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = (items[i], items[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                pair_sales[pair] = pair_sales.get(pair, 0.0) + order_sales.get(order_id, 0.0)

    rows = []
    total_sales = df["item_gross_sales"].sum()
    for (item_a, item_b), count in pair_counts.items():
        support = count / total_orders
        if support < min_support:
            continue
        conf = count / item_counts[item_a] if item_counts[item_a] else 0.0
        lift = conf / (item_counts[item_b] / total_orders) if item_counts[item_b] else 0.0
        if lift < min_lift:
            continue
        sales = pair_sales.get((item_a, item_b), 0.0)
        sales_pct = sales / total_sales if total_sales else 0.0
        rows.append(
            {
                "item_a": item_a,
                "item_b": item_b,
                "count": count,
                "support": support,
                "confidence": conf,
                "lift": lift,
                "pair_sales": sales,
                "pair_sales_pct_of_total": sales_pct,
                "total_transactions": total_orders,
            }
        )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.sort_values(["lift", "support"], ascending=False)
    return pairs


def _compute_item_hourly_sales(df: pd.DataFrame, item_query: str) -> pd.DataFrame:
    """Compute hourly sales for a specific item name query."""
    if df.empty:
        return pd.DataFrame(columns=["hour", "total_sales", "sales_pct_of_total"])

    if "item_name" not in df.columns:
        raise ValueError("Missing required column: item_name")

    item_mask = (
        df["item_name"]
        .astype(str)
        .str.contains(item_query, case=False, na=False)
    )
    item_df = df[item_mask]
    return _compute_hourly_sales(item_df)


def _compute_weekday_weekend_hourly(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute hourly sales for weekday vs weekend business hours."""
    if df.empty:
        empty = pd.DataFrame(columns=["hour", "total_sales", "sales_pct_of_total"])
        return empty, empty

    weekday_df = df[df["order_datetime"].dt.dayofweek.isin([0, 1, 2, 3])]
    weekend_df = df[df["order_datetime"].dt.dayofweek.isin([4, 5, 6])]

    weekday_df = weekday_df[weekday_df["order_datetime"].dt.hour.between(12, 20)]
    weekend_df = weekend_df[weekend_df["order_datetime"].dt.hour.between(11, 21)]

    return _compute_hourly_sales(weekday_df), _compute_hourly_sales(weekend_df)


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
    private_dir = base_dir / "data" / "private"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and standardize raw export data.
    df = load_square_exports(raw_dir)
    df = _normalize_columns(df)
    df = _build_order_datetime(df)

    channel_df = None
    channel_source_path = private_dir / "channelmix-raw.csv"
    if channel_source_path.exists():
        channel_df = pd.read_csv(channel_source_path)
        channel_df = _normalize_columns(channel_df)
        channel_df = _build_order_datetime(channel_df)

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
    df = _assign_tea_base(df)
    df = _assign_milk_type(df)

    if channel_df is not None:
        channel_df = _coerce_sales(channel_df)
        channel_df = _filter_refunds(channel_df)
        channel_df = _filter_non_product_items(channel_df)
        channel_df = _assign_channel(channel_df)

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

    channel_last_month = df_last_month
    channel_last_3_months = df_last_3_months
    channel_global = df
    if channel_df is not None:
        channel_last_month = channel_df[
            (channel_df["order_datetime"] >= last_month_start)
            & (channel_df["order_datetime"] <= last_month_end)
        ]
        channel_last_3_months = channel_df[
            (channel_df["order_datetime"] >= last_3_start)
            & (channel_df["order_datetime"] <= last_month_end)
        ]
        channel_global = channel_df

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
    last_month_top25_products = _compute_top_products_with_other(df_last_month, top_n=24)
    last_month_channel = _compute_channel_mix(channel_last_month)
    last_month_in_person = _compute_in_person_mix(channel_last_month)
    last_month_tea_base = _compute_tea_base_mix(df_last_month)
    last_month_milk_type = _compute_milk_type_mix(df_last_month)
    last_month_fresh_fruit_tea_base = _compute_fresh_fruit_tea_base_mix(df_last_month)
    last_month_top_item_by_tea_base = _compute_top_item_by_tea_base(df_last_month)
    last_month_hourly = _compute_hourly_sales(df_last_month)
    last_month_weekday_hourly, last_month_weekend_hourly = _compute_weekday_weekend_hourly(
        df_last_month
    )
    last_3_category = _compute_category_mix(df_last_3_months)
    last_3_product = _compute_product_mix(df_last_3_months)
    last_3_top25_products = _compute_top_products_with_other(df_last_3_months, top_n=24)
    last_3_channel = _compute_channel_mix(channel_last_3_months)
    last_3_in_person = _compute_in_person_mix(channel_last_3_months)
    last_3_tea_base = _compute_tea_base_mix(df_last_3_months)
    last_3_milk_type = _compute_milk_type_mix(df_last_3_months)
    last_3_fresh_fruit_tea_base = _compute_fresh_fruit_tea_base_mix(df_last_3_months)
    last_3_top_item_by_tea_base = _compute_top_item_by_tea_base(df_last_3_months)
    last_3_topping_popularity = _compute_topping_popularity(df_last_3_months)
    last_3_order_count = df_last_3_months["order_id"].nunique()
    last_3_hourly = _compute_hourly_sales(df_last_3_months)
    last_3_weekday_hourly, last_3_weekend_hourly = _compute_weekday_weekend_hourly(
        df_last_3_months
    )
    global_category = _compute_category_mix(df)
    global_product = _compute_product_mix(df)
    global_channel = _compute_channel_mix(channel_global)
    global_in_person = _compute_in_person_mix(channel_global)
    global_tea_base = _compute_tea_base_mix(df)
    global_milk_type = _compute_milk_type_mix(df)
    global_fresh_fruit_tea_base = _compute_fresh_fruit_tea_base_mix(df)
    global_top_item_by_tea_base = _compute_top_item_by_tea_base(df)
    global_daily_sales = _compute_daily_sales(df)
    global_daily_sales_zscore = _compute_daily_sales_zscore(global_daily_sales)
    global_daily_sales_anomalies = _compute_daily_anomalies_by_threshold(
        global_daily_sales_zscore
    )
    global_daily_sales_top10 = _compute_top_daily_anomalies_by_abs_zscore(
        global_daily_sales_zscore
    )
    global_daily_sales_rolling = _compute_daily_sales_rolling_zscore(global_daily_sales)
    global_daily_sales_robust = _compute_daily_sales_robust_zscore(global_daily_sales)
    global_sugar_pct = _compute_modifier_pct_mix(df, "Sugar")
    global_ice_pct = _compute_modifier_pct_mix(df, "Ice")
    last_3_item_pair_stats = _compute_item_pair_stats(df_last_3_months)
    last_3_item_pair_top10 = last_3_item_pair_stats.head(10)
    global_hourly = _compute_hourly_sales(df)
    global_weekday_hourly, global_weekend_hourly = _compute_weekday_weekend_hourly(df)
    last_month_featured_item_hourly = _compute_item_hourly_sales(
        df_last_month, FEATURED_ITEM_QUERY
    )

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
    last_month_top25_products.to_csv(
        processed_dir / "last_month_top_25_products_with_other.csv", index=False
    )
    last_3_category.to_csv(
        processed_dir / "last_3_months_category_mix.csv", index=False
    )
    last_3_product.to_csv(
        processed_dir / "last_3_months_product_mix.csv", index=False
    )
    last_3_top25_products.to_csv(
        processed_dir / "last_3_months_top_25_products_with_other.csv", index=False
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
    last_month_tea_base.to_csv(
        processed_dir / "last_month_tea_base_mix.csv", index=False
    )
    last_month_milk_type.to_csv(
        processed_dir / "last_month_milk_type_mix.csv", index=False
    )
    last_month_fresh_fruit_tea_base.to_csv(
        processed_dir / "last_month_fresh_fruit_tea_base_mix.csv", index=False
    )
    last_month_top_item_by_tea_base.to_csv(
        processed_dir / "last_month_top_item_by_tea_base.csv", index=False
    )
    last_3_in_person.to_csv(
        processed_dir / "last_3_months_in_person_mix.csv", index=False
    )
    last_3_tea_base.to_csv(
        processed_dir / "last_3_months_tea_base_mix.csv", index=False
    )
    last_3_milk_type.to_csv(
        processed_dir / "last_3_months_milk_type_mix.csv", index=False
    )
    last_3_fresh_fruit_tea_base.to_csv(
        processed_dir / "last_3_months_fresh_fruit_tea_base_mix.csv", index=False
    )
    last_3_top_item_by_tea_base.to_csv(
        processed_dir / "last_3_months_top_item_by_tea_base.csv", index=False
    )
    last_3_topping_popularity.to_csv(
        processed_dir / "last_3_months_topping_popularity.csv", index=False
    )
    pd.DataFrame(
        [{"metric": "last_3_months_order_count", "value": last_3_order_count}]
    ).to_csv(processed_dir / "last_3_months_order_count.csv", index=False)
    global_in_person.to_csv(
        processed_dir / "global_in_person_mix.csv", index=False
    )
    global_tea_base.to_csv(
        processed_dir / "global_tea_base_mix.csv", index=False
    )
    global_milk_type.to_csv(
        processed_dir / "global_milk_type_mix.csv", index=False
    )
    global_fresh_fruit_tea_base.to_csv(
        processed_dir / "global_fresh_fruit_tea_base_mix.csv", index=False
    )
    global_top_item_by_tea_base.to_csv(
        processed_dir / "global_top_item_by_tea_base.csv", index=False
    )
    global_daily_sales.to_csv(
        processed_dir / "global_daily_sales.csv", index=False
    )
    global_daily_sales_zscore.to_csv(
        processed_dir / "global_daily_sales_zscore.csv", index=False
    )
    global_daily_sales_anomalies.to_csv(
        processed_dir / "global_daily_sales_anomalies.csv", index=False
    )
    global_daily_sales_top10.to_csv(
        processed_dir / "global_daily_sales_top10_anomalies.csv", index=False
    )
    global_daily_sales_rolling.to_csv(
        processed_dir / "global_daily_sales_rolling_zscore.csv", index=False
    )
    global_daily_sales_robust.to_csv(
        processed_dir / "global_daily_sales_robust_zscore.csv", index=False
    )
    global_sugar_pct.to_csv(
        processed_dir / "global_sugar_pct_mix.csv", index=False
    )
    global_ice_pct.to_csv(
        processed_dir / "global_ice_pct_mix.csv", index=False
    )
    last_3_item_pair_stats.to_csv(
        processed_dir / "last_3_months_item_pair_stats.csv", index=False
    )
    last_3_item_pair_top10.to_csv(
        processed_dir / "last_3_months_item_pair_top10.csv", index=False
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
    last_month_weekday_hourly.to_csv(
        processed_dir / "last_month_weekday_hourly_sales.csv", index=False
    )
    last_month_weekend_hourly.to_csv(
        processed_dir / "last_month_weekend_hourly_sales.csv", index=False
    )
    last_3_weekday_hourly.to_csv(
        processed_dir / "last_3_months_weekday_hourly_sales.csv", index=False
    )
    last_3_weekend_hourly.to_csv(
        processed_dir / "last_3_months_weekend_hourly_sales.csv", index=False
    )
    global_weekday_hourly.to_csv(
        processed_dir / "global_weekday_hourly_sales.csv", index=False
    )
    global_weekend_hourly.to_csv(
        processed_dir / "global_weekend_hourly_sales.csv", index=False
    )
    last_month_featured_item_hourly.to_csv(
        processed_dir / "last_month_featured_item_hourly_sales.csv", index=False
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
