"""Generate PNG figures from processed sales mix outputs."""
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd

from config import EXCLUDE_ITEM_PATTERNS


# --- Formatting helpers ---
def _format_pct(value: float) -> str:
    """Format a decimal percent for axis labels."""
    return f"{value * 100:.1f}%"


def _format_currency(value: float) -> str:
    """Format currency values for axis labels."""
    return f"${value:,.2f}"


def _set_cjk_font() -> str | None:
    """Set a CJK-capable font if available; return the chosen font."""
    preferred_fonts = [
        "PingFang SC",
        "PingFang TC",
        "Heiti SC",
        "STHeiti",
        "Hiragino Sans GB",
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Noto Sans SC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            return font_name
    return None


# --- Product mix figures ---
def generate_product_mix_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a horizontal bar chart for a product mix file."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed product mix is empty; no figure generated.")

    df = df[
        ~df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(EXCLUDE_ITEM_PATTERNS), regex=True, na=False)
    ]

    df = df.sort_values("product_sales_pct_of_total", ascending=True)
    df["item_label"] = df["item_name"].fillna("Unknown Item")
    df["category_label"] = df["category_name"].fillna("Uncategorized")
    duplicate_items = df["item_label"].duplicated(keep=False)
    df["label"] = df["item_label"]
    df.loc[duplicate_items, "label"] = (
        df.loc[duplicate_items, "item_label"]
        + " ("
        + df.loc[duplicate_items, "category_label"]
        + ")"
    )

    categories = sorted(df["category_label"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}
    colors = df["category_label"].map(color_map)

    fig_height = max(4, min(20, 0.3 * len(df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(df["label"], df["product_sales_pct_of_total"], color=colors)
    ax.set_title(title, pad=4)
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel("Product")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.bar_label(bars, labels=[_format_pct(v) for v in df["product_sales_pct_of_total"]], padding=3)
    max_pct = df["product_sales_pct_of_total"].max()
    ax.set_xlim(0, max_pct * 1.15)

    handles = [
        plt.Line2D([0], [0], color=color_map[cat], lw=6, label=cat)
        for cat in categories
    ]
    ax.legend(
        handles=handles,
        title="Category",
        loc="upper left",
        frameon=False,
    )

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "items"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_top_products_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
    top_n: int = 10,
) -> Path:
    """Create a horizontal bar chart for top-N products."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed product mix is empty; no figure generated.")

    df = df[
        ~df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(EXCLUDE_ITEM_PATTERNS), regex=True, na=False)
    ]

    df = df.sort_values("product_sales_pct_of_total", ascending=False).head(top_n)
    df = df.sort_values("product_sales_pct_of_total", ascending=True)
    df["item_label"] = df["item_name"].fillna("Unknown Item")
    df["category_label"] = df["category_name"].fillna("Uncategorized")
    duplicate_items = df["item_label"].duplicated(keep=False)
    df["label"] = df["item_label"]
    df.loc[duplicate_items, "label"] = (
        df.loc[duplicate_items, "item_label"]
        + " ("
        + df.loc[duplicate_items, "category_label"]
        + ")"
    )

    categories = sorted(df["category_label"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {cat: cmap(i % cmap.N) for i, cat in enumerate(categories)}
    colors = df["category_label"].map(color_map)

    fig_height = max(4, min(12, 0.5 * len(df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(df["label"], df["product_sales_pct_of_total"], color=colors)
    ax.set_title(title, pad=4)
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel("Product")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.bar_label(bars, labels=[_format_pct(v) for v in df["product_sales_pct_of_total"]], padding=3)
    max_pct = df["product_sales_pct_of_total"].max()
    ax.set_xlim(0, max_pct * 1.15)

    handles = [
        plt.Line2D([0], [0], color=color_map[cat], lw=6, label=cat)
        for cat in categories
    ]
    ax.legend(
        handles=handles,
        title="Category",
        loc="lower right",
        frameon=False,
    )

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "items"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# --- Category mix figures ---
def generate_category_mix_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a horizontal bar chart for category mix."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed category mix is empty; no figure generated.")

    df = df.sort_values("category_sales_pct_of_total", ascending=True)
    df["label"] = df["category_name"].fillna("Uncategorized")

    fig_height = max(4, min(10, 0.5 * len(df)))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    bars = ax.barh(df["label"], df["category_sales_pct_of_total"], color="#2A6F8F")
    ax.set_title(title, pad=4)
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel("Category")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.bar_label(bars, labels=[_format_pct(v) for v in df["category_sales_pct_of_total"]], padding=3)
    max_pct = df["category_sales_pct_of_total"].max()
    ax.set_xlim(0, max_pct * 1.15)

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "drink_share"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# --- Pareto figure ---
def generate_pareto_products_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a Pareto chart for product sales mix."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed product mix is empty; no figure generated.")

    df = df[
        ~df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(EXCLUDE_ITEM_PATTERNS), regex=True, na=False)
    ]

    df = df.sort_values("product_sales_pct_of_total", ascending=False).reset_index(drop=True)
    df["item_label"] = df["item_name"].fillna("Unknown Item")
    df["category_label"] = df["category_name"].fillna("Uncategorized")
    duplicate_items = df["item_label"].duplicated(keep=False)
    df["label"] = df["item_label"]
    df.loc[duplicate_items, "label"] = (
        df.loc[duplicate_items, "item_label"]
        + " ("
        + df.loc[duplicate_items, "category_label"]
        + ")"
    )

    df["cumulative_pct"] = df["product_sales_pct_of_total"].cumsum()
    x = range(len(df))

    fig_height = max(5, min(12, 0.22 * len(df)))
    fig, ax1 = plt.subplots(figsize=(12, fig_height))
    bars = ax1.bar(x, df["product_sales_pct_of_total"], color="#2A6F8F")
    ax1.set_title(title, pad=4)
    ax1.set_ylabel("Percent of Total Sales")
    ax1.set_ylim(0, df["product_sales_pct_of_total"].max() * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["label"], rotation=75, ha="right", fontsize=8)

    for label in ax1.get_xticklabels():
        label.set_fontweight("bold")

    ax1.set_yticklabels([_format_pct(tick) for tick in ax1.get_yticks()])

    ax2 = ax1.twinx()
    ax2.plot(x, df["cumulative_pct"], color="#D17A00", linewidth=2)
    ax2.set_ylabel("Cumulative Percent of Total Sales")
    ax2.set_ylim(0, 1.05)
    ax2.set_yticklabels([_format_pct(tick) for tick in ax2.get_yticks()])

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "items"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# --- Donut figure ---
def generate_category_share_donut(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a donut chart for category share."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed category mix is empty; no figure generated.")

    df = df.sort_values("category_sales_pct_of_total", ascending=False)
    labels = df["category_name"].fillna("Uncategorized")
    values = df["category_sales_pct_of_total"]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.4, "edgecolor": "white"},
        textprops={"fontsize": 9},
        pctdistance=0.75,
    )
    ax.set_title(title, pad=8)
    ax.axis("equal")

    for text in autotexts:
        text.set_fontweight("bold")
        text.set_color("#1F2937")

    ax.legend(
        wedges,
        labels,
        title="Category",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "drink_share"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# --- Channel mix figures ---
def generate_channel_mix_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
    label_column: str,
    pct_column: str,
    y_label: str = "Channel",
    color_map: dict[str, str] | None = None,
    fallback_color: str = "#4C7EA8",
) -> Path:
    """Create a horizontal bar chart for channel mix files."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed channel mix is empty; no figure generated.")

    df = df.sort_values(pct_column, ascending=True)
    df["label"] = df[label_column].fillna("Unknown")

    fig_height = max(3.5, min(8, 0.6 * len(df)))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    if color_map:
        colors = df["label"].map(color_map).fillna(fallback_color)
    else:
        colors = fallback_color
    bars = ax.barh(df["label"], df[pct_column], color=colors)
    ax.set_title(title, pad=4)
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel(y_label)

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.bar_label(bars, labels=[_format_pct(v) for v in df[pct_column]], padding=3)
    max_pct = df[pct_column].max()
    ax.set_xlim(0, max_pct * 1.15)

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "drink_share"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_tea_base_mix_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a horizontal bar chart for tea base mix."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed tea base mix is empty; no figure generated.")

    df = df.sort_values("tea_base_sales_pct_of_total", ascending=True)
    df["label"] = df["tea_base"].fillna("Unknown")

    fig_height = max(3.5, min(8, 0.6 * len(df)))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(df["label"], df["tea_base_sales_pct_of_total"], color="#4B7B9B")
    ax.set_title(title, pad=4)
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel("Tea Base")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.bar_label(bars, labels=[_format_pct(v) for v in df["tea_base_sales_pct_of_total"]], padding=3)
    max_pct = df["tea_base_sales_pct_of_total"].max()
    ax.set_xlim(0, max_pct * 1.15)

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "tea_base"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_peak_hours_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
    y_label: str = "Percent of Total Sales",
    bar_color: str = "#2F6F5E",
) -> Path:
    """Create a bar chart for hourly sales share."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed hourly sales is empty; no figure generated.")

    df = df.sort_values("hour")
    def _format_hour_12h(hour: int) -> str:
        hour_int = int(hour)
        suffix = "AM" if hour_int < 12 else "PM"
        hour_12 = hour_int % 12 or 12
        return f"{hour_12}:00 {suffix}"

    df["hour_label"] = df["hour"].apply(_format_hour_12h)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(df["hour_label"], df["sales_pct_of_total"], color=bar_color)
    ax.set_title(title, pad=6)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(y_label)

    ticks = ax.get_yticks()
    ax.set_yticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.bar_label(bars, labels=[_format_pct(v) for v in df["sales_pct_of_total"]], padding=2, fontsize=8)
    ax.set_ylim(0, df["sales_pct_of_total"].max() * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "drink_share"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_top_item_by_tea_base_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a bar chart showing top item by tea base."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    chosen_font = _set_cjk_font()
    if chosen_font is None:
        print(
            "Warning: no CJK font found; Chinese characters may not render. "
            "Install a font like Noto Sans CJK SC."
        )

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed top item by tea base is empty; no figure generated.")

    df = df.sort_values("total_sales", ascending=True)
    df["tea_base_label"] = df["tea_base"].fillna("Unknown")
    df["item_label"] = df["item_name"].fillna("Unknown Item")
    df["pct_label"] = df["item_sales_pct_of_base"].apply(_format_pct)

    fig_height = max(4, min(10, 0.6 * len(df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(df["tea_base_label"], df["total_sales"], color="#4B7B9B")
    ax.set_title(title, pad=4)
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Tea Base")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_currency(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    max_sales = df["total_sales"].max()
    for bar, item, pct in zip(bars, df["item_label"], df["pct_label"]):
        ax.text(
            bar.get_width() + max_sales * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{item} ({pct})",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(0, max_sales * 1.25)
    fig.tight_layout()

    figures_dir = base_dir / "figures" / "tea_base"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_daily_sales_anomalies_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
    threshold: float = 2.25,
) -> Path:
    """Create a scatter plot of daily sales z-scores."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed daily sales z-score is empty; no figure generated.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(df["date"], df["z_score"], color="#6E7656", s=30, alpha=0.8)
    ax.axhline(0, color="#9CA3AF", linewidth=1)
    ax.axhline(threshold, color="#D17A00", linestyle="--", linewidth=1)
    ax.axhline(-threshold, color="#D17A00", linestyle="--", linewidth=1)

    anomaly_mask = df["z_score"].abs() >= threshold
    if anomaly_mask.any():
        for _, row in df[anomaly_mask].iterrows():
            ax.annotate(
                row["date"].strftime("%Y-%m-%d"),
                (row["date"], row["z_score"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="#D17A00",
            )

    ax.set_title(title, pad=6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Z-Score (Daily Sales vs Weekday Baseline)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "anomaly_detection"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_pct_mix_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
    pct_label: str,
    bar_color: str,
    order_count: int | None = None,
) -> Path:
    """Create a bar chart for sugar/ice percent mix."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed percent mix is empty; no figure generated.")

    df = df.sort_values(pct_label)
    labels = df[pct_label].astype(int).astype(str) + "%"

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, df["share"], color=bar_color)
    ax.set_title(title, pad=6)
    ax.set_xlabel("Percent")
    ax.set_ylabel("Share of Orders")

    ticks = ax.get_yticks()
    ax.set_yticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.bar_label(bars, labels=[_format_pct(v) for v in df["share"]], padding=2, fontsize=8)
    ax.set_ylim(0, df["share"].max() * 1.2)

    if order_count is not None:
        ax.text(
            0.98,
            0.98,
            f"Last 3 Months Orders: {order_count:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#4B5563",
        )

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "toppings_mix"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_rolling_zscore_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a scatter plot of rolling z-scores for daily sales."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed rolling z-score is empty; no figure generated.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(df["date"], df["z_score"], color="#4F5A3F", s=30, alpha=0.8)
    ax.axhline(0, color="#9CA3AF", linewidth=1)

    ax.set_title(title, pad=6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Z-Score (14-day)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "anomaly_detection"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_robust_zscore_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a scatter plot of robust z-scores for daily sales."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed robust z-score is empty; no figure generated.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(df["date"], df["z_score"], color="#6E7656", s=30, alpha=0.8)
    ax.axhline(0, color="#9CA3AF", linewidth=1)

    ax.set_title(title, pad=6)
    ax.set_xlabel("Date")
    ax.set_ylabel("Robust Z-Score (Median/MAD)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()

    figures_dir = base_dir / "figures" / "anomaly_detection"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_top10_anomaly_sales_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
) -> Path:
    """Create a bar chart of top anomaly days by absolute z-score."""
    processed_path = base_dir / "data" / "processed" / processed_name
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed top anomalies is empty; no figure generated.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("abs_z_score", ascending=True)
    df["date_label"] = df["date"].dt.strftime("%Y-%m-%d")

    fig_height = max(4, min(10, 0.6 * len(df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(df["date_label"], df["total_sales"], color="#6E7656")
    ax.set_title(title, pad=6)
    ax.set_xlabel("Gross Sales")
    ax.set_ylabel("Date")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_currency(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar, z in zip(bars, df["z_score"]):
        ax.text(
            bar.get_width() + df["total_sales"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"z={z:.2f}",
            va="center",
            fontsize=8,
            color="#374151",
        )

    ax.set_xlim(0, df["total_sales"].max() * 1.25)
    fig.tight_layout()

    figures_dir = base_dir / "figures" / "anomaly_detection"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# --- Entry point ---
def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    order_count_path = base_dir / "data" / "processed" / "last_3_months_order_count.csv"
    order_count = None
    if order_count_path.exists():
        try:
            order_count = int(pd.read_csv(order_count_path)["value"].iloc[0])
        except (KeyError, ValueError, IndexError):
            order_count = None
    last_month_output = generate_product_mix_figure(
        base_dir,
        "last_month_product_mix.csv",
        "last_month_product_mix.png",
        "Last Month Product Mix",
    )
    last_3_months_output = generate_product_mix_figure(
        base_dir,
        "last_3_months_product_mix.csv",
        "last_3_months_product_mix.png",
        "Last 3 Months Product Mix",
    )
    top_products_output = generate_top_products_figure(
        base_dir,
        "last_month_product_mix.csv",
        "last_month_top_10_products.png",
        "Top 10 Products (Last Month)",
        top_n=10,
    )
    category_output = generate_category_mix_figure(
        base_dir,
        "last_month_category_mix.csv",
        "last_month_category_mix.png",
        "Category Mix (Last Month)",
    )
    top_products_3_months_output = generate_top_products_figure(
        base_dir,
        "last_3_months_product_mix.csv",
        "last_3_months_top_10_products.png",
        "Top 10 Products (Last 3 Months)",
        top_n=10,
    )
    category_3_months_output = generate_category_mix_figure(
        base_dir,
        "last_3_months_category_mix.csv",
        "last_3_months_category_mix.png",
        "Category Mix (Last 3 Months)",
    )
    pareto_last_month_output = generate_pareto_products_figure(
        base_dir,
        "last_month_product_mix.csv",
        "last_month_pareto.png",
        "Pareto: Product Mix (Last Month)",
    )
    donut_last_month_output = generate_category_share_donut(
        base_dir,
        "last_month_category_mix.csv",
        "last_month_category_share.png",
        "Category Share (Last Month)",
    )
    donut_last_3_months_output = generate_category_share_donut(
        base_dir,
        "last_3_months_category_mix.csv",
        "last_3_months_category_share.png",
        "Category Share (Last 3 Months)",
    )
    channel_last_month_output = generate_channel_mix_figure(
        base_dir,
        "last_month_channel_mix.csv",
        "last_month_channel_mix.png",
        "Channel Mix (Last Month)",
        "channel_group",
        "channel_sales_pct_of_total",
    )
    channel_last_3_months_output = generate_channel_mix_figure(
        base_dir,
        "last_3_months_channel_mix.csv",
        "last_3_months_channel_mix.png",
        "Channel Mix (Last 3 Months)",
        "channel_group",
        "channel_sales_pct_of_total",
    )
    in_person_last_month_output = generate_channel_mix_figure(
        base_dir,
        "last_month_in_person_mix.csv",
        "last_month_in_person_mix.png",
        "In-Person Mix (Last Month)",
        "in_person_channel",
        "in_person_sales_pct_of_total",
        y_label="In-Person Channel",
        color_map={
            "Kiosk": "#6E7656",
            "Counter": "#4F5A3F",
        },
    )
    in_person_last_3_months_output = generate_channel_mix_figure(
        base_dir,
        "last_3_months_in_person_mix.csv",
        "last_3_months_in_person_mix.png",
        "In-Person Mix (Last 3 Months)",
        "in_person_channel",
        "in_person_sales_pct_of_total",
        y_label="In-Person Channel",
        color_map={
            "Kiosk": "#6E7656",
            "Counter": "#4F5A3F",
        },
    )
    milk_type_last_month_output = generate_channel_mix_figure(
        base_dir,
        "last_month_milk_type_mix.csv",
        "last_month_milk_type_mix.png",
        "Milk Tea vs Au Lait (Last Month)",
        "milk_type",
        "milk_type_sales_pct_of_total",
        y_label="Drink Type",
        color_map={
            "Milk Tea": "#5A8F3A",
            "Au Lait": "#3E6B2C",
        },
    )
    milk_type_last_3_months_output = generate_channel_mix_figure(
        base_dir,
        "last_3_months_milk_type_mix.csv",
        "last_3_months_milk_type_mix.png",
        "Milk Tea vs Au Lait (Last 3 Months)",
        "milk_type",
        "milk_type_sales_pct_of_total",
        y_label="Drink Type",
        color_map={
            "Milk Tea": "#5A8F3A",
            "Au Lait": "#3E6B2C",
        },
    )
    milk_type_global_output = generate_channel_mix_figure(
        base_dir,
        "global_milk_type_mix.csv",
        "global_milk_type_mix.png",
        "Milk Tea vs Au Lait (All Data)",
        "milk_type",
        "milk_type_sales_pct_of_total",
        y_label="Drink Type",
        color_map={
            "Milk Tea": "#5A8F3A",
            "Au Lait": "#3E6B2C",
        },
    )
    fresh_fruit_tea_base_last_month_output = generate_channel_mix_figure(
        base_dir,
        "last_month_fresh_fruit_tea_base_mix.csv",
        "last_month_fresh_fruit_tea_base_mix.png",
        "Fresh Fruit Tea Base Mix (Last Month)",
        "tea_base",
        "tea_base_sales_pct_of_total",
        y_label="Tea Base",
        color_map={
            "Green": "#5A8F3A",
            "Four Seasons": "#3E6B2C",
        },
    )
    fresh_fruit_tea_base_last_3_months_output = generate_channel_mix_figure(
        base_dir,
        "last_3_months_fresh_fruit_tea_base_mix.csv",
        "last_3_months_fresh_fruit_tea_base_mix.png",
        "Fresh Fruit Tea Base Mix (Last 3 Months)",
        "tea_base",
        "tea_base_sales_pct_of_total",
        y_label="Tea Base",
        color_map={
            "Green": "#5A8F3A",
            "Four Seasons": "#3E6B2C",
        },
    )
    fresh_fruit_tea_base_global_output = generate_channel_mix_figure(
        base_dir,
        "global_fresh_fruit_tea_base_mix.csv",
        "global_fresh_fruit_tea_base_mix.png",
        "Fresh Fruit Tea Base Mix (All Data)",
        "tea_base",
        "tea_base_sales_pct_of_total",
        y_label="Tea Base",
        color_map={
            "Green": "#5A8F3A",
            "Four Seasons": "#3E6B2C",
        },
    )
    top_item_by_base_last_month_output = generate_top_item_by_tea_base_figure(
        base_dir,
        "last_month_top_item_by_tea_base.csv",
        "last_month_top_item_by_tea_base.png",
        "Top Item by Tea Base (Last Month)",
    )
    top_item_by_base_last_3_months_output = generate_top_item_by_tea_base_figure(
        base_dir,
        "last_3_months_top_item_by_tea_base.csv",
        "last_3_months_top_item_by_tea_base.png",
        "Top Item by Tea Base (Last 3 Months)",
    )
    top_item_by_base_global_output = generate_top_item_by_tea_base_figure(
        base_dir,
        "global_top_item_by_tea_base.csv",
        "global_top_item_by_tea_base.png",
        "Top Item by Tea Base (All Data)",
    )
    daily_sales_anomaly_output = generate_daily_sales_anomalies_figure(
        base_dir,
        "global_daily_sales_zscore.csv",
        "global_daily_sales_anomalies.png",
        "Daily Sales Anomalies (Z-Score vs Weekday Baseline)",
        threshold=2.25,
    )
    rolling_zscore_output = generate_rolling_zscore_figure(
        base_dir,
        "global_daily_sales_rolling_zscore.csv",
        "global_daily_sales_rolling_zscore.png",
        "Daily Sales Rolling Z-Score (14-day window)",
    )
    robust_zscore_output = generate_robust_zscore_figure(
        base_dir,
        "global_daily_sales_robust_zscore.csv",
        "global_daily_sales_robust_zscore.png",
        "Daily Sales Robust Z-Score (Median/MAD)",
    )
    top10_anomaly_sales_output = generate_top10_anomaly_sales_figure(
        base_dir,
        "global_daily_sales_top10_anomalies.csv",
        "global_daily_sales_top10_anomalies.png",
        "Top 10 Anomaly Days by Z-Score (Gross Sales)",
    )
    sugar_pct_output = generate_pct_mix_figure(
        base_dir,
        "global_sugar_pct_mix.csv",
        "global_sugar_pct_mix.png",
        "Sugar Level Mix (All Data)",
        "sugar_pct",
        "#5A8F3A",
        order_count=order_count,
    )
    ice_pct_output = generate_pct_mix_figure(
        base_dir,
        "global_ice_pct_mix.csv",
        "global_ice_pct_mix.png",
        "Ice Level Mix (All Data)",
        "ice_pct",
        "#3E6B2C",
        order_count=order_count,
    )
    tea_base_last_month_output = generate_tea_base_mix_figure(
        base_dir,
        "last_month_tea_base_mix.csv",
        "last_month_tea_base_mix.png",
        "Tea Base Mix (Last Month)",
    )
    tea_base_last_3_months_output = generate_tea_base_mix_figure(
        base_dir,
        "last_3_months_tea_base_mix.csv",
        "last_3_months_tea_base_mix.png",
        "Tea Base Mix (Last 3 Months)",
    )
    tea_base_global_output = generate_tea_base_mix_figure(
        base_dir,
        "global_tea_base_mix.csv",
        "global_tea_base_mix.png",
        "Tea Base Mix (All Data)",
    )
    peak_hours_last_month_output = generate_peak_hours_figure(
        base_dir,
        "last_month_hourly_sales.csv",
        "last_month_peak_hours.png",
        "Peak Hours (Last Month)",
    )
    peak_hours_weekday_last_month_output = generate_peak_hours_figure(
        base_dir,
        "last_month_weekday_hourly_sales.csv",
        "last_month_peak_hours_weekday.png",
        "Peak Hours (Weekdays, Last Month)",
        bar_color="#6E7656",
    )
    peak_hours_weekend_last_month_output = generate_peak_hours_figure(
        base_dir,
        "last_month_weekend_hourly_sales.csv",
        "last_month_peak_hours_weekend.png",
        "Peak Hours (Weekends, Last Month)",
        bar_color="#4F5A3F",
    )
    featured_item_last_month_output = generate_peak_hours_figure(
        base_dir,
        "last_month_featured_item_hourly_sales.csv",
        "last_month_tgy_special_by_hour.png",
        "TGY Special by Hour (Last Month)",
        y_label="Percent of Item Sales",
    )
    top_products_global_output = generate_top_products_figure(
        base_dir,
        "global_product_mix.csv",
        "global_top_10_products.png",
        "Top 10 Products (All Data)",
        top_n=10,
    )
    category_global_output = generate_category_mix_figure(
        base_dir,
        "global_category_mix.csv",
        "global_category_mix.png",
        "Category Mix (All Data)",
    )
    print(f"Saved figure: {last_month_output}")
    print(f"Saved figure: {last_3_months_output}")
    print(f"Saved figure: {top_products_output}")
    print(f"Saved figure: {category_output}")
    print(f"Saved figure: {top_products_3_months_output}")
    print(f"Saved figure: {category_3_months_output}")
    print(f"Saved figure: {pareto_last_month_output}")
    print(f"Saved figure: {donut_last_month_output}")
    print(f"Saved figure: {donut_last_3_months_output}")
    print(f"Saved figure: {channel_last_month_output}")
    print(f"Saved figure: {channel_last_3_months_output}")
    print(f"Saved figure: {in_person_last_month_output}")
    print(f"Saved figure: {in_person_last_3_months_output}")
    print(f"Saved figure: {milk_type_last_month_output}")
    print(f"Saved figure: {milk_type_last_3_months_output}")
    print(f"Saved figure: {milk_type_global_output}")
    print(f"Saved figure: {fresh_fruit_tea_base_last_month_output}")
    print(f"Saved figure: {fresh_fruit_tea_base_last_3_months_output}")
    print(f"Saved figure: {fresh_fruit_tea_base_global_output}")
    print(f"Saved figure: {top_item_by_base_last_month_output}")
    print(f"Saved figure: {top_item_by_base_last_3_months_output}")
    print(f"Saved figure: {top_item_by_base_global_output}")
    print(f"Saved figure: {daily_sales_anomaly_output}")
    print(f"Saved figure: {rolling_zscore_output}")
    print(f"Saved figure: {robust_zscore_output}")
    print(f"Saved figure: {top10_anomaly_sales_output}")
    print(f"Saved figure: {sugar_pct_output}")
    print(f"Saved figure: {ice_pct_output}")
    print(f"Saved figure: {tea_base_last_month_output}")
    print(f"Saved figure: {tea_base_last_3_months_output}")
    print(f"Saved figure: {tea_base_global_output}")
    print(f"Saved figure: {peak_hours_last_month_output}")
    print(f"Saved figure: {peak_hours_weekday_last_month_output}")
    print(f"Saved figure: {peak_hours_weekend_last_month_output}")
    print(f"Saved figure: {featured_item_last_month_output}")
    print(f"Saved figure: {top_products_global_output}")
    print(f"Saved figure: {category_global_output}")


if __name__ == "__main__":
    main()
