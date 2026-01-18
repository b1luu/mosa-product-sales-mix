"""Generate PNG figures from processed sales mix outputs."""
from __future__ import annotations

from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd


def _format_pct(value: float) -> str:
    """Format a decimal percent for axis labels."""
    return f"{value * 100:.1f}%"


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

    exclude_patterns = (
        r"\btips?\b",
        r"boba tea tote bag",
        r"free drink",
        r"custom amount",
    )
    df = df[
        ~df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(exclude_patterns), regex=True, na=False)
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
        loc="lower right",
        frameon=False,
    )

    fig.tight_layout()

    figures_dir = base_dir / "figures"
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

    exclude_patterns = (
        r"\btips?\b",
        r"boba tea tote bag",
        r"free drink",
        r"custom amount",
    )
    df = df[
        ~df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(exclude_patterns), regex=True, na=False)
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

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


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

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


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

    exclude_patterns = (
        r"\btips?\b",
        r"boba tea tote bag",
        r"free drink",
        r"custom amount",
    )
    df = df[
        ~df["item_name"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("|".join(exclude_patterns), regex=True, na=False)
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

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


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

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def generate_channel_mix_figure(
    base_dir: Path,
    processed_name: str,
    output_name: str,
    title: str,
    label_column: str,
    pct_column: str,
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
    bars = ax.barh(df["label"], df[pct_column], color="#4C7EA8")
    ax.set_title(title, pad=4)
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel("Channel")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.bar_label(bars, labels=[_format_pct(v) for v in df[pct_column]], padding=3)
    max_pct = df[pct_column].max()
    ax.set_xlim(0, max_pct * 1.15)

    fig.tight_layout()

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
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
    )
    in_person_last_3_months_output = generate_channel_mix_figure(
        base_dir,
        "last_3_months_in_person_mix.csv",
        "last_3_months_in_person_mix.png",
        "In-Person Mix (Last 3 Months)",
        "in_person_channel",
        "in_person_sales_pct_of_total",
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


if __name__ == "__main__":
    main()
