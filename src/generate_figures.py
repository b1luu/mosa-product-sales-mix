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
    print(f"Saved figure: {last_month_output}")
    print(f"Saved figure: {last_3_months_output}")


if __name__ == "__main__":
    main()
