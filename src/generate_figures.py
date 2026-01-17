"""Generate PNG figures from processed sales mix outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _format_pct(value: float) -> str:
    """Format a decimal percent for axis labels."""
    return f"{value * 100:.1f}%"


def generate_last_month_product_mix_figure(base_dir: Path) -> Path:
    """Create a horizontal bar chart for last month's product mix."""
    processed_path = base_dir / "data" / "processed" / "last_month_product_mix.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"Missing processed file: {processed_path}")

    plt.rcParams["font.family"] = "PingFang SC"

    df = pd.read_csv(processed_path)
    if df.empty:
        raise ValueError("Processed product mix is empty; no figure generated.")

    df = df.sort_values("product_sales_pct_of_total", ascending=True)
    df["label"] = df["item_name"].fillna("Unknown Item")

    fig_height = max(4, min(20, 0.3 * len(df)))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(df["label"], df["product_sales_pct_of_total"], color="#2A6F8F")
    ax.set_title("Last Month Product Mix")
    ax.set_xlabel("Percent of Total Sales")
    ax.set_ylabel("Product")

    ticks = ax.get_xticks()
    ax.set_xticklabels([_format_pct(tick) for tick in ticks])
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()

    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "last_month_product_mix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    output_path = generate_last_month_product_mix_figure(base_dir)
    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()
