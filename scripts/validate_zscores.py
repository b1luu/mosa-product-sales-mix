"""Quick sanity checks for z-score outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    zscore_path = base_dir / "data" / "processed" / "global_daily_sales_zscore.csv"
    rolling_path = base_dir / "data" / "processed" / "global_daily_sales_rolling_zscore.csv"
    robust_path = base_dir / "data" / "processed" / "global_daily_sales_robust_zscore.csv"

    checks = []
    for label, path in [
        ("weekday z-score", zscore_path),
        ("rolling z-score", rolling_path),
        ("robust z-score", robust_path),
    ]:
        if not path.exists():
            print(f"{label}: missing {path}")
            checks.append(False)
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"{label}: empty")
            checks.append(False)
            continue
        mean_z = df["z_score"].mean()
        max_z = df["z_score"].abs().max()
        print(f"{label}: mean z={mean_z:.3f}, max |z|={max_z:.3f}")
        checks.append(True)

    if zscore_path.exists():
        df = pd.read_csv(zscore_path)
        if not df.empty:
            sample = df.iloc[0]
            print(
                "sample check:",
                f"date={sample['date']},",
                f"z={sample['z_score']:.3f},",
                f"baseline_mean={sample['baseline_mean']:.2f},",
                f"baseline_std={sample['baseline_std']:.2f}",
            )

    if checks and all(checks):
        print("PASS: z-score outputs present and non-empty.")
    else:
        print("WARN: one or more z-score outputs missing or empty.")


if __name__ == "__main__":
    main()
