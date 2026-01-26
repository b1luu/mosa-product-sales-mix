# Mosa Product Sales Mix

## Project Description
This repository analyzes the percentage of total sales by product category and by individual product using Square export data. The focus is on a clean, reproducible pipeline and clear analysis artifacts.

## Objectives
- Summarize sales mix by product category.
- Summarize sales mix by individual product.
- Summarize sales mix by tea base and milk type.
- Visualize peak hours and featured-item trends.
- Produce tidy, reusable outputs for reporting and visualization.

## Data Privacy Note
All raw exports and sensitive mappings must stay local and out of version control. Only aggregated or anonymized outputs belong in `data/processed/` when appropriate.

## Requirements
- Python 3.11+
- `pandas`

## Analysis Scope
- Inputs: Square detailed line-item export CSVs in `data/raw/`.
- Outputs: aggregated datasets in `data/processed/`, figures in `figures/`, and written results in `reports/`.
- Methods: Python 3.11+ scripts in `src/`.

## Input Expectations (Detailed Line-Item Export)
The pipeline expects a Square detailed line-item export with columns that map to:
- Order ID (e.g., `Transaction ID`)
- Order datetime (either `order_datetime` or `Date` + `Time`)
- Category (e.g., `Category`)
- Item (e.g., `Item`)
- Quantity (e.g., `Qty`)
- Gross sales amount (e.g., `Gross Sales`)

If those fields are missing after normalization, the script will exit with a clear error.

## How To Run
1) Export a detailed line-item CSV from Square and place it in `data/raw/`.
2) Ensure there is only one CSV in `data/raw/` or name it `orders.csv`.
3) Run:
```bash
python3 src/compute_sales_mix.py
```

## Quick Start
```bash
python3 src/compute_sales_mix.py
python3 src/generate_figures.py
```
Expected outputs:
- CSVs in `data/processed/`
- PNGs in `figures/`

## Pipeline Order
1) Generate processed CSVs:
```bash
python3 src/compute_sales_mix.py
```
2) Generate figures:
```bash
python3 src/generate_figures.py
```

## Config & Local Files
- Local-only inputs (not in git):
  - `data/raw/` (Square exports)
  - `data/private/` (channel mix exports, event lists)
- Example local files:
  - `data/private/channelmix-raw.csv` (optional)
  - `data/private/event_days.csv` with columns `date` and `event_name`

## Monthly Update Checklist
1) Replace the raw export in `data/raw/`.
2) Run `python3 src/compute_sales_mix.py`.
3) Run `python3 src/generate_figures.py`.
4) Review updated `reports/` notes and key charts.

## Key Outputs Map
- Product mix: `data/processed/last_3_months_product_mix.csv` → `figures/items/last_3_months_product_mix.png`
- Category mix: `data/processed/last_3_months_category_mix.csv` → `figures/drink_share/last_3_months_category_mix.png`
- Tea base mix: `data/processed/last_3_months_tea_base_mix.csv` → `figures/tea_base/last_3_months_tea_base_mix.png`
- Tea base by drink category: `data/processed/last_3_months_tea_base_by_drink_category_all.csv` → `figures/tea_base/last_3_months_tgy_oolong_by_category_pie.png`
- Event day analysis: `data/processed/event_day_summary.csv` → table review in Excel/Tableau

## Known Assumptions
- Refund handling: refunds removed unless notes match Hungry Panda patterns.
- Tea base mapping is rule-based and documented in `reports/tea_base_mapping.md`.
- Modifiers containing "jelly" are ignored for tea base assignment to avoid topping leakage.
- "Green-inclusive" tea base breakdowns roll `Genmai Green` into `Green` for those charts only.

## Troubleshooting
- Missing required columns: ensure the export is the detailed line-item report.
- Empty outputs: check that the date range includes data and sales fields are populated.
- Event analysis missing: add `data/private/event_days.csv` and re-run the pipeline.

## Current Outputs
- `data/processed/last_month_category_mix.csv`
- `data/processed/last_month_product_mix.csv`
- `data/processed/last_3_months_category_mix.csv`
- `data/processed/last_3_months_product_mix.csv`
- `data/processed/global_category_mix.csv`
- `data/processed/global_product_mix.csv`
- `data/processed/last_month_channel_mix.csv`
- `data/processed/last_3_months_channel_mix.csv`
- `data/processed/global_channel_mix.csv`
- `data/processed/last_month_in_person_mix.csv`
- `data/processed/last_3_months_in_person_mix.csv`
- `data/processed/global_in_person_mix.csv`
- `data/processed/last_month_tea_base_mix.csv`
- `data/processed/last_3_months_tea_base_mix.csv`
- `data/processed/global_tea_base_mix.csv`
- `data/processed/last_month_milk_type_mix.csv`
- `data/processed/last_3_months_milk_type_mix.csv`
- `data/processed/global_milk_type_mix.csv`
- `data/processed/last_month_hourly_sales.csv`
- `data/processed/last_3_months_hourly_sales.csv`
- `data/processed/global_hourly_sales.csv`
- `data/processed/last_month_weekday_hourly_sales.csv`
- `data/processed/last_month_weekend_hourly_sales.csv`
- `data/processed/last_3_months_weekday_hourly_sales.csv`
- `data/processed/last_3_months_weekend_hourly_sales.csv`
- `data/processed/global_weekday_hourly_sales.csv`
- `data/processed/global_weekend_hourly_sales.csv`
- `data/processed/last_month_featured_item_hourly_sales.csv`
- `data/processed/last_month_fresh_fruit_tea_base_mix.csv`
- `data/processed/last_3_months_fresh_fruit_tea_base_mix.csv`
- `data/processed/global_fresh_fruit_tea_base_mix.csv`
- `data/processed/last_month_top_item_by_tea_base.csv`
- `data/processed/last_3_months_top_item_by_tea_base.csv`
- `data/processed/global_top_item_by_tea_base.csv`
- `data/processed/last_month_top_25_products_with_other.csv`
- `data/processed/last_3_months_top_25_products_with_other.csv`
- `data/processed/last_3_months_order_count.csv`
- `data/processed/last_3_months_topping_popularity.csv`
- `data/processed/last_3_months_item_pair_stats.csv`
- `data/processed/last_3_months_item_pair_top10.csv`

## Output Schema
- Category mix files: `category_name`, `total_sales`, `category_sales_pct_of_total`
- Category net mix files: `category_name`, `total_net_sales`, `category_sales_pct_of_total`
- Product mix files: `category_name`, `item_name`, `total_sales`, `product_sales_pct_of_category`, `product_sales_pct_of_total`
- Tea base mix files: `tea_base`, `total_sales`, `tea_base_sales_pct_of_total`
- Milk type mix files: `milk_type`, `total_sales`, `milk_type_sales_pct_of_total`
- Fresh fruit tea base mix files: `tea_base`, `total_sales`, `tea_base_sales_pct_of_total`
- Top item by tea base files: `tea_base`, `item_name`, `total_sales`, `item_sales_pct_of_base`
- Hourly sales files: `hour`, `total_sales`, `sales_pct_of_total`
- Item pair stats files: `item_a`, `item_b`, `count`, `support`, `confidence`, `lift`, `pair_sales`, `pair_sales_pct_of_total`, `total_transactions`
- Topping popularity files: `topping`, `count`, `share_of_toppings`
- Rolling z-score file: `date`, `total_sales`, `rolling_mean`, `rolling_std`, `z_score`
- Robust z-score file: `date`, `total_sales`, `median`, `mad`, `z_score`
- Top anomaly days file: `date`, `total_sales`, `weekday`, `baseline_mean`, `baseline_std`, `z_score`, `abs_z_score`
- Percentages are decimals in the range 0-1 (multiply by 100 for percent).

## Notes
- Baselines default to last full month and last 3 full months.
- The “last 3 months” window is Oct 1 to Dec 31 and is used for higher statistical stability.
- A global (all data) sales mix is also generated.
- Refund handling: rows with `Event Type` = `Refund` are excluded unless `Notes` indicates a valid Hungry Panda sale (`Hp`, `HP`, `Hp ####`, `Hp Order`, `Panda`, `Pandaa`). Rows with `Notes` containing `Canceled Order` are always removed. Valid Hungry Panda refunds are treated as positive sales.
- Channel mix: `channel_group` classifies rows into Hungry Panda, DoorDash, Uber Eats, Square Online, In Person, or Other. In-person orders are split into `Kiosk` vs `Counter`.
- Channel mix uses `data/private/channelmix-raw.csv` when available to map `Source` values (Register, Kiosk, Square Online) more accurately than the standard export.
- Tea base mapping: derived from item names, modifiers, and categories. Modifiers containing "jelly" are ignored so toppings like Osmanthus Tie Guan Yin Jelly do not force a TGY base. See `reports/tea_base_mapping.md` for rule order and signature overrides.
- Tea base by drink category: primary outputs use the standard tea base mapping; additional "green-inclusive" outputs roll `Genmai Green` into `Green` for category breakdown charts only (files ending in `_green_inclusive.csv`).
- Event day analysis: add a local `data/private/event_days.csv` with columns `date` (YYYY-MM-DD) and `event_name`. When present, the pipeline writes `data/processed/event_day_analysis.csv` (per-date metrics) and `data/processed/event_day_summary.csv` (event averages) using weekday-baseline z-scores from `global_daily_sales_zscore.csv`.
- Milk type mapping: uses `category_name` to classify `Milk Tea` vs `Au Lait`.
- Fresh fruit tea base mix: filtered to items where `item_name` contains `Fresh Fruit Tea` and base is either `Green` or `Four Seasons`.
- Item co-purchase analysis: groups items by `Transaction ID`, keeps unique items per order, filters to baskets with 2-6 items, and outputs support, confidence, lift, and pair-level sales for the last 3 months. Pairs below 0.5% support or lift < 1.5 are filtered out to reduce noise.
- Co-purchase disclaimer: current pair supports are under 1% of transactions, so results are noted as a practical attempt but are not considered actionable and are not driving decisions.
- Topping popularity: extracted from `Modifiers Applied` by removing tea base choices and sugar/ice level entries; counts reflect how often toppings appear across orders.
- Anomaly detection notes: z-score is the number of standard deviations from the weekday baseline mean; a normal distribution rule of thumb is ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ (approximate). For the current 92-day sample, anomaly counts were 2.5 -> 2 days, 2.25 -> 2 days, 2.0 -> 3 days, 1.75 -> 6 days.
- Anomaly threshold guidance: 2.25 is intentionally strict and highlights only major deviations. Use the top-10 by |z| list for a broader review set when you want more investigate-worthy days.
- Rolling z-score notes: rolling z-score compares each day to the prior 14-day window (mean and std) to highlight short-term deviations after smoothing recent trends.
- Robust z-score notes: uses median and MAD (median absolute deviation) instead of mean/std, which is less sensitive to extreme outliers.
- See `reports/summary.md` for an executive summary and limitations.

## Future Improvements
- Co-purchase analysis: limit to top-N items, add lift thresholds, and segment by channel to improve signal quality.
- Add item category-level co-purchase to reduce sparse pairs.
- Track trends in pair strength over time (rolling 4-week).

## Channel Rules
- Hungry Panda: `Notes` contains `Hp`, `HP`, `Hp ####`, `Hp Order`, `Panda`, or `Pandaa`.
- DoorDash / Uber Eats / Square Online: inferred from `Channel` values (case-insensitive match).
- In Person: default when no delivery or HP rule matches.
- In-person split: `Kiosk` when `Channel` contains `Kiosk`, otherwise `Counter`.
- Other: `Channel` exists but does not match any known delivery or in-person label.

## Troubleshooting
- Missing required columns: confirm the export is the detailed line-item report and includes Date, Time, Category, Item, Qty, Gross Sales, and Transaction ID (or equivalent).
- Multiple CSVs in `data/raw/`: keep a single export file or rename the target file to `orders.csv`.
- Empty outputs: check that the date range in the export includes recent data and that `Gross Sales` values are not blank or zero.

## Testing
- Run all tests: `python3 -m unittest`
- Run smoke tests only: `python3 -m unittest tests.test_smoke`

## Figures
- Run `python3 src/generate_figures.py` to create PNGs in goal-based folders under `figures/`:
  - `figures/anomaly_detection/`: daily anomalies, rolling z-score, robust z-score
  - `figures/tea_base/`: tea base mix, fresh fruit base mix, top item by base
  - `figures/drink_share/`: category mix/share, channel mix, in-person mix, milk type mix, peak hours
  - `figures/toppings_mix/`: sugar % mix, ice % mix, top toppings
  - `figures/items/`: product mix, top 10 products, top 10 products by sales, Pareto chart
