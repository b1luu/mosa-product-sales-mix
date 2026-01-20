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

## Pipeline Order
1) Generate processed CSVs:
```bash
python3 src/compute_sales_mix.py
```
2) Generate figures:
```bash
python3 src/generate_figures.py
```

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
- `data/processed/last_3_months_order_count.csv`
- `data/processed/last_3_months_item_pair_stats.csv`

## Output Schema
- Category mix files: `category_name`, `total_sales`, `category_sales_pct_of_total`
- Product mix files: `category_name`, `item_name`, `total_sales`, `product_sales_pct_of_category`, `product_sales_pct_of_total`
- Tea base mix files: `tea_base`, `total_sales`, `tea_base_sales_pct_of_total`
- Milk type mix files: `milk_type`, `total_sales`, `milk_type_sales_pct_of_total`
- Fresh fruit tea base mix files: `tea_base`, `total_sales`, `tea_base_sales_pct_of_total`
- Top item by tea base files: `tea_base`, `item_name`, `total_sales`, `item_sales_pct_of_base`
- Hourly sales files: `hour`, `total_sales`, `sales_pct_of_total`
- Item pair stats files: `item_a`, `item_b`, `count`, `support`, `confidence`, `lift`
- Percentages are decimals in the range 0-1 (multiply by 100 for percent).

## Notes
- Baselines default to last full month and last 3 full months.
- A global (all data) sales mix is also generated.
- Refund handling: rows with `Event Type` = `Refund` are excluded unless `Notes` indicates a valid Hungry Panda sale (`Hp`, `HP`, `Hp ####`, `Hp Order`, `Panda`, `Pandaa`). Rows with `Notes` containing `Canceled Order` are always removed. Valid Hungry Panda refunds are treated as positive sales.
- Channel mix: `channel_group` classifies rows into Hungry Panda, DoorDash, Uber Eats, Square Online, In Person, or Other. In-person orders are split into `Kiosk` vs `Counter`.
- Tea base mapping: derived from item names, modifiers, and categories. See `reports/tea_base_mapping.md` for rule order and signature overrides.
- Milk type mapping: uses `category_name` to classify `Milk Tea` vs `Au Lait`.
- Fresh fruit tea base mix: filtered to items where `item_name` contains `Fresh Fruit Tea` and base is either `Green` or `Four Seasons`.
- Item co-purchase analysis: groups items by `Transaction ID`, keeps unique items per order, counts item pairs, and outputs support, confidence, and lift for the last 3 months. Pairs below 1% support are filtered out to reduce noise.

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
- Run `python3 src/generate_figures.py` to create PNGs in `figures/` for product mix, top 10 products, category mix/share, channel mix, in-person mix, tea base mix, milk type mix, fresh fruit tea base mix, top item by tea base, peak hours, daily sales anomalies, and a Pareto chart.
