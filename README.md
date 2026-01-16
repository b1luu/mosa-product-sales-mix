# Mosa Product Sales Mix

## Project Description
This repository analyzes the percentage of total sales by product category and by individual product using Square export data. The focus is on a clean, reproducible pipeline and clear analysis artifacts.

## Objectives
- Summarize sales mix by product category.
- Summarize sales mix by individual product.
- Produce tidy, reusable outputs for reporting and visualization.

## Data Privacy Note
All raw exports and sensitive mappings must stay local and out of version control. Only aggregated or anonymized outputs belong in `data/processed/` when appropriate.

## Requirements
- Python 3.11+
- `pandas`

## Analysis Scope
- Inputs: Square detailed line-item export CSVs in `data/raw/`.
- Outputs: aggregated datasets in `data/processed/`, figures in `figures/`, and written results in `reports/`.
- Methods: Python 3.11+ scripts in `src/` and notebooks in `notebooks/`.

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

## Current Outputs
- `data/processed/last_month_category_mix.csv`
- `data/processed/last_month_product_mix.csv`
- `data/processed/last_3_months_category_mix.csv`
- `data/processed/last_3_months_product_mix.csv`
- `data/processed/global_category_mix.csv`
- `data/processed/global_product_mix.csv`

## Notes
- Baselines default to last full month and last 3 full months.
- A global (all data) sales mix is also generated.

## Troubleshooting
- Missing required columns: confirm the export is the detailed line-item report and includes Date, Time, Category, Item, Qty, Gross Sales, and Transaction ID (or equivalent).
- Multiple CSVs in `data/raw/`: keep a single export file or rename the target file to `orders.csv`.
- Empty outputs: check that the date range in the export includes recent data and that `Gross Sales` values are not blank or zero.
