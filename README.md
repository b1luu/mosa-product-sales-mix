# Mosa Product Sales Mix

## Project Description
This repository analyzes the percentage of total sales by product category and by individual product using Square export data. The focus is on a clean, reproducible pipeline and clear analysis artifacts.

## Objectives
- Summarize sales mix by product category.
- Summarize sales mix by individual product.
- Produce tidy, reusable outputs for reporting and visualization.

## Data Privacy Note
All raw exports and sensitive mappings must stay local and out of version control. Only aggregated or anonymized outputs belong in `data/processed/` when appropriate.

## Analysis Scope
- Inputs: Square detailed line-item export CSVs in `data/raw/`.
- Outputs: aggregated datasets in `data/processed/`, figures in `figures/`, and written results in `reports/`.
- Methods: Python 3.11+ scripts in `src/` and notebooks in `notebooks/`.

## Current Outputs
- `data/processed/last_month_category_mix.csv`
- `data/processed/last_month_product_mix.csv`
- `data/processed/last_3_months_category_mix.csv`
- `data/processed/last_3_months_product_mix.csv`
- `data/processed/global_category_mix.csv`
- `data/processed/global_product_mix.csv`
