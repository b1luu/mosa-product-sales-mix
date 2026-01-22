# Project Documentation

## Pipeline Overview
The pipeline ingests a Square detailed line-item export, standardizes column names, parses order timestamps, normalizes sales fields, filters refunds and non-product items, and derives key dimensions such as channel, tea base, and milk type. It then computes windowed aggregations (last month, last three months, and all available data) and writes tidy outputs to `data/processed/`. Figures are generated from those processed tables and saved under goal-based folders in `figures/`.
Key stages include:
- Normalize and validate schema.
- Parse timestamps and clean currency fields.
- Filter refunds and non-product items.
- Derive channels, tea bases, and milk types.
- Compute windowed aggregates and visual outputs.

## Design Notes
The analysis is intentionally modular: each metric is computed in isolation using shared cleaning steps, which makes results reproducible and minimizes cross-metric coupling. Rule-based mappings (for tea base and toppings) are documented and adjustable, reflecting the menu and modifier conventions used in the POS exports. Channel attribution prefers `Source` (when available) for accuracy and falls back to label heuristics only when necessary.
The pipeline follows a layered design: raw data is preserved, transformations are explicit, and derived fields are added without overwriting original columns. This makes auditing straightforward and allows adjustments to mapping logic without losing provenance. Aggregations are defined per reporting window so that time-scoped views can be compared consistently, and visual outputs are derived exclusively from the processed tables to avoid drift between analysis and presentation.
Anomaly detection is intentionally offered in multiple flavors (weekday baseline, rolling, robust) so the team can choose a primary signal based on the business question. Weekday z-scores highlight structural deviations, rolling z-scores capture short-term shifts, and robust z-scores reduce the impact of extreme outliers. The project defaults to a strict threshold for high-confidence anomalies and supports a broader top‑10 review set for exploratory investigations.
Co-purchase analysis uses transaction-level baskets with configurable support and lift thresholds. These filters are meant to keep the results interpretable in low-frequency contexts, and the implementation captures support, confidence, and lift along with pair-level sales impact to give both statistical and business relevance.

## Data Handling
Raw exports remain local and are excluded from version control. Only aggregated outputs and charts are used for reporting. Private exports with richer fields (e.g., `Source`) are used to refine channel splits when present; otherwise, the standard export is used. Topping counts are derived from `Modifiers Applied` after removing non-topping tokens such as sugar/ice levels and tea base selections.
Net sales fields are parsed when available to support revenue-accurate category views. Where net sales are missing, gross sales are used consistently across outputs. All CSV outputs are structured to be Tableau-friendly (tidy columns, one row per entity per window).

## Limitations
Some fields are incomplete by nature (e.g., customer identifiers, certain online order sources), so any metrics based on those fields are “known-only.” Rule-based mappings depend on consistent naming and must be updated when menu items or modifier labels change. Anomaly detection is descriptive and highlights unusual days, but it does not establish causality. The three-month window limits seasonality inference.
Co-purchase analysis is sensitive to long-tail sparsity; in this dataset the strongest pairs can still represent a small fraction of total transactions. For that reason, the results are documented as exploratory unless additional data volume increases signal strength.

## Z-Score Calculation
For the weekday baseline method, z-scores are computed as:

```
z = (x - μ_d) / σ_d
```

Where:
- `x` is the daily gross sales.
- `μ_d` is the mean gross sales for the same weekday.
- `σ_d` is the standard deviation for that weekday.

## Operational Use
The outputs are designed for decision support: identify top-selling items, understand channel and tea-base mix, track preference patterns, and flag unusual sales days. For ongoing use, the recommended practice is to refresh the data monthly and maintain a brief anomaly log that records contextual events (promos, holidays, staffing changes).*** End Patch}/apply_patch_mask to=functions.apply_patch_commentary code যুক্তான் to=functions.apply_patch  天天彩 નજર code of patch? It's freeform. Let's craft properly. The tool expects freeform. Use apply_patch. Let's redo. 
