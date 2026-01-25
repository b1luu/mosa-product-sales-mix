# Summary

## Executive Summary
- This project extends beyond Square POS reporting by standardizing raw exports, computing product/category sales mix across multiple time windows, and adding operational views (channels, tea base, sugar/ice mix, peak hours, and anomaly detection).
- Results are designed for decision support and prioritization, not causal claims.

## Scope
- Data source: Square detailed line-item exports.
- Time window: last full month, last 3 months, and all data (as available).
- Outputs: processed tables in `data/processed/` and charts in `figures/` organized by goal.

## Key Deliverables
- Product sales mix by category and product.
- Channel and in-person mix.
- Tea base mix (with documented mapping rules).
- Sugar/ice preference mix.
- Peak hours and anomaly detection.

## Notes on Milk Tea vs Au Lait Mix
- The mix is computed from line-item sales by summing `item_gross_sales` for items labeled as Milk Tea or Au Lait.
- Labels are assigned primarily from `category_name`; when categories are missing those terms, we explicitly include known Mosa Signature items: TGY Special as Milk Tea and Taiwanese Retro as Au Lait.
- Other Mosa Signature drinks remain excluded unless their category explicitly indicates Milk Tea or Au Lait.

## Tea Base Mapping Update
- Modifiers containing "jelly" are ignored for tea base assignment so toppings like Osmanthus Tie Guan Yin Jelly do not force a TGY base.

## Limitations
- Coverage bias: some fields (customer ID, modifiers) are incomplete; analyses using those fields apply to known records only.
- Rule-based mapping: tea base classification depends on menu naming consistency and needs updates as menu items change.
- Causal ambiguity: patterns (e.g., anomalies or co-purchases) are descriptive and do not prove cause-and-effect.
- Short horizon: three months of data limits seasonality conclusions.

## Recommended Next Steps
- Keep the rule-based mappings updated with menu changes.
- Track anomalies with contextual notes (events, promos, staffing changes).
- Extend to longer time windows to improve seasonality analysis.
