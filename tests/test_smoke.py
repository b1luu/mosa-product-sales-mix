"""Basic smoke tests for module imports and expected functions."""
import unittest

import pandas as pd

from src import compute_sales_mix
from src import load_data


class TestSmoke(unittest.TestCase):
    def test_compute_sales_mix_helpers_exist(self) -> None:
        self.assertTrue(hasattr(compute_sales_mix, "_normalize_columns"))
        self.assertTrue(hasattr(compute_sales_mix, "_compute_category_mix"))
        self.assertTrue(hasattr(compute_sales_mix, "_compute_product_mix"))

    def test_load_data_helpers_exist(self) -> None:
        self.assertTrue(hasattr(load_data, "_select_export_csv"))
        self.assertTrue(hasattr(load_data, "load_square_exports"))

    def test_compute_category_mix_totals(self) -> None:
        df = pd.DataFrame(
            {
                "category_name": ["Coffee", "Coffee", "Tea"],
                "item_gross_sales": [10.0, 5.0, 5.0],
            }
        )
        result = compute_sales_mix._compute_category_mix(df)
        totals = dict(zip(result["category_name"], result["total_sales"]))
        pct = dict(
            zip(
                result["category_name"],
                result["category_sales_pct_of_total"],
            )
        )
        self.assertEqual(totals["Coffee"], 15.0)
        self.assertEqual(totals["Tea"], 5.0)
        self.assertAlmostEqual(pct["Coffee"], 0.75)
        self.assertAlmostEqual(pct["Tea"], 0.25)


if __name__ == "__main__":
    unittest.main()
