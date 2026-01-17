"""Basic smoke tests for module imports and expected functions."""
import tempfile
import unittest
from pathlib import Path

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
        self.assertTrue(hasattr(load_data, "_validate_detailed_export_schema"))

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

    def test_compute_product_mix_totals(self) -> None:
        df = pd.DataFrame(
            {
                "category_name": ["Coffee", "Coffee", "Coffee", "Tea"],
                "item_name": ["Latte", "Latte", "Drip", "Green"],
                "item_gross_sales": [6.0, 4.0, 10.0, 10.0],
            }
        )
        result = compute_sales_mix._compute_product_mix(df)
        latte = result[
            (result["category_name"] == "Coffee")
            & (result["item_name"] == "Latte")
        ].iloc[0]
        drip = result[
            (result["category_name"] == "Coffee") & (result["item_name"] == "Drip")
        ].iloc[0]
        green = result[
            (result["category_name"] == "Tea") & (result["item_name"] == "Green")
        ].iloc[0]

        self.assertAlmostEqual(latte["product_sales_pct_of_total"], 10.0 / 30.0)
        self.assertAlmostEqual(latte["product_sales_pct_of_category"], 10.0 / 20.0)
        self.assertAlmostEqual(drip["product_sales_pct_of_category"], 10.0 / 20.0)
        self.assertAlmostEqual(green["product_sales_pct_of_category"], 1.0)

    def test_normalize_columns_maps_variants(self) -> None:
        df = pd.DataFrame(
            {
                "Transaction ID": ["t1"],
                "Date": ["2025-01-01"],
                "Time": ["10:00"],
                "Category": ["Coffee"],
                "Item": ["Latte"],
                "Qty": [1],
                "Gross Sales": ["$5.00"],
            }
        )
        result = compute_sales_mix._build_order_datetime(
            compute_sales_mix._normalize_columns(df)
        )
        self.assertIn("order_id", result.columns)
        self.assertIn("order_datetime", result.columns)
        self.assertIn("category_name", result.columns)
        self.assertIn("item_name", result.columns)
        self.assertIn("quantity", result.columns)
        self.assertIn("item_gross_sales", result.columns)

    def test_coerce_sales_handles_missing_and_currency(self) -> None:
        df = pd.DataFrame({"item_gross_sales": ["$1,200.50", None, "0"]})
        result = compute_sales_mix._coerce_sales(df)
        self.assertEqual(result["item_gross_sales"].tolist(), [1200.50, 0.0, 0.0])

    def test_category_mix_empty_dataframe(self) -> None:
        df = pd.DataFrame(columns=["category_name", "item_gross_sales"])
        result = compute_sales_mix._compute_category_mix(df)
        self.assertEqual(
            list(result.columns),
            ["category_name", "total_sales", "category_sales_pct_of_total"],
        )
        self.assertTrue(result.empty)

    def test_category_mix_zero_total_sales(self) -> None:
        df = pd.DataFrame(
            {
                "category_name": ["Coffee", "Tea"],
                "item_gross_sales": [0.0, 0.0],
            }
        )
        result = compute_sales_mix._compute_category_mix(df)
        self.assertTrue((result["category_sales_pct_of_total"] == 0.0).all())

    def test_normalize_columns_strips_whitespace_and_case(self) -> None:
        df = pd.DataFrame(
            {
                "  cAtEgOrY  ": ["Coffee"],
                " ITEM ": ["Latte"],
                "  GROSS SALES  ": ["$3.00"],
                " tRaNsAcTiOn Id ": ["t1"],
                " daTe ": ["2025-01-01"],
                " tImE ": ["10:00"],
                " qTy ": [1],
            }
        )
        result = compute_sales_mix._build_order_datetime(
            compute_sales_mix._normalize_columns(df)
        )
        self.assertIn("category_name", result.columns)
        self.assertIn("item_name", result.columns)
        self.assertIn("item_gross_sales", result.columns)
        self.assertIn("order_id", result.columns)
        self.assertIn("order_datetime", result.columns)
        self.assertIn("quantity", result.columns)

    def test_build_order_datetime_from_date_time(self) -> None:
        df = pd.DataFrame(
            {
                "Date": ["2025-01-01"],
                "Time": ["10:15"],
            }
        )
        result = compute_sales_mix._build_order_datetime(df)
        self.assertIn("order_datetime", result.columns)
        self.assertFalse(result["order_datetime"].isna().any())

    def test_build_order_datetime_invalid_values(self) -> None:
        df = pd.DataFrame(
            {
                "Date": ["not-a-date"],
                "Time": ["not-a-time"],
            }
        )
        result = compute_sales_mix._build_order_datetime(df)
        self.assertIn("order_datetime", result.columns)
        self.assertTrue(result["order_datetime"].isna().all())

    def test_filter_refunds_keeps_panda(self) -> None:
        df = pd.DataFrame(
            {
                "Event Type": ["Payment", "Refund", "Refund"],
                "Notes": ["ok", "Panda", "Accidental Charge"],
                "item_name": ["A", "B", "C"],
            }
        )
        result = compute_sales_mix._filter_refunds(df)
        self.assertEqual(result["item_name"].tolist(), ["A", "B"])

    def test_filter_refunds_drops_canceled_orders(self) -> None:
        df = pd.DataFrame(
            {
                "Event Type": ["Payment", "Refund", "Payment"],
                "Notes": ["Canceled Order", "Panda", "canceled order"],
                "item_name": ["A", "B", "C"],
            }
        )
        result = compute_sales_mix._filter_refunds(df)
        self.assertEqual(result["item_name"].tolist(), ["B"])

    def test_validate_schema_accepts_date_time(self) -> None:
        df = pd.DataFrame(
            {
                "Transaction ID": ["t1"],
                "Date": ["2025-01-01"],
                "Time": ["10:00"],
                "Category": ["Coffee"],
                "Item": ["Latte"],
                "Qty": [1],
                "Gross Sales": [5.0],
            }
        )
        load_data._validate_detailed_export_schema(df)

    def test_validate_schema_rejects_missing_fields(self) -> None:
        df = pd.DataFrame(
            {
                "Date": ["2025-01-01"],
                "Time": ["10:00"],
                "Category": ["Coffee"],
                "Item": ["Latte"],
                "Qty": [1],
            }
        )
        with self.assertRaises(ValueError):
            load_data._validate_detailed_export_schema(df)

    def test_select_export_csv_prefers_orders_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "orders.csv").write_text("header\n", encoding="utf-8")
            (tmp_path / "other.csv").write_text("header\n", encoding="utf-8")
            selected = load_data._select_export_csv(tmp_path)
            self.assertEqual(selected.name, "orders.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
