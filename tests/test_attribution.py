from __future__ import annotations

import json
import unittest
from pathlib import Path

from fingerprint import (
    AMBIGUOUS_MARGIN,
    CLOSE_MARGIN,
    CLOSE_TOP,
    analyze_global_outputs,
    closeness_report,
    load_bank,
)


def ranked(*pairs: tuple[str, float, str | None]) -> list[dict]:
    rows = []
    for item in pairs:
        model, probability, family = item if len(item) == 3 else (*item, None)
        rows.append(
            {
                "model": model,
                "display_name": model,
                "probability": probability,
                "family": family,
                "family_name": family.upper() if family else None,
            }
        )
    return rows


class ClosenessReportTests(unittest.TestCase):
    def test_decisive_requires_both_margin_and_top_mass(self):
        report = closeness_report(
            ranked(("gpt-6-sol", 0.82, "gpt"), ("gpt-5.5", 0.11, "gpt")),
            id_key="model",
            name_key="display_name",
            family_key="family",
        )
        self.assertEqual(report["verdict"], "decisive")
        self.assertAlmostEqual(report["margin"], 0.71)
        self.assertEqual(report["runner_up"], "gpt-5.5")
        self.assertTrue(report["same_family_top_two"])

    def test_close_when_top_is_only_about_half(self):
        report = closeness_report(
            ranked(("gpt-6-sol", 0.52, "gpt"), ("gpt-5.5", 0.20, "gpt")),
            id_key="model",
            name_key="display_name",
            family_key="family",
        )
        self.assertEqual(report["verdict"], "close")
        self.assertGreater(report["margin"], AMBIGUOUS_MARGIN)

    def test_ambiguous_when_top_two_are_within_eight_points(self):
        report = closeness_report(
            ranked(
                ("gpt-6-sol", 0.41, "gpt"),
                ("gpt-5.5", 0.39, "gpt"),
                ("claude-sonnet-5", 0.12, "claude"),
            ),
            id_key="model",
            name_key="display_name",
            family_key="family",
        )
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertEqual([item["id"] for item in report["contenders"]], ["gpt-6-sol", "gpt-5.5"])
        self.assertLess(report["margin"], AMBIGUOUS_MARGIN)

    def test_ambiguous_when_leading_probability_is_low(self):
        report = closeness_report(
            ranked(("gpt-5.5", 0.40, "gpt"), ("claude-sonnet-4-6", 0.22, "claude")),
            id_key="model",
            name_key="display_name",
            family_key="family",
        )
        self.assertEqual(report["verdict"], "ambiguous")
        self.assertFalse(report["same_family_top_two"])

    def test_single_candidate_is_decisive_if_mass_is_high(self):
        report = closeness_report(
            ranked(("only", 1.0, "gpt")),
            id_key="model",
            name_key="display_name",
        )
        self.assertEqual(report["verdict"], "decisive")
        self.assertIsNone(report["runner_up"])
        self.assertEqual(len(report["contenders"]), 1)

    def test_empty_ranked_raises(self):
        with self.assertRaises(ValueError):
            closeness_report([], id_key="model", name_key="display_name")


class AnalyzeOutputsClosenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = load_bank(Path("data/unified_bank.json"))
        rows = []
        with Path("data/gpt_reference.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("strict_valid") and row.get("model_id") == "gpt-5.5":
                    rows.append(row)
                    if len(rows) == 3:
                        break
        cls.outputs = [
            {"text": row["text"], "expected_count": row["requested_count"]} for row in rows
        ]
        if len(cls.outputs) < 3:
            raise AssertionError("need 3 saved gpt-5.5 reference rows")

    def test_saved_references_remain_a_decisive_closed_set_hit(self):
        result = analyze_global_outputs(self.outputs, self.bank)
        self.assertEqual(result["prediction"], "gpt-5.5")
        self.assertEqual(result["verdict"], "decisive")
        self.assertGreaterEqual(result["margin"], CLOSE_MARGIN)
        self.assertGreaterEqual(result["probability"], CLOSE_TOP)
        self.assertEqual(result["family_prediction"], "gpt")
        self.assertEqual(result["family_verdict"], "decisive")
        self.assertEqual(result["contenders"][0]["id"], "gpt-5.5")
        self.assertIn("family", result["contenders"][0])
        self.assertEqual(result["closeness_thresholds"]["ambiguous_margin"], AMBIGUOUS_MARGIN)

    def test_legacy_prediction_fields_are_unchanged(self):
        result = analyze_global_outputs(self.outputs, self.bank)
        self.assertEqual(result["prediction_name"], result["results"][0]["display_name"])
        self.assertAlmostEqual(result["probability"], result["results"][0]["probability"])
        self.assertGreaterEqual(result["used_outputs"], 1)

    def test_analyze_endpoint_includes_verdict_fields(self):
        from app import app

        client = app.test_client()
        response = client.post("/api/analyze", json={"outputs": self.outputs})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["verdict"], "decisive")
        self.assertIn("margin", payload)
        self.assertIn("runner_up", payload)
        self.assertIn("family_verdict", payload)
        self.assertEqual(payload["prediction"], "gpt-5.5")


if __name__ == "__main__":
    unittest.main()
