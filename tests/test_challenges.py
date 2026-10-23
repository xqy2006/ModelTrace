from __future__ import annotations

import unittest

from fingerprint import generate_challenges, normalize_challenge_options


class ChallengePackTests(unittest.TestCase):
    def test_default_is_standard_chinese(self):
        challenges = generate_challenges(3)
        self.assertEqual(len(challenges), 3)
        counts = [item["expected_count"] for item in challenges]
        self.assertEqual(len(set(counts)), 3)
        for item in challenges:
            self.assertEqual(item["style"], "standard")
            self.assertEqual(item["language"], "zh")
            self.assertIn(str(item["expected_count"]), item["prompt"])
            self.assertIn("1 到 355", item["prompt"])
            self.assertIn("Python、代码执行器", item["prompt"])
            self.assertGreater(len(item["prompt"]), 120)

    def test_compact_chinese_is_short_and_asks_to_stop(self):
        challenges = generate_challenges(3, style="compact", language="zh")
        for item in challenges:
            prompt = item["prompt"]
            self.assertEqual(item["style"], "compact")
            self.assertTrue(prompt.startswith("随机生成 "))
            self.assertIn("不使用外部工具", prompt)
            self.assertIn("立即停止", prompt)
            self.assertNotIn("Python、代码执行器", prompt)
            self.assertLess(len(prompt), 120)
            self.assertEqual(prompt.count(str(item["expected_count"])), 2)

    def test_compact_english_stops_after_the_requested_count(self):
        challenges = generate_challenges(3, "compact", "en")
        for item in challenges:
            self.assertEqual(item["language"], "en")
            self.assertIn(f"Randomly choose {item['expected_count']} integers from 1 to 355", item["prompt"])
            self.assertIn(f"Stop immediately after {item['expected_count']} integers", item["prompt"])
            self.assertNotIn("code interpreter", item["prompt"])

    def test_standard_english_keeps_the_full_instruction(self):
        challenges = generate_challenges(3, style="standard", language="en")
        for item in challenges:
            self.assertIn("integers from 1 to 355 inclusive", item["prompt"])
            self.assertIn("do not write or run code first", item["prompt"])
            self.assertGreater(len(item["prompt"]), 200)

    def test_unknown_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "风格"):
            generate_challenges(3, style="telegraph")
        with self.assertRaisesRegex(ValueError, "语言"):
            generate_challenges(3, language="ja")
        with self.assertRaisesRegex(ValueError, "风格"):
            normalize_challenge_options("TELEGRAPH", "zh")

    def test_challenges_endpoint_accepts_query_and_rejects_unknown(self):
        from app import app

        client = app.test_client()
        ok = client.get("/api/challenges?style=compact&language=en")
        self.assertEqual(ok.status_code, 200)
        payload = ok.get_json()
        self.assertEqual(payload["style"], "compact")
        self.assertEqual(payload["language"], "en")
        self.assertEqual(len(payload["challenges"]), 3)
        self.assertTrue(payload["challenges"][0]["prompt"].startswith("Randomly choose "))

        default = client.get("/api/challenges")
        self.assertEqual(default.get_json()["challenges"][0]["style"], "standard")
        self.assertEqual(default.get_json()["challenges"][0]["language"], "zh")

        bad = client.get("/api/challenges?style=telegraph")
        self.assertEqual(bad.status_code, 400)
        self.assertIn("风格", bad.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
