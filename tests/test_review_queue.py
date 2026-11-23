import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.tools.review_queue import build_review_queue, MAX_BATCH

ROOT = Path(__file__).resolve().parents[1]


class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"run_id": "C", "failure_probability": .9},
            {"run_id": "A", "failure_probability": .4},
            {"run_id": "B", "failure_probability": .9},
            {"run_id": "D", "failure_probability": .2},
        ]

    def test_budget_and_deterministic_ranking(self):
        result = build_review_queue(self.records, budget=2)
        self.assertEqual([r["run_id"] for r in result["selected"]], ["B", "C"])
        self.assertEqual([r["priority_rank"] for r in result["selected"]], [1, 2])
        self.assertEqual(result["deferred_eligible_records"], 1)
        self.assertTrue(result["human_review_required"])
        self.assertEqual(result["automatic_actions"], 0)

    def test_empty_zero_and_excess_capacity(self):
        self.assertEqual(build_review_queue([], budget=2)["selected_records"], 0)
        self.assertEqual(build_review_queue(self.records, budget=0)["selected_records"], 0)
        result = build_review_queue(self.records, budget=10, min_probability=.4)
        self.assertEqual(result["selected_records"], 3)
        self.assertEqual(result["below_threshold_records"], 1)

    def test_invalid_probabilities(self):
        for score in [math.nan, math.inf, -.1, 1.1, True, "0.9", None]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                build_review_queue([{"run_id": "A", "failure_probability": score}], budget=0)

    def test_invalid_policy_parameters(self):
        for budget in [-1, 1001, 1.2, True]:
            with self.assertRaises(ValueError):
                build_review_queue([], budget=budget)
        for threshold in [-.1, 1.1, math.nan, math.inf, True, "0.2"]:
            with self.assertRaises(ValueError):
                build_review_queue([], budget=1, min_probability=threshold)

    def test_labels_extra_fields_and_duplicate_ids_rejected(self):
        cases = [
            [self.records[0], self.records[0]],
            [{"run_id": "A", "failure_probability": .8, "target_failed": 1}],
            [{"run_id": "A"}],
        ]
        for records in cases:
            with self.assertRaises(ValueError):
                build_review_queue(records, budget=1)

    def test_order_independence_and_input_preservation(self):
        before = copy.deepcopy(self.records)
        self.assertEqual(build_review_queue(self.records, budget=1),
                         build_review_queue(list(reversed(self.records)), budget=1))
        self.assertEqual(before, self.records)

    def test_batch_and_identifier_bounds(self):
        for records in [{}, [None] * (MAX_BATCH + 1)]:
            with self.assertRaises(ValueError):
                build_review_queue(records, budget=1)
        for run_id in ["", " A", "A" + chr(10), "A" * 129, 1]:
            with self.assertRaises(ValueError):
                build_review_queue([{"run_id": run_id, "failure_probability": .8}], budget=1)

    def test_cli_demo_and_invalid_json_file(self):
        command = [sys.executable, "-m", "src.tools.review_queue"]
        result = subprocess.run(command + ["--demo", "--budget", "2"], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["score_source"], "handcrafted_demo")
        self.assertEqual([r["run_id"] for r in report["selected"]], ["run-B", "run-C"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.json"
            path.write_text(json.dumps(self.records))
            valid = subprocess.run(command + ["--input", str(path), "--budget", "1"],
                                   cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertEqual(json.loads(valid.stdout)["selected"][0]["run_id"], "B")
            for text in ['[{"run_id":"A","run_id":"B","failure_probability":0.8}]',
                         '[{"run_id":"A","failure_probability":NaN}]', "x" * 1_000_001]:
                path.write_text(text)
                result = subprocess.run(command + ["--input", str(path)], cwd=ROOT,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
