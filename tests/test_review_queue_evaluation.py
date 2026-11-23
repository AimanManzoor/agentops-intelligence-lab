import contextlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.ingestion import generate_realistic_incidents as generator
from evals.evaluate_review_queue import (
    FEATURES, dataset_parts, validate_protocol, select_baselines,
    evaluate_selection, average_reports, run_experiment, summary_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
SMALL = {
    "schema_version": 1, "training_seed": 42, "training_rows": 100,
    "batch_seeds": [2001, 2002], "batch_size": 10,
    "review_budget": 2, "random_seeds": [11, 22],
}


class ReviewQueueEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_experiment(SMALL)

    def test_generator_reproducibility(self):
        first = generator.generate_records(seed=42, num_runs=10)
        self.assertEqual(first, generator.generate_records(seed=42, num_runs=10))
        self.assertNotEqual(first, generator.generate_records(seed=43, num_runs=10))

    def test_generator_bounds(self):
        for seed, count in [(True, 10), (-1, 10), (42, 0), (42, 10001), (42, True)]:
            with self.assertRaises(ValueError):
                generator.generate_records(seed=seed, num_runs=count)

    def test_original_database_writer_uses_same_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "fixture.db"
            with patch.object(generator, "DB_PATH", database), patch.object(generator, "NUM_RUNS", 20):
                with contextlib.redirect_stdout(io.StringIO()):
                    generator.main()
            with sqlite3.connect(database) as conn:
                rows = conn.execute("SELECT * FROM realistic_incidents ORDER BY run_id").fetchall()
            self.assertEqual(rows, generator.generate_records(seed=42, num_runs=20))

    def test_invalid_protocols(self):
        validate_protocol(SMALL)
        cases = [
            {"training_seed": True}, {"schema_version": True}, {"review_budget": 11},
            {"batch_seeds": [42, 2002]}, {"batch_seeds": [2001, 2001]},
            {"random_seeds": []}, {"batch_size": 1}, {"extra": 1},
        ]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_protocol({**SMALL, **changes})

    def test_oracle_probability_and_root_cause_excluded(self):
        rows = generator.generate_records(seed=42, num_runs=1)
        ids, features, outcomes = dataset_parts(rows, namespace="test")
        modified = list(rows[0])
        modified[7], modified[9] = 999, "untrusted oracle information"
        other_ids, other_features, other_outcomes = dataset_parts([modified], namespace="test")
        self.assertEqual((ids, features, outcomes), (other_ids, other_features, other_outcomes))
        self.assertEqual(set(features[0]), set(FEATURES))

    def test_baselines_fixed_budget_and_order_invariance(self):
        ids = ["C", "A", "B"]
        workloads = {"A": 5, "B": 10, "C": 10}
        highest, random_ids = select_baselines(ids, workloads, budget=2, seed="fixed")
        self.assertEqual(highest, ["B", "C"])
        self.assertEqual(len(set(random_ids)), 2)
        self.assertEqual((highest, random_ids),
                         select_baselines(list(reversed(ids)), workloads, budget=2, seed="fixed"))

    def test_metric_counts_and_undefined_rates(self):
        row = evaluate_selection(["A", "B"], {"A": 1, "B": 0, "C": 1})
        self.assertEqual((row["failures_caught"], row["missed_failures"]), (1, 1))
        self.assertEqual((row["precision_at_budget"], row["failure_coverage"]), (.5, .5))
        self.assertIsNone(evaluate_selection([], {"A": 0})["precision_at_budget"])
        self.assertIsNone(evaluate_selection(["A"], {"A": 0})["failure_coverage"])
        for ids, outcomes in [(["A", "A"], {"A": 1}), (["B"], {"A": 1}), (["A"], {"A": True})]:
            with self.assertRaises(ValueError):
                evaluate_selection(ids, outcomes)

    def test_labels_and_oracle_changes_do_not_change_selection(self):
        original = generator.generate_records
        def altered(*, seed, num_runs):
            rows = original(seed=seed, num_runs=num_runs)
            if seed != SMALL["training_seed"]:
                rows = [tuple([*row[:7], 0.999, 1 - row[8], "changed outcome"]) for row in rows]
            return rows
        with patch("evals.evaluate_review_queue.generate_records", side_effect=altered):
            changed = run_experiment(SMALL)
        for first, second in zip(self.report["batches"], changed["batches"]):
            self.assertEqual(first["selected_run_ids"], second["selected_run_ids"])
            for method in ("risk_ranking", "highest_workload", "random_mean"):
                self.assertAlmostEqual(first["metrics"][method]["failures_caught"] +
                                       second["metrics"][method]["failures_caught"], 2)

    def test_equal_budgets_and_report_provenance(self):
        report = self.report
        self.assertEqual(report["protocol"], SMALL)
        self.assertFalse(report["deployed"])
        self.assertEqual(len(report["protocol_sha256"]), 64)
        for batch in report["batches"]:
            for metrics in batch["metrics"].values():
                self.assertEqual(metrics["selected"], SMALL["review_budget"])
            self.assertEqual(len(batch["random_draw_metrics"]), 2)
        self.assertIn("same logistic generator", summary_markdown(report))

    def test_macro_average_handles_undefined_coverage(self):
        result = average_reports([{"precision": .5, "coverage": None},
                                  {"precision": 1, "coverage": None}])
        self.assertEqual(result, {"precision": .75, "coverage": None})

    def test_cli_rejects_invalid_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.json"
            config.write_text(json.dumps({**SMALL, "batch_seeds": [42, 42]}))
            result = subprocess.run(
                [sys.executable, "-m", "evals.evaluate_review_queue", "--config", str(config)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("Evaluation stopped", result.stderr)


if __name__ == "__main__":
    unittest.main()
