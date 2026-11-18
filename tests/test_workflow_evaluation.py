import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.evaluate_workflow import evaluate, load_reference, run_case
from src.observability.tracing import RunTrace


class WorkflowEvaluationTests(unittest.TestCase):
    def test_versioned_suite_passes_all_eight_contracts(self):
        report = evaluate()
        self.assertEqual((report["scenario_count"], report["passed"], report["failed"]),
                         (8, 8, 0))
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(len(report["reference_sha256"]), 64)

    def test_wrong_reference_fails_and_changes_fingerprint(self):
        reference, original_hash = load_reference()
        reference["cases"][0]["expected"]["total_runs"] = 999
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            path.write_text(json.dumps(reference))
            report = evaluate(path)
        self.assertEqual(report["failed"], 1)
        self.assertIn("total_runs", report["results"][0]["failed_checks"])
        self.assertNotEqual(report["reference_sha256"], original_hash)

    def test_checker_detects_broken_routing(self):
        case = load_reference()[0]["cases"][0]
        with patch("src.agents.graph.route_by_risk", return_value="normal_processing"):
            report = run_case(case)
        self.assertFalse(report["passed"])
        self.assertIn("route", report["failed_checks"])
        self.assertIn("calls", report["failed_checks"])

    def test_checker_detects_payload_added_to_trace(self):
        original = RunTrace.report

        def leak(trace, **kwargs):
            return {**original(trace, **kwargs), "input_features": {"secret": 1}}

        with patch.object(RunTrace, "report", leak):
            report = run_case(load_reference()[0]["cases"][0])
        self.assertIn("trace_omits_payloads", report["failed_checks"])

    def test_checker_detects_evidence_cutoff_regression(self):
        from src.tools.sql_investigation_tool import get_pipeline_evidence

        def include_future(name, *, as_of, db_path):
            return get_pipeline_evidence(name, as_of="2026-01-05", db_path=db_path)

        with patch("evals.evaluate_workflow.get_pipeline_evidence", side_effect=include_future):
            report = run_case(load_reference()[0]["cases"][0])
        self.assertIn("total_runs", report["failed_checks"])
        self.assertIn("source_ids", report["failed_checks"])

    def test_empty_duplicate_and_incomplete_references_are_rejected(self):
        reference = load_reference()[0]
        variants = [dict(reference, cases=[]),
                    dict(reference, cases=[reference["cases"][0]] * 2)]
        incomplete = copy.deepcopy(reference)
        del incomplete["cases"][0]["expected"]["calls"]
        variants.append(incomplete)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            for variant in variants:
                with self.subTest(variant=variant):
                    path.write_text(json.dumps(variant))
                    with self.assertRaises(ValueError):
                        load_reference(path)

    def test_cli_exit_codes_for_pass_fail_and_invalid_reference(self):
        root = Path(__file__).resolve().parents[1]
        reference = load_reference()[0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.json"
            for expected_code in (0, 1, 2):
                if expected_code == 1:
                    reference["cases"][0]["expected"]["calls"] = [0, 0, 0]
                elif expected_code == 2:
                    reference["cases"] = []
                path.write_text(json.dumps(reference))
                process = subprocess.run(
                    [sys.executable, "-m", "evals.evaluate_workflow", "--reference", str(path)],
                    cwd=root, capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, expected_code, process.stderr)
                json.loads(process.stdout)


if __name__ == "__main__":
    unittest.main()
