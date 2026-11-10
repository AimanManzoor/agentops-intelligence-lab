import json
import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from src.agents.graph import build_graph
from src.observability.run_workflow import main, run_with_trace
from src.observability.tracing import RunTrace
from src.tools.incident_retrieval_tool import retrieve_similar_incidents
from src.tools.sql_investigation_tool import get_pipeline_evidence

HIGH = {"failure_probability": 0.9, "predicted_failure": True}
LOW = {"failure_probability": 0.01, "predicted_failure": False}


class RunTracingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.db = self.directory / "incidents.db"
        self.traces = self.directory / "runs"
        self.features = dict(expected_workload=5000, cpu_usage_pct=50.0,
                             memory_usage_pct=50.0, recent_failure_rate=0.1)
        self.request = {
            "pipeline_name": "payment_etl", "input_features": self.features,
            "as_of_utc": "2026-01-03T00:00:00+00:00",
        }
        with closing(sqlite3.connect(self.db)) as conn, conn:
            conn.execute(
                "CREATE TABLE realistic_incidents (run_id INTEGER PRIMARY KEY, "
                "pipeline_name TEXT, started_at TEXT, expected_workload INTEGER, "
                "cpu_usage_pct REAL, memory_usage_pct REAL, recent_failure_rate REAL, "
                "target_failed INTEGER, root_cause TEXT)"
            )
            conn.execute("INSERT INTO realistic_incidents VALUES (?,?,?,?,?,?,?,?,?)",
                         (1, "payment_etl", "2026-01-01", 5000, 50, 50, 0.1, 1, "PRIVATE_LABEL"))
        prediction = patch("src.agents.graph.predict_failure", return_value=HIGH)
        self.prediction = prediction.start()
        self.addCleanup(prediction.stop)
        sql = patch("src.agents.graph.get_pipeline_evidence", side_effect=self.evidence)
        self.sql = sql.start()
        self.addCleanup(sql.stop)
        retrieval = patch("src.agents.graph.retrieve_similar_incidents", side_effect=self.retrieve)
        self.retrieval = retrieval.start()
        self.addCleanup(retrieval.stop)

    def evidence(self, name, *, as_of):
        return get_pipeline_evidence(name, as_of=as_of, db_path=self.db)

    def retrieve(self, name, features, *, as_of, k):
        return retrieve_similar_incidents(name, features, as_of=as_of, k=k, db_path=self.db)

    def run_traced(self, request=None):
        result, path = run_with_trace(
            self.request if request is None else request, trace_dir=self.traces,
        )
        return result, path, json.loads(path.read_text())

    def test_high_risk_records_order_timing_and_status_without_evidence_payloads(self):
        result, path, report = self.run_traced()
        self.assertEqual(result["investigation_status"], "complete")
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(str(UUID(report["run_id"])), path.stem)
        self.assertEqual(report["execution_status"], "completed")
        self.assertEqual(report["route_taken"], "risk_review")
        self.assertEqual(report["investigation_status"], "complete")
        self.assertIsNone(report["error_type"])
        self.assertEqual([e["node"] for e in report["events"]], [
            "validate_input", "prediction", "investigate", "retrieve_incidents", "risk_review",
        ])
        self.assertTrue(all(e["status"] == "completed" for e in report["events"]))
        start = datetime.fromisoformat(report["started_at_utc"])
        end = datetime.fromisoformat(report["finished_at_utc"])
        self.assertIsNotNone(start.utcoffset())
        self.assertGreaterEqual(end, start)
        for event in report["events"]:
            self.assertGreaterEqual(event["duration_ms"], 0)
            self.assertGreaterEqual(report["duration_ms"] + 1, event["duration_ms"])
        for excluded in ["PRIVATE_LABEL", "input_features", "cpu_usage_pct", str(self.db), "final_message"]:
            self.assertNotIn(excluded, path.read_text())

    def test_low_risk_trace_proves_both_evidence_tools_were_skipped(self):
        self.prediction.return_value = LOW
        _, _, report = self.run_traced()
        self.assertEqual(report["route_taken"], "normal_processing")
        self.assertEqual(report["investigation_status"], "not_requested")
        self.assertEqual([e["node"] for e in report["events"]], [
            "validate_input", "prediction", "normal_processing",
        ])
        self.sql.assert_not_called()
        self.retrieval.assert_not_called()

    def test_unavailable_sql_is_degraded_evidence_but_completed_execution(self):
        self.db.unlink()
        _, _, report = self.run_traced()
        self.assertEqual(report["execution_status"], "completed")
        self.assertEqual(report["investigation_status"], "unavailable")
        self.assertEqual([e["node"] for e in report["events"]], [
            "validate_input", "prediction", "investigate", "risk_review",
        ])
        self.assertEqual(report["events"][2]["status"], "degraded")
        self.assertEqual(report["tool_errors"], [
            {"tool": "sql_investigation", "error_type": "FileNotFoundError"},
        ])
        self.retrieval.assert_not_called()

    def test_partial_retrieval_records_structured_error_without_raw_message(self):
        self.retrieval.side_effect = sqlite3.OperationalError("PRIVATE_EXCEPTION_MESSAGE")
        _, path, report = self.run_traced()
        self.assertEqual(report["execution_status"], "completed")
        self.assertEqual(report["investigation_status"], "partial")
        self.assertEqual(report["events"][3]["node"], "retrieve_incidents")
        self.assertEqual(report["events"][3]["status"], "degraded")
        self.assertEqual(report["tool_errors"][0]["error_type"], "OperationalError")
        self.assertNotIn("PRIVATE_EXCEPTION_MESSAGE", path.read_text())

    def test_unhandled_model_error_saves_failed_node_and_reraises_original(self):
        error = FileNotFoundError("PRIVATE_MODEL_PATH")
        self.prediction.side_effect = error
        with self.assertRaises(FileNotFoundError) as caught:
            self.run_traced()
        self.assertIs(caught.exception, error)
        path, = self.traces.glob("*.json")
        report = json.loads(path.read_text())
        self.assertEqual(report["execution_status"], "failed")
        self.assertEqual(report["failed_node"], "prediction")
        self.assertEqual(report["error_type"], "FileNotFoundError")
        self.assertEqual([e["node"] for e in report["events"]], ["validate_input", "prediction"])
        self.assertEqual(report["events"][-1]["status"], "failed")
        self.assertNotIn("PRIVATE_MODEL_PATH", path.read_text())
        self.sql.assert_not_called()

    def test_validation_error_is_traced_before_prediction(self):
        with self.assertRaises(ValueError):
            self.run_traced({**self.request, "pipeline_name": ""})
        path, = self.traces.glob("*.json")
        report = json.loads(path.read_text())
        self.assertEqual(report["execution_status"], "failed")
        self.assertEqual(report["failed_node"], "validate_input")
        self.assertEqual(len(report["events"]), 1)
        self.prediction.assert_not_called()

    def test_tracing_preserves_result_and_runs_have_independent_files(self):
        original = build_graph().invoke(self.request)
        first, path1, report1 = self.run_traced()
        second, path2, report2 = self.run_traced()
        self.assertEqual(first, original)
        self.assertEqual(second, original)
        self.assertNotEqual(path1, path2)
        self.assertEqual(len(list(self.traces.glob("*.json"))), 2)
        self.assertEqual(len(report1["events"]), 5)
        self.assertEqual(len(report2["events"]), 5)
        self.assertEqual(json.loads(path1.read_text()), report1)

    def test_trace_write_failure_does_not_replace_workflow_exception(self):
        error = RuntimeError("workflow failure")
        self.prediction.side_effect = error
        with patch.object(RunTrace, "save", side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError) as caught:
                self.run_traced()
        self.assertIs(caught.exception, error)
        self.assertIn("Trace could not be saved: OSError", error.__notes__)

    def test_invalid_trace_directory_stops_before_prediction(self):
        self.traces.write_text("existing file")
        with self.assertRaises(FileExistsError):
            self.run_traced()
        self.prediction.assert_not_called()

    def test_cli_both_examples_produce_independent_high_and_low_traces(self):
        self.prediction.side_effect = lambda **features: (
            HIGH if features["expected_workload"] == 9000 else LOW
        )
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_workflow", "--example", "both",
                                       "--trace-dir", str(self.traces)]), redirect_stdout(output):
            main()
        reports = [json.loads(path.read_text()) for path in self.traces.glob("*.json")]
        self.assertEqual(len(reports), 2)
        self.assertEqual({r["route_taken"] for r in reports},
                         {"risk_review", "normal_processing"})
        self.assertTrue(all(r["execution_status"] == "completed" for r in reports))
        self.assertIn('"example": "high"', output.getvalue())
        self.assertIn('"example": "low"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
