import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.graph import build_graph
from src.tools.incident_retrieval_tool import retrieve_similar_incidents
from src.tools.sql_investigation_tool import get_pipeline_evidence

CUTOFF = "2026-01-03T00:00:00+00:00"
HIGH_RISK = {"failure_probability": 0.9, "predicted_failure": True}


class WorkflowReliabilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "incidents.db"
        self.features = {
            "expected_workload": 5000, "cpu_usage_pct": 50.0,
            "memory_usage_pct": 50.0, "recent_failure_rate": 0.1,
        }
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "CREATE TABLE realistic_incidents (run_id INTEGER PRIMARY KEY, "
                "pipeline_name TEXT, started_at TEXT, expected_workload INTEGER, "
                "cpu_usage_pct REAL, memory_usage_pct REAL, recent_failure_rate REAL, "
                "target_failed INTEGER, root_cause TEXT)"
            )
            conn.executemany("INSERT INTO realistic_incidents VALUES (?,?,?,?,?,?,?,?,?)", [
                (1, "payment_etl", "2026-01-01", 5000, 50, 50, 0.1, 1, "Memory pressure"),
                (2, "payment_etl", "2026-01-02", 5000, 50, 50, 0.1, 0, None),
            ])
        self.request = {
            "pipeline_name": "payment_etl", "input_features": self.features,
            "as_of_utc": CUTOFF,
        }

    def evidence(self, name, *, as_of):
        return get_pipeline_evidence(name, as_of=as_of, db_path=self.db)

    def retrieve(self, name, features, *, as_of, k):
        return retrieve_similar_incidents(
            name, features, as_of=as_of, k=k, db_path=self.db,
        )

    def invoke(self, *, request=None, prediction=HIGH_RISK,
               sql_effect=None, retrieval_effect=None):
        with (
            patch("src.agents.graph.predict_failure", return_value=prediction),
            patch("src.agents.graph.get_pipeline_evidence",
                  side_effect=self.evidence if sql_effect is None else sql_effect) as sql,
            patch("src.agents.graph.retrieve_similar_incidents",
                  side_effect=self.retrieve if retrieval_effect is None else retrieval_effect) as retrieval,
        ):
            result = build_graph().invoke(self.request if request is None else request)
        return result, sql, retrieval

    def test_healthy_sources_return_complete_investigation(self):
        result, sql, retrieval = self.invoke()
        self.assertEqual(result["investigation_status"], "complete")
        self.assertEqual(result["tool_errors"], [])
        self.assertIn("50.00%", result["final_message"])
        self.assertIn("realistic_incidents:1", result["final_message"])
        self.assertEqual(result["prediction"], HIGH_RISK)
        sql.assert_called_once()
        retrieval.assert_called_once()

    def test_empty_history_is_complete_rather_than_unavailable(self):
        result, _, retrieval = self.invoke(
            request={**self.request, "pipeline_name": "unknown"},
        )
        self.assertEqual(result["investigation_status"], "complete")
        self.assertEqual(result["tool_errors"], [])
        self.assertEqual(result["evidence"]["total_runs"], 0)
        self.assertIsNone(result["evidence"]["historical_failure_rate"])
        self.assertEqual(result["retrieval"]["matches"], [])
        self.assertIn("No prior matching runs", result["final_message"])
        retrieval.assert_called_once()

    def test_missing_database_returns_unavailable_and_skips_retrieval(self):
        self.db.unlink()
        result, _, retrieval = self.invoke(request={
            **self.request,
            "evidence": {"total_runs": 100, "historical_failure_rate": 0.99},
            "retrieval": {"matches": [{"source_id": "stale:999"}]},
        })
        self.assertEqual(result["route_taken"], "risk_review")
        self.assertEqual(result["investigation_status"], "unavailable")
        self.assertEqual(result["prediction"], HIGH_RISK)
        self.assertIsNone(result["evidence"])
        self.assertIsNone(result["retrieval"])
        self.assertEqual(result["tool_errors"], [
            {"tool": "sql_investigation", "error_type": "FileNotFoundError"},
        ])
        self.assertIn("SQL evidence is unavailable", result["final_message"])
        self.assertNotIn("No prior matching runs", result["final_message"])
        self.assertNotIn("50.00%", result["final_message"])
        self.assertNotIn("stale:999", result["final_message"])
        self.assertNotIn(str(self.db), result["final_message"])
        retrieval.assert_not_called()
        self.assertFalse(self.db.exists())

    def test_missing_table_returns_structured_sql_error(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("DROP TABLE realistic_incidents")
        result, _, retrieval = self.invoke()
        self.assertEqual(result["investigation_status"], "unavailable")
        self.assertEqual(result["tool_errors"][0]["error_type"], "OperationalError")
        self.assertIsNone(result["evidence"])
        retrieval.assert_not_called()

    def test_retrieval_storage_errors_preserve_sql_without_inventing_matches(self):
        for error in [FileNotFoundError("missing"), sqlite3.OperationalError("locked")]:
            with self.subTest(error=type(error).__name__):
                result, _, retrieval = self.invoke(retrieval_effect=error)
                self.assertEqual(result["investigation_status"], "partial")
                self.assertEqual(result["prediction"], HIGH_RISK)
                self.assertEqual(result["evidence"]["total_runs"], 2)
                self.assertEqual(result["tool_errors"], [
                    {"tool": "incident_retrieval", "error_type": type(error).__name__},
                ])
                self.assertIsNone(result["retrieval"])
                self.assertIn("50.00%", result["final_message"])
                self.assertIn("retrieval is unavailable", result["final_message"])
                self.assertNotIn("No prior failures matched", result["final_message"])
                self.assertNotIn("realistic_incidents:", result["final_message"])
                retrieval.assert_called_once_with(
                    "payment_etl", self.features, as_of=CUTOFF, k=3,
                )

    def test_low_risk_skips_unavailable_tools(self):
        result, sql, retrieval = self.invoke(
            prediction={"failure_probability": 0.01, "predicted_failure": False},
            sql_effect=FileNotFoundError(), retrieval_effect=FileNotFoundError(),
        )
        self.assertEqual(result["route_taken"], "normal_processing")
        self.assertEqual(result["investigation_status"], "not_requested")
        self.assertEqual(result["tool_errors"], [])
        self.assertNotIn("evidence", result)
        self.assertNotIn("retrieval", result)
        sql.assert_not_called()
        retrieval.assert_not_called()

    def test_validation_and_programming_errors_still_propagate(self):
        for tool in ["sql", "retrieval"]:
            for error in [ValueError("invalid input"), RuntimeError("unexpected")]:
                with self.subTest(tool=tool, error=type(error).__name__):
                    options = {"sql_effect" if tool == "sql" else "retrieval_effect": error}
                    with self.assertRaises(type(error)):
                        self.invoke(**options)


if __name__ == "__main__":
    unittest.main()
