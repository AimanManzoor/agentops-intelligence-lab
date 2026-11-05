import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.graph import build_graph
from src.tools.sql_investigation_tool import get_pipeline_evidence

CUTOFF = "2026-01-03T00:00:00+00:00"


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "test.db"
        conn = sqlite3.connect(self.db)
        with conn:
            conn.execute("""CREATE TABLE realistic_incidents (
                pipeline_name TEXT, started_at TEXT,
                target_failed INTEGER, root_cause TEXT)""")
            conn.executemany("INSERT INTO realistic_incidents VALUES (?,?,?,?)", [
                ("payment_etl", "2026-01-01T00:00:00", 1, "Memory pressure"),
                ("payment_etl", "2026-01-02T00:00:00", 0, None),
                ("payment_etl", "2026-01-03T00:00:00", 1, "CPU saturation"),
                ("inventory_sync", "2026-01-01T00:00:00", 1, "High workload"),
            ])
        conn.close()
        self.request = {
            "pipeline_name": "payment_etl",
            "input_features": {},
            "as_of_utc": CUTOFF,
        }

    def evidence(self, name, *, as_of=CUTOFF):
        return get_pipeline_evidence(name, as_of=as_of, db_path=self.db)

    def test_sql_scope_cutoff_and_no_writes(self):
        before = self.db.read_bytes()
        evidence = self.evidence("payment_etl")
        self.assertEqual(evidence["total_runs"], 2)
        self.assertEqual(evidence["failed_runs"], 1)
        self.assertEqual(evidence["historical_failure_rate"], 0.5)
        self.assertEqual(evidence["recorded_failure_labels"], [
            {"label": "Memory pressure", "failure_count": 1}
        ])
        self.assertEqual(self.db.read_bytes(), before)

    def test_unknown_and_sql_like_names_return_no_history(self):
        for name in ["unknown", "payment_etl' OR 1=1 --"]:
            with self.subTest(name=name):
                evidence = self.evidence(name)
                self.assertEqual(evidence["total_runs"], 0)
                self.assertIsNone(evidence["historical_failure_rate"])

    def test_high_risk_fetches_sql_evidence(self):
        prediction = {"failure_probability": 0.9, "predicted_failure": True}
        with patch("src.agents.graph.predict_failure", return_value=prediction):
            with patch("src.agents.graph.get_pipeline_evidence",
                       side_effect=self.evidence) as sql:
                result = build_graph().invoke(self.request)
        sql.assert_called_once_with("payment_etl", as_of=CUTOFF)
        self.assertEqual(result["route_taken"], "risk_review")
        self.assertEqual(result["evidence"]["failed_runs"], 1)
        self.assertIn("50.00%", result["final_message"])

    def test_low_risk_skips_sql(self):
        prediction = {"failure_probability": 0.01, "predicted_failure": False}
        with patch("src.agents.graph.predict_failure", return_value=prediction):
            with patch("src.agents.graph.get_pipeline_evidence") as sql:
                result = build_graph().invoke(self.request)
        sql.assert_not_called()
        self.assertEqual(result["route_taken"], "normal_processing")
        self.assertNotIn("evidence", result)


if __name__ == "__main__":
    unittest.main()
