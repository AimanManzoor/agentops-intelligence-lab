import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.graph import build_graph
from src.tools.incident_retrieval_tool import retrieve_similar_incidents
from src.tools.sql_investigation_tool import get_pipeline_evidence

CUTOFF = "2026-01-03T00:00:00+00:00"


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = Path(temp.name) / "test.db"
        self.features = dict(expected_workload=5000, cpu_usage_pct=50.0,
                             memory_usage_pct=50.0, recent_failure_rate=0.1)
        conn = sqlite3.connect(self.db)
        with conn:
            conn.execute(
                "CREATE TABLE realistic_incidents (run_id INTEGER PRIMARY KEY, "
                "pipeline_name TEXT, started_at TEXT, expected_workload INTEGER, "
                "cpu_usage_pct REAL, memory_usage_pct REAL, recent_failure_rate REAL, "
                "target_failed INTEGER, root_cause TEXT)"
            )
            conn.executemany("INSERT INTO realistic_incidents VALUES (?,?,?,?,?,?,?,?,?)", [
                (1, "payment_etl", "2026-01-01", 5000, 50, 50, 0.1, 1, "A"),
                (2, "payment_etl", "2026-01-01", 6000, 50, 50, 0.1, 1, "B"),
                (3, "payment_etl", "2026-01-01", 5000, 60, 50, 0.1, 1, "C"),
                (4, "other",       "2026-01-01", 5000, 50, 50, 0.1, 1, "A"),
                (5, "payment_etl", "2026-01-01", 5000, 50, 50, 0.1, 0, None),
                (6, "payment_etl", "2026-01-03", 5000, 50, 50, 0.1, 1, "A"),
            ])
        conn.close()

    def retrieve(self, name="payment_etl", features=None, *, as_of=CUTOFF, k=3):
        return retrieve_similar_incidents(
            name, self.features if features is None else features,
            as_of=as_of, k=k, db_path=self.db,
        )

    def summary(self, name, *, as_of=None):
        return get_pipeline_evidence(name, as_of=as_of, db_path=self.db)

    def test_ranking_scaling_filters_and_sources(self):
        result = self.retrieve()
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual([m["run_id"] for m in result["matches"]], [1, 2, 3])
        self.assertEqual([m["distance"] for m in result["matches"]], [0, 0.1, 0.1])
        self.assertEqual(result["matches"][0]["source_id"], "realistic_incidents:1")

    def test_empty_results_and_k_limits(self):
        for name in ["unknown", "payment_etl' OR 1=1 --"]:
            self.assertEqual(self.retrieve(name)["matches"], [])
        self.assertEqual(len(self.retrieve(k=1)["matches"]), 1)
        for k in [0, 21, True]:
            with self.assertRaises(ValueError):
                self.retrieve(k=k)

    def test_no_writes_and_labels_do_not_change_ranking(self):
        before = self.db.read_bytes()
        original = self.retrieve()["matches"]
        self.assertEqual(self.db.read_bytes(), before)
        conn = sqlite3.connect(self.db)
        with conn:
            conn.execute("UPDATE realistic_incidents SET root_cause='changed' WHERE run_id=1")
        conn.close()
        changed = self.retrieve()["matches"]
        self.assertEqual([(m["run_id"], m["distance"]) for m in original],
                         [(m["run_id"], m["distance"]) for m in changed])
        self.assertEqual(changed[0]["recorded_failure_label"], "changed")

    def test_graph_shares_cutoff_and_returns_references(self):
        prediction = {"failure_probability": 0.9, "predicted_failure": True}
        with (
            patch("src.agents.graph.predict_failure", return_value=prediction),
            patch("src.agents.graph.get_pipeline_evidence", side_effect=self.summary),
            patch("src.agents.graph.retrieve_similar_incidents",
                  side_effect=self.retrieve) as retriever,
        ):
            result = build_graph().invoke({
                "pipeline_name": "payment_etl", "input_features": self.features,
                "as_of_utc": "2026-01-03",
            })
        retriever.assert_called_once_with("payment_etl", self.features,
                                         as_of=CUTOFF, k=3)
        self.assertEqual(result["as_of_utc"], result["evidence"]["as_of_utc"])
        self.assertEqual(result["as_of_utc"], result["retrieval"]["as_of_utc"])
        for match in result["retrieval"]["matches"]:
            self.assertIn(match["source_id"], result["final_message"])


if __name__ == "__main__":
    unittest.main()
