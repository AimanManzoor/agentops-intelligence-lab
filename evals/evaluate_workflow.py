"""Offline workflow-contract evaluation with controlled predictions, not ML scoring."""

import argparse
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.observability.run_workflow import run_with_trace
from src.tools.incident_retrieval_tool import retrieve_similar_incidents
from src.tools.sql_investigation_tool import get_pipeline_evidence

REFERENCE = Path(__file__).with_name("workflow_scenarios_v1.json")
CUTOFF = "2026-01-03T00:00:00+00:00"
MODES = {"healthy", "low", "empty", "missing_database", "missing_table",
         "retrieval_error", "prediction_error", "invalid_request"}
FEATURES = dict(expected_workload=5000, cpu_usage_pct=50.0,
                memory_usage_pct=50.0, recent_failure_rate=0.1)
EXPECTED_KEYS = {"route", "investigation", "error", "failed_node", "calls",
                 "nodes", "event_statuses", "tool_errors", "total_runs",
                 "failed_runs", "source_ids", "cutoffs_match",
                 "message_contains", "message_excludes"}


def load_reference(path=REFERENCE):
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("suite_id") != "workflow_scenarios_v1"
            or not isinstance(value.get("cases"), list) or not value["cases"]):
        raise ValueError("Expected a non-empty workflow_scenarios_v1 reference")
    ids = set()
    for case in value["cases"]:
        if (not isinstance(case, dict) or set(case) != {"id", "mode", "expected"}
                or not isinstance(case["id"], str) or not case["id"]
                or case["id"] in ids or case["mode"] not in MODES
                or not isinstance(case["expected"], dict)
                or set(case["expected"]) != EXPECTED_KEYS):
            raise ValueError("Invalid or duplicate workflow scenario")
        ids.add(case["id"])
    return value, hashlib.sha256(raw).hexdigest()


def create_fixture(path):
    """Two prior runs, plus exact-cutoff, future and other-pipeline decoys."""
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE realistic_incidents (run_id INTEGER PRIMARY KEY, "
            "pipeline_name TEXT, started_at TEXT, expected_workload INTEGER, "
            "cpu_usage_pct REAL, memory_usage_pct REAL, recent_failure_rate REAL, "
            "target_failed INTEGER, root_cause TEXT)"
        )
        rows = [
            (1, "payment_etl", "2026-01-01", 1),
            (2, "payment_etl", "2026-01-02", 0),
            (3, "payment_etl", CUTOFF, 1),
            (4, "payment_etl", "2026-01-04", 1),
            (5, "another_pipeline", "2026-01-01", 1),
        ]
        connection.executemany(
            "INSERT INTO realistic_incidents VALUES (?,?,?,?,?,?,?,?,?)",
            [(i, name, timestamp, 5000, 50, 50, 0.1, failed,
              "Synthetic failure" if failed else None)
             for i, name, timestamp, failed in rows],
        )


def run_case(case):
    mode = case["mode"]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        database = root / "incidents.db"
        create_fixture(database)
        if mode == "missing_database":
            database.unlink()
        elif mode == "missing_table":
            with sqlite3.connect(database) as connection:
                connection.execute("DROP TABLE realistic_incidents")
        source_before = database.read_bytes() if database.exists() else None
        request = {"pipeline_name": "unknown" if mode == "empty" else "payment_etl",
                   "input_features": dict(FEATURES), "as_of_utc": CUTOFF}
        if mode == "invalid_request":
            request["pipeline_name"] = " "

        def sql(name, *, as_of):
            return get_pipeline_evidence(name, as_of=as_of, db_path=database)

        def retrieve(name, features, *, as_of, k):
            if mode == "retrieval_error":
                raise sqlite3.OperationalError("PRIVATE_FAULT_DETAIL")
            return retrieve_similar_incidents(
                name, features, as_of=as_of, k=k, db_path=database)

        prediction = {"failure_probability": 0.01 if mode == "low" else 0.9,
                      "predicted_failure": mode != "low"}
        result, error_type = {}, None
        with (
            patch("src.agents.graph.predict_failure", return_value=prediction,
                  side_effect=ValueError("PRIVATE_FAULT_DETAIL")
                  if mode == "prediction_error" else None) as predict,
            patch("src.agents.graph.get_pipeline_evidence", side_effect=sql) as evidence,
            patch("src.agents.graph.retrieve_similar_incidents", side_effect=retrieve) as retrieval,
        ):
            try:
                result, _ = run_with_trace(request, trace_dir=root / "traces")
            except Exception as error:
                # A failure is an observation; it passes only if independently expected.
                error_type = type(error).__name__
        trace_files = list((root / "traces").glob("*.json"))
        trace = json.loads(trace_files[0].read_text()) if len(trace_files) == 1 else {}
        events = trace.get("events", [])
        message = result.get("final_message", "")
        sql_result = result.get("evidence") or {}
        matches = (result.get("retrieval") or {}).get("matches", [])
        actual = {
            "route": result.get("route_taken"),
            "investigation": result.get("investigation_status"),
            "error": error_type, "failed_node": trace.get("failed_node"),
            "calls": [predict.call_count, evidence.call_count, retrieval.call_count],
            "nodes": [e["node"] for e in events],
            "event_statuses": [e["status"] for e in events],
            "tool_errors": result.get("tool_errors", []),
            "total_runs": sql_result.get("total_runs"),
            "failed_runs": sql_result.get("failed_runs"),
            "source_ids": [m["source_id"] for m in matches],
            "cutoffs_match": all(call.kwargs.get("as_of") == CUTOFF
                                 for tool in (evidence, retrieval)
                                 for call in tool.call_args_list),
        }
        checks = {key: actual[key] == expected for key, expected in case["expected"].items()
                  if key not in {"message_contains", "message_excludes"}}
        checks["message_contains"] = all(
            text in message for text in case["expected"]["message_contains"])
        checks["message_excludes"] = all(
            text not in message for text in case["expected"]["message_excludes"])
        checks["single_trace"] = len(trace_files) == 1
        checks["trace_outcome"] = (
            trace.get("execution_status") == ("failed" if error_type else "completed")
            and trace.get("error_type") == error_type
            and trace.get("route_taken") == actual["route"]
            and trace.get("investigation_status") == actual["investigation"]
            and trace.get("tool_errors") == actual["tool_errors"])
        serialized_trace = json.dumps(trace)
        checks["trace_omits_payloads"] = all(
            forbidden not in serialized_trace for forbidden in
            ["input_features", "final_message", "recorded_failure_labels",
             "PRIVATE_FAULT_DETAIL", str(database)])
        checks["source_unchanged"] = (
            (database.read_bytes() if database.exists() else None) == source_before)
        return {"id": case["id"], "passed": all(checks.values()),
                "checks": checks, "actual": actual,
                "failed_checks": [key for key, passed in checks.items() if not passed]}


def evaluate(reference_path=REFERENCE):
    reference, fingerprint = load_reference(reference_path)
    results = [run_case(case) for case in reference["cases"]]
    passed = sum(result["passed"] for result in results)
    return {"schema_version": 1, "suite_id": reference["suite_id"],
            "reference_sha256": fingerprint, "prediction_mode": "controlled_stub",
            "model_calls": 0, "scenario_count": len(results), "passed": passed,
            "failed": len(results) - passed, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    try:
        report = evaluate(args.reference)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "invalid_reference", "error_type": type(error).__name__}))
        return 2
    print(json.dumps(report, indent=2))
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
