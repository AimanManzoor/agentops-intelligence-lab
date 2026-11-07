import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "agentops.db"

SCALES = {
    "expected_workload": 10000.0,
    "cpu_usage_pct": 100.0,
    "memory_usage_pct": 100.0,
    "recent_failure_rate": 1.0,
}


def validate_features(features):
    if not isinstance(features, dict) or set(features) != set(SCALES):
        raise ValueError(f"Provide exactly these features: {list(SCALES)}")

    for name, scale in SCALES.items():
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
        if name == "expected_workload":
            if not isinstance(value, int):
                raise ValueError("expected_workload must be an integer")
        elif value > scale:
            raise ValueError(f"{name} must be between 0 and {scale}")


def retrieve_similar_incidents(pipeline_name, input_features, *,
                               as_of=None, k=3, db_path=DB_PATH):
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        raise ValueError("pipeline_name must be a non-empty string")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError("k must be an integer between 1 and 20")

    validate_features(input_features)
    pipeline_name = pipeline_name.strip()

    cutoff = (
        datetime.fromisoformat(as_of)
        if as_of is not None else datetime.now(timezone.utc)
    )
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    cutoff = cutoff.astimezone(timezone.utc).isoformat()

    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")

    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT run_id, started_at, expected_workload, cpu_usage_pct, "
            "memory_usage_pct, recent_failure_rate, root_cause "
            "FROM realistic_incidents WHERE pipeline_name = ? "
            "AND target_failed = 1 AND julianday(started_at) < julianday(?)",
            (pipeline_name, cutoff),
        ).fetchall()

    ranked = []
    for row in rows:
        features = {name: row[name] for name in SCALES}
        validate_features(features)

        distance = math.hypot(*(
            (features[name] - input_features[name]) / scale
            for name, scale in SCALES.items()
        ))

        ranked.append((distance, row["run_id"], {
            "source_id": f"realistic_incidents:{row['run_id']}",
            "run_id": row["run_id"],
            "started_at": row["started_at"],
            "distance": round(distance, 6),
            "features": features,
            "recorded_failure_label": row["root_cause"],
        }))

    ranked.sort(key=lambda item: (item[0], item[1]))

    return {
        "pipeline_name": pipeline_name,
        "data_type": "synthetic",
        "as_of_utc": cutoff,
        "method": "fixed-scale Euclidean distance; lower is closer",
        "feature_scales": dict(SCALES),
        "candidate_count": len(ranked),
        "matches": [item[2] for item in ranked[:k]],
    }


if __name__ == "__main__":
    result = retrieve_similar_incidents("payment_etl", {
        "expected_workload": 9000,
        "cpu_usage_pct": 92.0,
        "memory_usage_pct": 88.0,
        "recent_failure_rate": 0.20,
    })
    print(json.dumps(result, indent=2))
