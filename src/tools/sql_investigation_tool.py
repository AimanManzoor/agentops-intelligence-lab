import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "agentops.db"


def get_pipeline_evidence(pipeline_name, *, as_of=None, db_path=DB_PATH):
    """Read synthetic incidents before a cutoff; naive timestamps mean UTC."""
    if not isinstance(pipeline_name, str) or not pipeline_name.strip():
        raise ValueError("pipeline_name must be a non-empty string")
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
        parameters = (pipeline_name, cutoff)

        summary = conn.execute("""
            SELECT COUNT(*) AS total_runs,
                   COALESCE(SUM(target_failed), 0) AS failed_runs
            FROM realistic_incidents
            WHERE pipeline_name = ?
              AND julianday(started_at) < julianday(?)
        """, parameters).fetchone()

        labels = conn.execute("""
            SELECT COALESCE(root_cause, 'Unspecified') AS label,
                   COUNT(*) AS failure_count
            FROM realistic_incidents
            WHERE pipeline_name = ?
              AND julianday(started_at) < julianday(?)
              AND target_failed = 1
            GROUP BY COALESCE(root_cause, 'Unspecified')
            ORDER BY failure_count DESC, label ASC
            LIMIT 5
        """, parameters).fetchall()

    total = summary["total_runs"]
    failed = summary["failed_runs"]

    return {
        "pipeline_name": pipeline_name,
        "source_table": "realistic_incidents",
        "data_type": "synthetic",
        "as_of_utc": cutoff,
        "total_runs": total,
        "failed_runs": failed,
        "historical_failure_rate": round(failed / total, 4) if total else None,
        "recorded_failure_labels": [dict(row) for row in labels],
    }


if __name__ == "__main__":
    print(json.dumps(get_pipeline_evidence("payment_etl"), indent=2))
