
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Resolve the project directory regardless of where Python runs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "agentops.db"

# Fixed seed ensures reproducible random values.
random.seed(42)

PIPELINES = [
    "customer_ingestion",
    "payment_etl",
    "inventory_sync",
    "event_processing",
    "analytics_refresh",
]

def generate_records(n=5000):
    records = []
    start_date = datetime(2026, 1, 1)

    for run_id in range(1, n + 1):
        pipeline = random.choice(PIPELINES)
        started_at = start_date + timedelta(hours=run_id)

        # Simulated pre-execution workload.
        workload = random.randint(100, 10000)

        # Simple baseline failure probability.
        failure_probability = 0.05
        if workload > 8000:
            failure_probability += 0.15

        failed = random.random() < failure_probability

        duration = round(
            workload / 100 + random.uniform(5, 50),
            2
        )
        error_count = random.randint(1, 15) if failed else 0
        status = "Failed" if failed else "Success"
        error_message = (
            "Pipeline processing error" if failed else None
        )

        records.append((
            run_id,
            pipeline,
            started_at.isoformat(),
            duration,
            workload,
            error_count,
            status,
            error_message,
        ))

    return records

def main():
    records = generate_records()

    with sqlite3.connect(DATABASE_PATH) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO pipeline_runs
            (run_id, pipeline_name, started_at,
             duration_seconds, records_processed,
             error_count, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

        total = conn.execute(
            "SELECT COUNT(*) FROM pipeline_runs"
        ).fetchone()[0]

    print(f"Generated {len(records)} records")
    print(f"Total database records: {total}")
    print(f"Database: {DATABASE_PATH}")

if __name__ == "__main__":
    main()
