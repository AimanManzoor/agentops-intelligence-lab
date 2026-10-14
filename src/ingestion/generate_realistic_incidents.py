
import math
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "agentops.db"
SEED = 42
NUM_RUNS = 5000

PIPELINES = [
    "customer_ingestion",
    "payment_etl",
    "inventory_sync",
    "event_processing",
    "analytics_refresh",
]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main():
    rng = random.Random(SEED)
    start_date = datetime(2026, 1, 1)
    records = []

    for i in range(1, NUM_RUNS + 1):
        pipeline = rng.choice(PIPELINES)
        workload = rng.randint(100, 10000)
        cpu = round(rng.uniform(15, 98), 2)
        memory = round(rng.uniform(20, 97), 2)
        recent_failure_rate = round(rng.uniform(0, 0.30), 3)

        # Features are available before the run.
        risk_score = (
            -4.5
            + 0.00015 * workload
            + 0.025 * cpu
            + 0.025 * memory
            + 2.0 * recent_failure_rate
        )

        failure_probability = sigmoid(risk_score)
        failed = int(rng.random() < failure_probability)

        if failed:
            if memory >= 85:
                root_cause = "Memory pressure"
            elif cpu >= 85:
                root_cause = "CPU saturation"
            elif workload >= 8000:
                root_cause = "High workload"
            else:
                root_cause = "Transient processing error"
        else:
            root_cause = None

        started_at = (
            start_date + timedelta(hours=i)
        ).isoformat()

        records.append((
            i, pipeline, started_at, workload,
            cpu, memory, recent_failure_rate,
            round(failure_probability, 4),
            failed, root_cause
        ))

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS realistic_incidents (
                run_id INTEGER PRIMARY KEY,
                pipeline_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                expected_workload INTEGER,
                cpu_usage_pct REAL,
                memory_usage_pct REAL,
                recent_failure_rate REAL,
                failure_probability REAL,
                target_failed INTEGER,
                root_cause TEXT
            )
        """)

        conn.executemany("""
            INSERT OR REPLACE INTO realistic_incidents
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

        count = conn.execute(
            "SELECT COUNT(*) FROM realistic_incidents"
        ).fetchone()[0]

        failures = conn.execute(
            "SELECT SUM(target_failed) FROM realistic_incidents"
        ).fetchone()[0]

    print(f"Generated {count} realistic incidents")
    print(f"Failed runs: {failures}")
    print(f"Database: {DB_PATH}")

if __name__ == "__main__":
    main()
