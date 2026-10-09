
import random
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "agentops.db"

random.seed(42)

def main():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_features (
                run_id INTEGER PRIMARY KEY,
                expected_workload INTEGER,
                cpu_usage_pct REAL,
                memory_usage_pct REAL,
                recent_failure_rate REAL,
                target_failed INTEGER NOT NULL,
                FOREIGN KEY (run_id)
                    REFERENCES pipeline_runs(run_id)
            )
        """)

        runs = conn.execute("""
            SELECT run_id, status
            FROM pipeline_runs
            ORDER BY run_id
        """).fetchall()

        records = []

        for run_id, status in runs:
            expected_workload = random.randint(100, 10000)
            cpu_usage = round(random.uniform(20, 95), 2)
            memory_usage = round(random.uniform(25, 90), 2)
            recent_failure_rate = round(
                random.uniform(0, 0.25), 3
            )

            target_failed = 1 if status == "Failed" else 0

            records.append((
                run_id,
                expected_workload,
                cpu_usage,
                memory_usage,
                recent_failure_rate,
                target_failed
            ))

        conn.executemany("""
            INSERT OR REPLACE INTO prediction_features
            VALUES (?, ?, ?, ?, ?, ?)
        """, records)

    print(f"Generated {len(records)} prediction records")

if __name__ == "__main__":
    main()
