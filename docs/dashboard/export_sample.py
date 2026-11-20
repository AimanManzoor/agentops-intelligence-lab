"""Regenerate only the seeded public synthetic dataset used by the dashboard."""
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingestion import generate_realistic_incidents as generator

with tempfile.TemporaryDirectory() as tmp:
    generator.DB_PATH = Path(tmp) / "synthetic.db"
    generator.main()
    with sqlite3.connect(generator.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM realistic_incidents ORDER BY run_id")]
    assert len(rows) == generator.NUM_RUNS
    destination = Path(__file__).with_name("data.js")
    keys = list(rows[0])
    compact = [[row[key] for key in keys] for row in rows]
    destination.write_text(
        "globalThis.AGENTOPS_DATA = " + json.dumps(compact, separators=(",", ":"))
        + ".map(row => Object.fromEntries(" + json.dumps(keys, separators=(",", ":"))
        + ".map((key, i) => [key, row[i]])));\n", encoding="utf-8"
    )
