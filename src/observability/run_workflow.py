"""Run the workflow and save one local JSON trace per invocation."""

import argparse
import json
from pathlib import Path

from src.observability.tracing import RunTrace

ROOT = Path(__file__).resolve().parents[2]


def run_with_trace(request, *, trace_dir=ROOT / "runs"):
    from src.agents.graph import build_graph

    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trace = RunTrace(pipeline_name=request.get("pipeline_name"))
    try:
        result = build_graph(trace=trace).invoke(request)
    except Exception as error:
        try:
            trace.save(directory, error=error)
        except OSError as save_error:
            error.add_note(f"Trace could not be saved: {type(save_error).__name__}")
        raise
    return result, trace.save(directory, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["high", "low", "both"], default="both")
    parser.add_argument("--trace-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    examples = {
        "high": dict(expected_workload=9000, cpu_usage_pct=92.0,
                     memory_usage_pct=88.0, recent_failure_rate=0.20),
        "low": dict(expected_workload=100, cpu_usage_pct=15.0,
                    memory_usage_pct=20.0, recent_failure_rate=0.0),
    }
    selected = examples if args.example == "both" else {args.example: examples[args.example]}
    for name, features in selected.items():
        result, path = run_with_trace(
            {"pipeline_name": "payment_etl", "input_features": features},
            trace_dir=args.trace_dir,
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({
            "example": name,
            "run_id": report["run_id"],
            "execution_status": report["execution_status"],
            "route": result["route_taken"],
            "investigation_status": result["investigation_status"],
            "duration_ms": report["duration_ms"],
            "nodes": [event["node"] for event in report["events"]],
            "trace_file": str(path),
        }, indent=2))


if __name__ == "__main__":
    main()
