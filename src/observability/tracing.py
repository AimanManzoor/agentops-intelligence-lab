"""Local, per-invocation traces for the synchronous AgentOps workflow."""

import json
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def tool_errors(value):
    """Record the structured contract, never raw exception messages."""
    if not isinstance(value, list):
        return []
    return [
        {key: item[key] for key in ("tool", "error_type")
         if isinstance(item.get(key), str)}
        for item in value if isinstance(item, dict)
    ]


class RunTrace:
    def __init__(self, pipeline_name=None):
        self.run_id = str(uuid4())
        self.started_at_utc = utc_now()
        self.started = perf_counter()
        self.pipeline_name = pipeline_name if isinstance(pipeline_name, str) else None
        self.events = []

    def wrap(self, name, function):
        @wraps(function)
        def observed(state):
            started = perf_counter()
            event = {"node": name, "started_at_utc": utc_now(), "status": "completed"}
            try:
                update = function(state)
            except Exception as error:
                event.update(status="failed", error_type=type(error).__name__)
                raise
            else:
                errors = tool_errors(update.get("tool_errors")) if isinstance(update, dict) else []
                event.update(status="degraded" if errors else "completed")
                if errors:
                    event["tool_errors"] = errors
                if isinstance(update, dict) and "investigation_status" in update:
                    event["investigation_status"] = update["investigation_status"]
                return update
            finally:
                event["duration_ms"] = round((perf_counter() - started) * 1000, 3)
                self.events.append(event)

        return observed

    def report(self, *, result=None, error=None):
        result = result or {}
        failed = next((e["node"] for e in self.events if e["status"] == "failed"), None)
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": utc_now(),
            "duration_ms": round((perf_counter() - self.started) * 1000, 3),
            "execution_status": "failed" if error is not None else "completed",
            "route_taken": result.get("route_taken"),
            "investigation_status": result.get("investigation_status"),
            "tool_errors": tool_errors(result.get("tool_errors")),
            "error_type": type(error).__name__ if error is not None else None,
            "failed_node": failed,
            "events": list(self.events),
        }

    def save(self, directory, *, result=None, error=None):
        path = Path(directory) / f"{self.run_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            json.dump(self.report(result=result, error=error), handle, indent=2)
            handle.write("\n")
        return path
