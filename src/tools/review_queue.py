"""Bounded offline review selection, not a prediction or autonomous action."""
import argparse
import json
import math
from pathlib import Path

MAX_BATCH = 10_000
MAX_BYTES = 1_000_000


def build_review_queue(records, *, budget, min_probability=0.26):
    if type(budget) is not int or not 0 <= budget <= 1000:
        raise ValueError("Budget must be an integer between 0 and 1000.")
    if (type(min_probability) not in (int, float) or
            not 0 <= min_probability <= 1 or not math.isfinite(min_probability)):
        raise ValueError("Minimum probability must be finite and in [0, 1].")
    if not isinstance(records, list) or len(records) > MAX_BATCH:
        raise ValueError("Input must be a list of at most 10000 records.")
    seen, eligible = set(), []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"run_id", "failure_probability"}:
            raise ValueError("Each record must contain only run_id and failure_probability.")
        run_id, score = record["run_id"], record["failure_probability"]
        if (not isinstance(run_id, str) or not run_id or run_id != run_id.strip()
                or len(run_id) > 128 or not run_id.isprintable() or run_id in seen):
            raise ValueError("Run IDs must be unique printable strings of 1-128 characters without outer spaces.")
        if (type(score) not in (int, float) or not 0 <= score <= 1
                or not math.isfinite(score)):
            raise ValueError("Probabilities must be finite numbers in [0, 1].")
        seen.add(run_id)
        if score >= min_probability:
            eligible.append({"run_id": run_id, "failure_probability": score})
    ranked = sorted(eligible, key=lambda record: (
        -record["failure_probability"], record["run_id"]))
    selected = [
        {"priority_rank": rank, **record}
        for rank, record in enumerate(ranked[:budget], start=1)
    ]
    return {
        "policy": "bounded_probability_queue_v1",
        "scope": "one supplied batch; no persistent or daily budget",
        "budget": budget, "min_probability": min_probability,
        "input_records": len(records), "eligible_records": len(eligible),
        "selected_records": len(selected),
        "deferred_eligible_records": len(eligible) - len(selected),
        "below_threshold_records": len(records) - len(eligible),
        "human_review_required": True, "automatic_actions": 0,
        "selected": selected,
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not permitted.")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use four handcrafted scores, not trained predictions.")
    source.add_argument("--input", type=Path, help="JSON list containing only run_id and failure_probability.")
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--min-probability", type=float, default=.26)
    args = parser.parse_args()
    try:
        if args.demo:
            records = [
                {"run_id": "run-A", "failure_probability": .40},
                {"run_id": "run-C", "failure_probability": .90},
                {"run_id": "run-B", "failure_probability": .90},
                {"run_id": "run-D", "failure_probability": .20},
            ]
        else:
            with args.input.open("rb") as handle:
                raw = handle.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError("Input exceeds the 1 MB file limit.")
            records = json.loads(raw, object_pairs_hook=unique_object)
        report = build_review_queue(records, budget=args.budget,
                                    min_probability=args.min_probability)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Queue stopped: {exc}\n")
    report["score_source"] = "handcrafted_demo" if args.demo else "caller_supplied_unverified_scores"
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
