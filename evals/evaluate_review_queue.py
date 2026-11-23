"""Frozen-protocol synthetic ranking experiment; never deploys a queue or model."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from src.ingestion.generate_realistic_incidents import generate_records
from src.tools.review_queue import build_review_queue

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["expected_workload", "cpu_usage_pct", "memory_usage_pct", "recent_failure_rate"]
METHODS = ("risk_ranking", "highest_workload", "random_mean")


def validate_protocol(config):
    required = {"schema_version", "training_seed", "training_rows", "batch_seeds",
                "batch_size", "review_budget", "random_seeds"}
    if not isinstance(config, dict) or set(config) != required or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Expected the version 1 protocol with exactly the documented fields.")
    bounds = {"training_seed": (0, 2**32 - 1), "training_rows": (100, 10000),
              "batch_size": (2, 10000), "review_budget": (1, 1000)}
    for key, (low, high) in bounds.items():
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid protocol field: {key}")
    if config["review_budget"] > config["batch_size"]:
        raise ValueError("Review budget cannot exceed batch size.")
    for key, maximum in (("batch_seeds", 30), ("random_seeds", 20)):
        values = config[key]
        if (not isinstance(values, list) or not 2 <= len(values) <= maximum
                or any(type(v) is not int or not 0 <= v < 2**32 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Invalid unique seed list: {key}")
    if config["training_seed"] in config["batch_seeds"]:
        raise ValueError("Evaluation seeds must exclude the training seed.")


def dataset_parts(rows, *, namespace):
    # Ignore stored generator probability (index 7), root cause (9) and time.
    ids = [f"{namespace}-run-{row[0]:05d}" for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated generated run IDs.")
    features = [dict(zip(FEATURES, row[3:7])) for row in rows]
    outcomes = {run_id: row[8] for run_id, row in zip(ids, rows)}
    return ids, features, outcomes


def select_baselines(ids, workloads, *, budget, seed):
    if type(budget) is not int or not 0 <= budget <= len(ids):
        raise ValueError("Invalid baseline budget.")
    if len(set(ids)) != len(ids) or set(ids) != set(workloads):
        raise ValueError("Baseline identifiers and workload keys must match uniquely.")
    highest = sorted(ids, key=lambda run_id: (-workloads[run_id], run_id))[:budget]
    randomized = random.Random(seed).sample(sorted(ids), budget)
    return highest, randomized


def evaluate_selection(selected, outcomes):
    if (len(set(selected)) != len(selected) or not set(selected).issubset(outcomes)
            or any(type(y) is not int or y not in (0, 1) for y in outcomes.values())):
        raise ValueError("Unique known selections and complete binary outcomes required.")
    total_failures = sum(outcomes.values())
    caught = sum(outcomes[run_id] for run_id in selected)
    return {
        "selected": len(selected), "total_failures": total_failures,
        "failures_caught": caught, "missed_failures": total_failures - caught,
        "precision_at_budget": caught / len(selected) if selected else None,
        "failure_coverage": caught / total_failures if total_failures else None,
    }


def average_reports(reports):
    result = {}
    for key in reports[0]:
        values = [row[key] for row in reports if row[key] is not None]
        result[key] = statistics.mean(values) if values else None
    return result


def run_experiment(config):
    validate_protocol(config)
    import platform
    from importlib.metadata import version
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    training = generate_records(seed=config["training_seed"], num_runs=config["training_rows"])
    ids, features, outcomes = dataset_parts(training, namespace="training")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(pd.DataFrame(features, columns=FEATURES), [outcomes[run_id] for run_id in ids])
    failure_column = list(model.classes_).index(1)
    batches = []
    budget = config["review_budget"]
    for batch_seed in config["batch_seeds"]:
        rows = generate_records(seed=batch_seed, num_runs=config["batch_size"])
        ids, features, outcomes = dataset_parts(rows, namespace=f"batch-{batch_seed}")
        scores = model.predict_proba(pd.DataFrame(features, columns=FEATURES))[:, failure_column]
        # The selector receives no outcomes, generator probabilities or workload fields.
        queue = build_review_queue([
            {"run_id": run_id, "failure_probability": float(score)}
            for run_id, score in zip(ids, scores)
        ], budget=budget, min_probability=0.0)
        risk_ids = [row["run_id"] for row in queue["selected"]]
        workloads = {run_id: feature["expected_workload"]
                     for run_id, feature in zip(ids, features)}
        randomized, random_selections, highest_ids = [], [], None
        for random_seed in config["random_seeds"]:
            highest_ids, random_ids = select_baselines(
                ids, workloads, budget=budget, seed=f"{batch_seed}:{random_seed}")
            random_selections.append(random_ids)
            randomized.append(evaluate_selection(random_ids, outcomes))
        result = {
            "risk_ranking": evaluate_selection(risk_ids, outcomes),
            "highest_workload": evaluate_selection(highest_ids, outcomes),
            "random_mean": average_reports(randomized),
        }
        if any(row["selected"] != budget for row in result.values()):
            raise ValueError("Unequal review budgets detected.")
        batches.append({
            "seed": batch_seed,
            "dataset_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
            "metrics": result, "random_draw_metrics": randomized,
            "selected_run_ids": {"risk_ranking": risk_ids, "highest_workload": highest_ids,
                                 "random_draws": random_selections},
        })
    summary = {method: average_reports([b["metrics"][method] for b in batches])
               for method in METHODS}
    paired = {}
    for baseline in ("highest_workload", "random_mean"):
        differences = [b["metrics"]["risk_ranking"]["precision_at_budget"] -
                       b["metrics"][baseline]["precision_at_budget"] for b in batches]
        paired[baseline] = {
            "mean_precision_difference": statistics.mean(differences),
            "batch_difference_range": [min(differences), max(differences)],
            "batches_risk_strictly_better": sum(value > 0 for value in differences),
        }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "same_generator_independent_seed_synthetic_experiment",
        "protocol": config,
        "protocol_sha256": hashlib.sha256(canonical).hexdigest(),
        "generator_sha256": hashlib.sha256(
            (ROOT / "src/ingestion/generate_realistic_incidents.py").read_bytes()).hexdigest(),
        "model": "fixed_standardized_logistic_regression",
        "versions": {"python": platform.python_version(), "scikit_learn": version("scikit-learn")},
        "ranking_score_floor": 0.0, "deployed": False,
        "summary": summary, "paired_precision_comparisons": paired, "batches": batches,
    }


def summary_markdown(report):
    protocol = report["protocol"]
    lines = [
        "# Review queue ranking results", "",
        f"Fixed budget: {protocol['review_budget']} of {protocol['batch_size']} runs per batch.",
        f"Evaluation: {len(protocol['batch_seeds'])} independent generator seeds; "
        f"{len(protocol['random_seeds'])} random draws per batch.", "",
        "| Method | Mean reviewed | Mean failures caught | Mean precision at budget | Mean failure coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["summary"][method]
        coverage = "N/A" if row["failure_coverage"] is None else f"{row['failure_coverage']:.1%}"
        lines.append(f"| {method} | {row['selected']:.0f} | {row['failures_caught']:.2f} | "
                     f"{row['precision_at_budget']:.1%} | {coverage} |")
    lines += ["", "Metrics are macro-averages across batches; random metrics first average "
              "the fixed draws within each batch. These are not confidence intervals.",
              "", "## Paired comparisons", ""]
    for baseline, comparison in report["paired_precision_comparisons"].items():
        low, high = comparison["batch_difference_range"]
        lines.append(
            f"- Risk minus {baseline}: mean precision difference "
            f"{100 * comparison['mean_precision_difference']:.2f} percentage points; "
            f"observed batch range {100 * low:.2f} to {100 * high:.2f}; "
            f"risk strictly better in {comparison['batches_risk_strictly_better']} batches.")
    lines += [
        "", "## Limits",
        "All data are synthetic draws from the same logistic generator, which favors "
        "logistic regression. New seeds do not establish domain-shift robustness, "
        "future-time validity, causality, production gains or real-world performance.",
        "All scores in a batch are assumed available before selection. Score floor "
        "0.0 is used only in this offline equal-budget experiment; the live workflow "
        "and its threshold remain unchanged.",
        "", f"Protocol SHA-256: {report['protocol_sha256']}",
        "", f"Generator SHA-256: {report['generator_sha256']}",
        "", f"Versions: {json.dumps(report['versions'], sort_keys=True)}",
        "", "Reproduce: python -m evals.evaluate_review_queue --write-summary",
        "", "See [frozen protocol and methodology](review_queue_evaluation.md).",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "evals/review_queue_protocol_v1.json")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        report = run_experiment(config)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Evaluation stopped: {exc}\n")
    destination = ROOT / "evals/results/review_queue_evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    markdown = summary_markdown(report)
    if args.write_summary:
        (ROOT / "docs/review_queue_results.md").write_text(markdown)
    print(markdown)
    print("Detailed local report:", destination)


if __name__ == "__main__":
    main()
