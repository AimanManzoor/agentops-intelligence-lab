"""Offline synthetic workload-shift experiment; no deployment changes."""
import hashlib
import json
import random
from pathlib import Path

from evals.evaluate_review_queue import (
    FEATURES,
    METHODS,
    average_reports,
    dataset_parts,
    evaluate_selection,
    select_baselines,
    validate_protocol,
)
from src.ingestion.generate_realistic_incidents import (
    generate_records,
    sigmoid,
)
from src.tools.review_queue import build_review_queue

ROOT = Path(__file__).resolve().parents[1]

# Specify these choices before inspecting the experiment's results.
PROTOCOL = {
    "schema_version": 1,
    "training_seed": 42,
    "training_rows": 3500,
    "batch_seeds": list(range(2001, 2011)),
    "batch_size": 250,
    "review_budget": 50,
    "random_seeds": [11, 22, 33, 44, 55],
}
WORKLOAD_INCREASE = 3000


def scenario_rows(rows, *, seed, increase):
    """Change workload and redraw outcomes under the SAME probability rule."""
    if increase not in (0, WORKLOAD_INCREASE):
        raise ValueError("Unsupported workload scenario.")

    rng = random.Random(f"{seed}:paired-outcomes")
    result = []

    for row in rows:
        changed = list(row)
        changed[3] = min(10000, row[3] + increase)

        probability = sigmoid(
            -4.5
            + 0.00015 * changed[3]
            + 0.025 * changed[4]
            + 0.025 * changed[5]
            + 2.0 * changed[6]
        )

        changed[7] = round(probability, 4)
        changed[8] = int(rng.random() < probability)
        changed[9] = None  # Do not reuse old post-outcome cause labels.
        result.append(tuple(changed))

    return result


def run_experiment():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    validate_protocol(PROTOCOL)

    training = generate_records(
        seed=PROTOCOL["training_seed"],
        num_runs=PROTOCOL["training_rows"],
    )
    ids, features, outcomes = dataset_parts(training, namespace="training")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(
        pd.DataFrame(features, columns=FEATURES),
        [outcomes[run_id] for run_id in ids],
    )
    failure_column = list(model.classes_).index(1)

    scenarios = {}

    for name, increase in (
        ("baseline", 0),
        ("higher_workload", WORKLOAD_INCREASE),
    ):
        batches = []

        for seed in PROTOCOL["batch_seeds"]:
            base = generate_records(
                seed=seed,
                num_runs=PROTOCOL["batch_size"],
            )
            rows = scenario_rows(base, seed=seed, increase=increase)
            ids, features, outcomes = dataset_parts(
                rows, namespace=f"batch-{seed}"
            )

            scores = model.predict_proba(
                pd.DataFrame(features, columns=FEATURES)
            )[:, failure_column]

            queue = build_review_queue(
                [
                    {
                        "run_id": run_id,
                        "failure_probability": float(score),
                    }
                    for run_id, score in zip(ids, scores)
                ],
                budget=PROTOCOL["review_budget"],
                min_probability=0.0,
            )
            selected = [item["run_id"] for item in queue["selected"]]

            workloads = {
                run_id: feature["expected_workload"]
                for run_id, feature in zip(ids, features)
            }

            random_metrics = []
            for random_seed in PROTOCOL["random_seeds"]:
                highest, random_ids = select_baselines(
                    ids,
                    workloads,
                    budget=PROTOCOL["review_budget"],
                    seed=f"{seed}:{random_seed}",
                )
                random_metrics.append(
                    evaluate_selection(random_ids, outcomes)
                )

            metrics = {
                "risk_ranking": evaluate_selection(selected, outcomes),
                "highest_workload": evaluate_selection(highest, outcomes),
                "random_mean": average_reports(random_metrics),
            }

            if any(
                item["selected"] != PROTOCOL["review_budget"]
                for item in metrics.values()
            ):
                raise ValueError("Review budgets differ.")

            batches.append({"seed": seed, "metrics": metrics})

        scenarios[name] = {
            "summary": {
                method: average_reports(
                    [batch["metrics"][method] for batch in batches]
                )
                for method in METHODS
            },
            "batches": batches,
        }

    return {
        "protocol": PROTOCOL,
        "workload_increase": WORKLOAD_INCREASE,
        "workload_cap": 10000,
        "experiment_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "generator_sha256": hashlib.sha256(
            (ROOT / "src/ingestion/generate_realistic_incidents.py")
            .read_bytes()
        ).hexdigest(),
        "deployed": False,
        "scenarios": scenarios,
    }


def main():
    report = run_experiment()

    lines = [
        "# Workload-shift experiment",
        "",
        "One experimental model; 50 reviews per 250-run batch.",
        "Ten paired synthetic seeds; five random selections per batch.",
        "",
        "| Scenario | Method | Mean failures caught | Precision at budget | Failure coverage |",
        "|---|---|---:|---:|---:|",
    ]

    for scenario, result in report["scenarios"].items():
        for method, metrics in result["summary"].items():
            coverage = metrics["failure_coverage"]
            coverage_text = "N/A" if coverage is None else f"{coverage:.1%}"
            lines.append(
                f"| {scenario} | {method} | "
                f"{metrics['failures_caught']:.2f} | "
                f"{metrics['precision_at_budget']:.1%} | "
                f"{coverage_text} |"
            )

    lines += [
        "",
        "Workload rises by 3000, capped at 10000. Outcomes are redrawn",
        "under the unchanged logistic probability rule. Paired scenarios",
        "use the same outcome random numbers. Cause labels are omitted.",
        "",
        "Metrics are averages, not confidence intervals. Failure prevalence",
        "can change, so compare ranking methods within each scenario.",
        "This is a synthetic covariate-shift stress test, not concept drift,",
        "real-world validation or evidence of monetary savings.",
        "The existing workflow threshold and saved model are unchanged.",
        "",
        f"Experiment code SHA-256: {report['experiment_code_sha256']}",
        "",
        f"Generator SHA-256: {report['generator_sha256']}",
    ]

    destination = ROOT / "evals/results/workload_shift.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = "\n".join(lines) + "\n"
    (ROOT / "docs/workload_shift_results.md").write_text(
        summary, encoding="utf-8"
    )
    print(summary)
    print("Detailed local report:", destination)


if __name__ == "__main__":
    main()
