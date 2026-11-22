"""Offline, retrospective alert-policy evaluation; never deploys a threshold."""
import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["expected_workload", "cpu_usage_pct", "memory_usage_pct", "recent_failure_rate"]


def measure(labels, scores, threshold, false_alert_cost, missed_failure_cost):
    labels, scores = list(labels), list(scores)
    if not labels or len(labels) != len(scores):
        raise ValueError("Nonempty, equally sized labels and scores required.")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary.")
    for value in [*scores, threshold]:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Scores and threshold must be finite and in [0, 1].")
    if any(not math.isfinite(c) or c <= 0 for c in (false_alert_cost, missed_failure_cost)):
        raise ValueError("Costs must be finite and positive.")
    predictions = [p >= threshold for p in scores]
    tp = sum(y == 1 and a for y, a in zip(labels, predictions))
    fp = sum(y == 0 and a for y, a in zip(labels, predictions))
    fn = sum(y == 1 and not a for y, a in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "alerts": tp + fp, "alerts_per_100": 100 * (tp + fp) / len(labels),
        "cost_units": fp * false_alert_cost + fn * missed_failure_cost,
    }


def choose(labels, scores, policy):
    labels, scores = list(labels), list(scores)
    cap = policy.get("max_alert_fraction", 1.0)
    if not math.isfinite(cap) or not 0 <= cap <= 1:
        raise ValueError("Alert fraction must be in [0, 1].")
    curve = [measure(labels, scores, i / 100,
                     policy["false_alert_cost"], policy["missed_failure_cost"])
             for i in range(101)]
    feasible = [row for row in curve if row["alerts"] <= cap * len(labels)]
    if not feasible:
        raise ValueError("No grid threshold meets the validation alert limit.")
    # Freeze tie-breaking: lowest cost, then fewer alerts, then higher threshold.
    winner = min(feasible, key=lambda row: (
        row["cost_units"], row["alerts"], -row["threshold"]))
    return winner, curve


def compare(validation_labels, validation_scores, test_labels, test_scores, policy):
    chosen, curve = choose(validation_labels, validation_scores, policy)
    evaluated = measure(test_labels, test_scores, chosen["threshold"],
                        policy["false_alert_cost"], policy["missed_failure_cost"])
    baseline = measure(test_labels, test_scores, 0.26,
                       policy["false_alert_cost"], policy["missed_failure_cost"])
    return {
        "assumptions": policy, "selected_on": "validation_only",
        "validation": chosen, "validation_curve": curve,
        "retrospective_test": evaluated, "baseline_0_26_test": baseline,
        "always_alert_test": measure(test_labels, test_scores, 0.0,
                                    policy["false_alert_cost"], policy["missed_failure_cost"]),
        "never_alert_test": dict(measure(test_labels, [0.0] * len(test_scores), 1.0,
                                        policy["false_alert_cost"], policy["missed_failure_cost"]),
                                threshold=None, decision="never_alert"),
        "test_alert_limit_met": evaluated["alerts_per_100"] <=
                               100 * policy.get("max_alert_fraction", 1.0),
        "test_cost_difference_vs_baseline": evaluated["cost_units"] - baseline["cost_units"],
    }


def main():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data/agentops.db")
    args = parser.parse_args()
    # SQLite mode=ro refuses missing files and cannot create or write a database.
    with sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        frame = pd.read_sql_query(
            "SELECT run_id, started_at, expected_workload, cpu_usage_pct, "
            "memory_usage_pct, recent_failure_rate, target_failed "
            "FROM realistic_incidents ORDER BY started_at, run_id", conn)
    if len(frame) < 100 or frame.isna().any().any():
        raise ValueError("At least 100 complete records required.")
    if not frame.run_id.is_unique or not set(frame.target_failed).issubset({0, 1}):
        raise ValueError("Unique run IDs and binary labels required.")
    train_end, val_end = int(len(frame) * .70), int(len(frame) * .85)
    train, validation, test = frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:]
    if train.target_failed.nunique() != 2:
        raise ValueError("Training requires both outcome classes.")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[FEATURES], train.target_failed)
    failure_column = list(model.classes_).index(1)
    val_scores = model.predict_proba(validation[FEATURES])[:, failure_column].tolist()
    test_scores = model.predict_proba(test[FEATURES])[:, failure_column].tolist()
    config_path = ROOT / "evals/alert_policies_v1.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config.get("schema_version") != 1 or not config.get("policies"):
        raise ValueError("Nonempty version 1 policies required.")
    report = {
        "evaluation_status": "retrospective_synthetic_experiment",
        "model": "fixed_logistic_regression_retrained_on_training_only",
        "deployed": False, "baseline_threshold": 0.26,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(frame.to_json(orient="records").encode()).hexdigest(),
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "results": {
            name: compare(validation.target_failed, val_scores, test.target_failed, test_scores, policy)
            for name, policy in config["policies"].items()
        },
    }
    destination = ROOT / "evals/results/alert_policy_evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Retrospective synthetic experiment. Costs are assumptions; no deployment.")
    print("Policy                  Threshold  Precision  Recall  Alerts/100  Cost  Baseline cost")
    for name, result in report["results"].items():
        row = result["retrospective_test"]
        print(f'{name:24} {row["threshold"]:9.2f} {row["precision"]:10.3f} '
              f'{row["recall"]:7.3f} {row["alerts_per_100"]:11.1f} '
              f'{row["cost_units"]:5.0f} {result["baseline_0_26_test"]["cost_units"]:14.0f}')
        print(f'  Constant-policy costs: always-alert={result["always_alert_test"]["cost_units"]}; '
              f'never-alert={result["never_alert_test"]["cost_units"]}')
        if not result["test_alert_limit_met"]:
            print("  Validation capacity limit was exceeded on test; it is not a guarantee.")
    print("Report:", destination)


if __name__ == "__main__":
    main()
