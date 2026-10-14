
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "agentops.db"
RESULT_DIR = ROOT / "evals" / "results"
ASSET_DIR = ROOT / "assets"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "expected_workload",
    "cpu_usage_pct",
    "memory_usage_pct",
    "recent_failure_rate",
]

def load_data():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. Generate incidents first."
        )

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT run_id, started_at,
                   expected_workload, cpu_usage_pct,
                   memory_usage_pct, recent_failure_rate,
                   target_failed
            FROM realistic_incidents
            ORDER BY started_at, run_id
            """,
            conn,
        )

    if len(df) < 100:
        raise ValueError("Insufficient records for evaluation.")

    if df[FEATURES + ["target_failed"]].isna().any().any():
        raise ValueError("Missing feature values or labels.")

    if not set(df["target_failed"].unique()).issubset({0, 1}):
        raise ValueError("Unexpected target values.")

    return df

def metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "precision": float(precision_score(
            y_true, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, predictions, zero_division=0
        )),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]
    test = df.iloc[validation_end:]

    X_train, y_train = train[FEATURES], train["target_failed"]
    X_val, y_val = validation[FEATURES], validation["target_failed"]
    X_test, y_test = test[FEATURES], test["target_failed"]

    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Test rows: {len(test)}")

    models = {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    validation_results = {}
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        val_prob = model.predict_proba(X_val)[:, 1]
        validation_results[name] = metrics(
            y_val, val_prob, 0.5
        )

        print(
            f"{name}: validation AP = "
            f"{validation_results[name]['average_precision']:.4f}"
        )

    # Select between learned models using validation data.
    winner = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: validation_results[name]["average_precision"]
    )

    selected = fitted[winner]
    val_prob = selected.predict_proba(X_val)[:, 1]

    # Choose threshold using only validation labels.
    candidate_thresholds = np.arange(0.10, 0.91, 0.01)
    threshold_scores = [
        f1_score(
            y_val,
            (val_prob >= threshold).astype(int),
            zero_division=0,
        )
        for threshold in candidate_thresholds
    ]

    best_threshold = float(
        candidate_thresholds[int(np.argmax(threshold_scores))]
    )

    # Freeze model and threshold before inspecting test labels.
    test_prob = selected.predict_proba(X_test)[:, 1]
    final_metrics = metrics(
        y_test, test_prob, best_threshold
    )

    report = {
        "dataset": "Synthetic realistic_incidents",
        "dataset_size": n,
        "split": {
            "training": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "features": FEATURES,
        "selected_model": winner,
        "selection_metric": "validation_average_precision",
        "threshold_selection": "validation_f1",
        "decision_threshold": best_threshold,
        "validation_results": validation_results,
        "final_test_results": final_metrics,
    }

    report_path = RESULT_DIR / "final_evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    # Save separately from our previous baseline.
    artifact = {
        "model": selected,
        "features": FEATURES,
        "threshold": best_threshold,
        "version": "v2-validation-selected",
    }
    joblib.dump(
        artifact,
        MODEL_DIR / "validated_failure_model.joblib"
    )

    # Final test confusion matrix.
    cm = np.array(final_metrics["confusion_matrix"])
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Success", "Failed"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Final Test Confusion Matrix — {winner}")
    plt.tight_layout()
    plt.savefig(
        ASSET_DIR / "final_confusion_matrix.png", dpi=160
    )
    plt.close()

    # Precision-recall curve from final test probabilities.
    precision, recall, _ = precision_recall_curve(
        y_test, test_prob
    )
    plt.figure()
    plt.plot(recall, precision, label=winner)
    plt.axhline(
        y=float(y_test.mean()),
        linestyle="--",
        label="Failure prevalence",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final Test Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()

    # Probability calibration curve.
    observed, predicted = calibration_curve(
        y_test, test_prob, n_bins=10, strategy="quantile"
    )
    plt.figure()
    plt.plot(predicted, observed, marker="o", label=winner)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.xlabel("Predicted failure probability")
    plt.ylabel("Observed failure rate")
    plt.title("Final Test Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "calibration_curve.png", dpi=160)
    plt.close()

    print("\nFINAL EVALUATION")
    print(f"Selected model: {winner}")
    print(f"Decision threshold: {best_threshold:.2f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1: {final_metrics['f1']:.4f}")
    print(f"Average precision: {final_metrics['average_precision']:.4f}")
    print(f"ROC-AUC: {final_metrics['roc_auc']:.4f}")
    print(f"Brier score: {final_metrics['brier_score']:.4f}")
    print("\nEvaluation artifacts saved successfully.")

if __name__ == "__main__":
    main()
