
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "agentops.db"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "evals" / "results"
ASSET_DIR = ROOT / "assets"

# --------------------------------------------------
# 2. Prediction features
# --------------------------------------------------

FEATURES = [
    "expected_workload",
    "cpu_usage_pct",
    "memory_usage_pct",
    "recent_failure_rate",
]

# --------------------------------------------------
# 3. Load data from SQLite
# --------------------------------------------------

def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                run_id,
                started_at,
                expected_workload,
                cpu_usage_pct,
                memory_usage_pct,
                recent_failure_rate,
                target_failed
            FROM realistic_incidents
            ORDER BY started_at, run_id
            """,
            conn,
        )

    if df.empty:
        raise ValueError("No training data found.")

    if df[FEATURES + ["target_failed"]].isnull().any().any():
        raise ValueError("Dataset contains missing model values.")

    if df["target_failed"].nunique() != 2:
        raise ValueError("Both outcome classes are required.")

    return df

# --------------------------------------------------
# 4. Train and evaluate models
# --------------------------------------------------

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print(f"Total records: {len(df)}")
    print(f"Total failures: {df['target_failed'].sum()}")

    # Chronological split
    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[FEATURES]
    y_train = train["target_failed"].astype(int)

    X_test = test[FEATURES]
    y_test = test["target_failed"].astype(int)

    print(f"Training records: {len(train)}")
    print(f"Testing records: {len(test)}")

    # Three models for comparison
    models = {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    trained = {}

    for name, model in models.items():

        print(f"\nTraining model: {name}")

        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(
            y_test,
            predicted,
            labels=[0, 1]
        )

        results[name] = {
            "precision": float(
                precision_score(
                    y_test,
                    predicted,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    y_test,
                    predicted,
                    zero_division=0
                )
            ),

            "f1": float(
                f1_score(
                    y_test,
                    predicted,
                    zero_division=0
                )
            ),

            "pr_auc": float(
                average_precision_score(
                    y_test,
                    probabilities
                )
            ),

            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    probabilities
                )
            ),

            "brier_score": float(
                brier_score_loss(
                    y_test,
                    probabilities
                )
            ),

            "confusion_matrix": cm.tolist(),
        }

        trained[name] = model

        print(
            classification_report(
                y_test,
                predicted,
                zero_division=0
            )
        )

    # --------------------------------------------------
    # 5. Select model
    # --------------------------------------------------

    # Initial experimental selection using holdout PR-AUC.
    # Before production, use separate validation and test sets.

    winner = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: results[name]["pr_auc"]
    )

    print(f"\nSelected model: {winner}")

    # --------------------------------------------------
    # 6. Save trained model
    # --------------------------------------------------

    model_path = MODEL_DIR / "failure_model.joblib"

    joblib.dump(
        trained[winner],
        model_path
    )

    print(f"Model saved to: {model_path}")

    # --------------------------------------------------
    # 7. Save evaluation metrics
    # --------------------------------------------------

    report = {
        "dataset_size": int(len(df)),
        "training_records": int(len(train)),
        "test_records": int(len(test)),
        "features": FEATURES,
        "selection_metric": "held_out_pr_auc",
        "selected_model": winner,
        "results": results,
    }

    metrics_path = RESULT_DIR / "model_metrics.json"

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"Evaluation metrics saved to: {metrics_path}")

    # --------------------------------------------------
    # 8. Generate confusion matrix
    # --------------------------------------------------

    # FIX:
    # ConfusionMatrixDisplay expects a NumPy array,
    # not a Python list.

    selected_cm = np.array(
        results[winner]["confusion_matrix"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=selected_cm,
        display_labels=["Success", "Failed"]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"Confusion Matrix - {winner}")
    plt.tight_layout()

    chart_path = ASSET_DIR / "confusion_matrix.png"

    plt.savefig(
        chart_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Confusion matrix saved to: {chart_path}")

    # --------------------------------------------------
    # 9. Display model comparison
    # --------------------------------------------------

    comparison = pd.DataFrame({
        name: {
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "PR_AUC": metrics["pr_auc"],
            "ROC_AUC": metrics["roc_auc"],
            "Brier": metrics["brier_score"]
        }
        for name, metrics in results.items()
    }).T

    print("\nMODEL COMPARISON")
    print(comparison.round(4).to_string())

    print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()
