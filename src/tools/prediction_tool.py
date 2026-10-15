
import json
import math
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "validated_failure_model.joblib"

REQUIRED_FEATURES = [
    "expected_workload",
    "cpu_usage_pct",
    "memory_usage_pct",
    "recent_failure_rate",
]


def predict_failure(
    expected_workload: int,
    cpu_usage_pct: float,
    memory_usage_pct: float,
    recent_failure_rate: float,
) -> dict:
    """
    Predict pipeline failure risk using our validated ML model.

    Inputs represent conditions available before execution.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    values = {
        "expected_workload": expected_workload,
        "cpu_usage_pct": cpu_usage_pct,
        "memory_usage_pct": memory_usage_pct,
        "recent_failure_rate": recent_failure_rate,
    }

    # Validate numeric types and ranges.
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise ValueError(f"{name} must be numeric")

        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if (
        not isinstance(expected_workload, int)
        or expected_workload < 0
    ):
        raise ValueError(
            "expected_workload must be a non-negative integer"
        )

    for feature in ["cpu_usage_pct", "memory_usage_pct"]:
        if not 0 <= values[feature] <= 100:
            raise ValueError(
                f"{feature} must be between 0 and 100"
            )

    if not 0 <= recent_failure_rate <= 1:
        raise ValueError(
            "recent_failure_rate must be between 0 and 1"
        )

    # Load only our locally generated, trusted model file.
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]
    threshold = float(artifact["threshold"])

    if features != REQUIRED_FEATURES:
        raise ValueError("Unexpected model feature schema")

    input_df = pd.DataFrame([values], columns=features)

    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    predicted_failure = probability >= threshold

    return {
        "failure_probability": round(probability, 4),
        "predicted_failure": bool(predicted_failure),
        "decision_threshold": threshold,
        "risk_level": (
            "Elevated risk" if predicted_failure
            else "Lower predicted risk"
        ),
        "model_version": artifact["version"],
        "input_features": values,
    }


if __name__ == "__main__":
    result = predict_failure(
        expected_workload=9000,
        cpu_usage_pct=92.0,
        memory_usage_pct=88.0,
        recent_failure_rate=0.20,
    )

    print(json.dumps(result, indent=2))
