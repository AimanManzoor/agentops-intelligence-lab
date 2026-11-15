
import json
import math
from numbers import Integral, Real
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

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

        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
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

    # Reject invalid requests before looking up or deserializing an artifact.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load only our locally generated, trusted model file.
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or not {
        "model", "features", "threshold", "version"
    }.issubset(artifact):
        raise ValueError("Model artifact is missing required metadata")
    model, features = artifact["model"], artifact["features"]
    if (not isinstance(features, list) or
            not all(isinstance(name, str) for name in features) or
            features != REQUIRED_FEATURES):
        raise ValueError("Unexpected model feature schema")
    raw_threshold = artifact["threshold"]
    if (isinstance(raw_threshold, (bool, np.bool_)) or
            not isinstance(raw_threshold, Real) or
            not 0 <= raw_threshold <= 1 or not math.isfinite(raw_threshold)):
        raise ValueError("Decision threshold must be a finite number between 0 and 1")
    threshold = float(raw_threshold)
    if not isinstance(artifact["version"], str) or not artifact["version"].strip():
        raise ValueError("Model version must be a non-empty string")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError("Model must provide predict_proba")
    classes = np.asarray(getattr(model, "classes_", []))
    if (classes.shape != (2,) or
            any(isinstance(label, (bool, np.bool_)) or not isinstance(label, Integral)
                for label in classes) or set(classes.tolist()) != {0, 1}):
        raise ValueError("Model must expose binary integer classes 0 and 1")

    input_df = pd.DataFrame([values], columns=features)

    probabilities = np.asarray(model.predict_proba(input_df))
    if (probabilities.shape != (1, 2) or probabilities.dtype.kind not in "iuf" or
            not np.isfinite(probabilities).all() or
            not ((probabilities >= 0) & (probabilities <= 1)).all() or
            not math.isclose(float(probabilities[0].sum()), 1.0,
                             rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError("predict_proba must return one normalized finite binary probability row")
    probability = float(probabilities[0, classes.tolist().index(1)])

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
