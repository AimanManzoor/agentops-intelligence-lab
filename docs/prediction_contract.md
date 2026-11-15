# Prediction tool contract

## Purpose

`src/tools/prediction_tool.py` wraps the project's locally generated binary
classifier. It validates a request and a trusted artifact, calls inference once,
and returns structured failure risk for deterministic LangGraph routing.

## Input validation precedes artifact access

| Feature | Accepted input |
|---|---|
| `expected_workload` | Non-negative built-in Python integer, representable as a finite numeric input |
| `cpu_usage_pct` | Finite built-in Python int/float from 0 to 100 inclusive |
| `memory_usage_pct` | Finite built-in Python int/float from 0 to 100 inclusive |
| `recent_failure_rate` | Finite built-in Python int/float from 0 to 1 inclusive |

Booleans, strings, missing values, NaN and infinity are rejected. Invalid requests
raise `ValueError` before artifact lookup or loading. A valid request with a
missing/non-file artifact path raises `FileNotFoundError` without creating it.
The public function signature still requires all four named arguments.

## Artifact and output checks

Only locally generated, trusted `.joblib` artifacts may be loaded. Joblib
deserialization can execute code before these checks run: metadata validation
does not make an untrusted file safe. Corrupt-file deserialization errors
propagate; the tool does not repair or overwrite an artifact.

The loaded artifact must contain a model, the exact ordered feature list, a
numeric finite decision threshold in [0,1], and a non-empty version string.
The model must provide `predict_proba` and binary integer class labels 0 and 1.
The failure probability is selected by label 1, not an assumed column position.

The inference result must be a real numeric array of shape (1,2), with finite
values in [0,1] whose row sums to 1 within relative/absolute tolerance 1e-6.
Invalid probabilities raise `ValueError`; they must not silently become a
lower-risk route. There is no inference retry or fallback prediction.

The decision uses the unrounded probability with `>= threshold`; the displayed
probability is rounded to four decimals. A rounded display value can therefore
equal the threshold while the unrounded value is slightly below it. Do not
reconstruct the routing decision from the rounded display.

## Graph behavior and verification

Prediction errors propagate out of the graph. SQL investigation and similar
incident retrieval do not run after invalid prediction output. Existing tracing
can record the failed prediction node; this milestone adds no repair behavior.

Twelve new test methods cover inputs, boundaries, artifact metadata, threshold
validity, missing/corrupt files, class ordering, output shape/range, threshold
rounding, graph failure behavior, and a small locally trained artifact roundtrip.
They require neither the project's full dataset nor a saved production-sized
model. The expanded suite has 37 tests. The new cases were also run against the
previous implementation and exposed failures; the fixed implementation passes.

These are software-contract checks. They do not measure predictive quality,
calibration on real pipelines, model provenance, artifact integrity, or latency.
