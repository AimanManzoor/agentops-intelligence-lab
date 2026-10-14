# Model Card — AgentOps Pipeline Failure Prediction

## Overview

AgentOps Intelligence Lab uses machine learning to estimate the probability that a data pipeline will fail before execution.

**Current status:** Research prototype evaluated on synthetic data. Not validated for production operations.

## Dataset

- Source: Reproducible synthetic incident generator
- Total records: 5,000
- Training records: 3,500 (70%)
- Validation records: 750 (15%)
- Final test records: 750 (15%)
- Target: `target_failed`, where 1 indicates failure and 0 indicates success

## Input Features

- `expected_workload`: Expected pipeline workload
- `cpu_usage_pct`: Simulated pre-run CPU utilization
- `memory_usage_pct`: Simulated pre-run memory utilization
- `recent_failure_rate`: Simulated historical failure-rate indicator

Post-execution fields and root-cause labels are excluded from model inputs to reduce target leakage.

## Model Selection

Three models were evaluated:

| Model | Validation Average Precision |
|---|---:|
| Majority baseline | 0.3947 |
| Logistic Regression | 0.6730 |
| Random Forest | 0.6548 |

Logistic Regression was selected using validation average precision.

A decision threshold of 0.26 was selected by maximizing validation F1-score.

## Final Test Results

| Metric | Result |
|---|---:|
| Precision | 0.4688 |
| Recall | 0.8633 |
| F1-score | 0.6076 |
| Average Precision | 0.6688 |
| ROC-AUC | 0.7537 |
| Brier Score | 0.1890 |

These results were measured on a separate 750-record synthetic test partition.

## Interpretation

The model identifies approximately 86% of simulated failures at the selected threshold, but its relatively low precision produces a substantial number of false alarms.

The risk threshold would need further cost-sensitive validation before operational use.

## Known Limitations

- Entirely synthetic training and evaluation data
- Simulated feature distributions may not match production telemetry
- Logistic failure-generation assumptions favor logistic regression
- No demonstrated robustness to changing infrastructure conditions or data drift
- No external real-world validation
- No operational deployment or post-deployment monitoring yet

## Planned Improvements

1. Test failure-rate and workload distribution shifts.
2. Add cost-sensitive threshold evaluation.
3. Integrate OpenTelemetry data.
4. Expose predictions through a validated, read-only inference tool.
5. Add agent orchestration, execution tracing, and automated evaluations.
6. Implement FastAPI, Docker, and CI/CD deployment.

## Reproducibility

Generate the synthetic dataset using:

```bash
python src/ingestion/generate_realistic_incidents.py
```

Run the evaluation harness:

```bash
python evals/evaluate_model.py
```

Evaluation charts are saved to `assets/`, and local model artifacts are stored in `models/`.