# Local reproduction check

## Environment

- Platform: Darwin arm64
- Python: 3.12.7
- scikit-learn: 1.9.1
- pandas: 3.0.6
- LangGraph: 1.2.15

## Procedure completed

1. Cloned the repository into a separate directory.
2. Created a virtual environment and installed requirements.
3. Ran dependency compatibility checks.
4. Generated the synthetic incident dataset.
5. Trained and evaluated the candidate models.
6. Ran the traced high-risk and low-risk workflow examples.
7. Ran the regression tests and versioned workflow scenarios.

## Recorded model evaluation

- Dataset size: 5000
- Training records: 3500
- Validation records: 750
- Test records: 750
- Selected model: logistic_regression
- Model selection metric: validation_average_precision
- Threshold selection: validation_f1
- Decision threshold: 0.26

| Test metric | Result |
|---|---:|
| Precision | 0.4688 |
| Recall | 0.8633 |
| F1 | 0.6076 |
| Average precision | 0.6688 |
| ROC-AUC | 0.7537 |
| Brier score | 0.1890 |

## Observed workflow behavior

The high-risk example completed investigation and incident retrieval before
risk review. The low-risk example followed normal processing and skipped
both evidence tools. Each invocation saved a local JSON trace.

## Interpretation and limitations

Model selection and threshold tuning use validation data. The final test
set evaluates the frozen selection.

The model detects many synthetic failures but also produces false alerts.
These results do not establish performance on real operational incidents.
The synthetic outcome generator uses a logistic probability function,
which favors Logistic Regression in this experiment.

Workflow scenario checks evaluate software contracts separately from
predictive performance. Local traces are diagnostic records, not a
production latency benchmark.
