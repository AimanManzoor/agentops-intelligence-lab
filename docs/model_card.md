# Model card: synthetic pipeline failure baseline

## Status and intended use

Portfolio/research prototype for estimating simulated pre-run pipeline failure
risk and testing downstream evidence retrieval and routing. It is not validated
for production decisions, autonomous remediation, or causal incident diagnosis.

The README records a previously evaluated Logistic Regression baseline,
`v2-validation-selected`, with decision threshold 0.26. This card documents that
reported experiment. The prediction-contract milestone does not retrain the
project model or rerun the full-dataset evaluation.

## Data and inputs

- Seeded generator: `src/ingestion/generate_realistic_incidents.py`.
- Previously observed dataset: 5,000 rows; 1,886 simulated failures (37.72%).
- Inputs: expected workload, CPU usage, memory usage, and recent failure rate.
- Target: binary `target_failed`. It and the assigned `root_cause` label are
  excluded from prediction inputs.
- The recent-failure signal is simulated, not derived from actual prior runs.
  Assigned cause labels are not independently verified causal ground truth.

## Reported evaluation

Chronological 70/15/15 split: 3,500 training, 750 validation, 750 final-test rows.
The evaluation compares a majority baseline, Logistic Regression and Random
Forest. Validation average precision chooses the learned model; validation F1
chooses the threshold. The final test evaluates the frozen selection.

| Previously reported final-test metric | Value |
|---|---:|
| Precision | 0.4688 |
| Recall | 0.8633 |
| F1 | 0.6076 |
| Average precision | 0.6688 |
| ROC-AUC | 0.7537 |
| Brier score | 0.1890 |

These are synthetic results reported in the README, not new measurements from
this reliability milestone. The generator uses a logistic probability function,
which favors Logistic Regression's inductive bias. High recall comes with false
alerts; operational alert costs have not been evaluated.

## Inference and limitations

See [prediction contract](prediction_contract.md) for request, artifact,
probability and routing checks. Only trusted locally generated artifacts should
be loaded. Schema checks do not establish artifact security or authenticity.

No generalization, fairness, drift, or calibration claims on real pipeline data
are supported. The workflow's similar incident IDs supply historical context,
not proof of a new run's cause. Live data validation, realistic temporal
features, alert-cost analysis, and deployment monitoring remain future work.
