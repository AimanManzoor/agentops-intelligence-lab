# Evaluation methodology

## Separate predictive quality from workflow correctness

The ML experiment asks whether pre-run features predict simulated failures.
Regression tests ask whether software follows its contracts: input validation,
routing, evidence cutoffs, source failures, prediction outputs, and tracing.
A passing regression suite is not evidence of predictive accuracy or grounded
LLM reasoning. No LLM reasoning is implemented in the current workflow.

## Predictive experiment

Implementation: `evals/evaluate_model.py`.

1. Load and validate the synthetic incident records.
2. Sort by `started_at`, then split into training (70%), validation (15%), and
   final test (15%). The current experiment does not implement grouped temporal
   cross-validation or a general tied-timestamp grouping guarantee.
3. Fit each candidate using training data only. Logistic Regression uses a
   fitted training-only scaler inside its sklearn pipeline.
4. Compare learned candidates using validation average precision. Keep the
   majority classifier as a reference baseline.
5. Select a threshold from 0.10 to 0.90 in 0.01 steps using validation F1.
6. Freeze model and threshold before computing final-test metrics and plots.
7. Save the report and trusted model artifact locally, excluded from Git.

Reported metrics include precision, recall, F1, average precision, ROC-AUC,
Brier score, confusion matrix, and a calibration curve. F1 is an experimental
threshold criterion, not an estimate of operational cost. See the
[model card](model_card.md) for previously reported values and assumptions.

The synthetic generator defines the label relationship; this is not independent
real-world validation. A chronological split alone does not establish realistic
temporal drift, historical feature construction, or causal generalization.

## Software checks

Run `python -m unittest discover -s tests -v`. The 37-test suite uses temporary
databases, controlled predictions, malformed-artifact/output cases, and one
small locally trained sklearn artifact. No provider API keys or project-sized
saved model are required. The inference contract tests do not replace model
quality evaluation.

Missing evidence is tested separately from no matching history. Prediction
failures stop execution; handled evidence-source failures preserve whatever
evidence remains and report incomplete investigation. Local traces record node
outcomes, not causal correctness or production latency.

## Remaining evaluation work

- Validate real pre-run features and derive historical windows strictly from
  information available at the decision time.
- Add tied-time grouping and rolling evaluation if adopting real temporal data.
- Measure threshold sensitivity, alert volume and false-alert/missed-failure costs.
- Use an independently specified incident-task set for future model-assisted
  investigation; measure source correctness and unsupported claims separately.
- Report failures and repeated-run variability before claiming agent improvement.

The existing full-data evaluation was not rerun for the prediction-contract
milestone, and its reported metrics have not been updated.
