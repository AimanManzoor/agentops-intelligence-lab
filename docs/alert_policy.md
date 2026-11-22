# Cost-aware alert policy experiment

## Question
How should a fixed failure model trade missed incidents against unnecessary alerts?

## Protocol
Train a fixed standardized logistic regression on the first 70% of chronologically
ordered synthetic incidents. Select thresholds on the next 15%; evaluate the frozen
choices on the last 15%. No model binary is loaded or written, and the deployed
workflow threshold is unchanged. SQLite is opened read-only.

The earlier test partition has already been inspected. Results are retrospective
diagnostics, not a new independent benchmark. Future confirmation needs an independent
incident dataset and a frozen protocol.

Versioned assumptions live in evals/alert_policies_v1.json:
- limited_capacity: false alert = 1 unit, missed failure = 2; validation alerts <=20%.
- high_sensitivity: false alert = 1 unit, missed failure = 10; no alert cap.

These are illustrative relative penalties, not measured costs or dollars. The
objective is FP * false_alert_cost + FN * missed_failure_cost. It excludes review
cost for true alerts, incident severity, duplicate alerts and avoided-loss effects.

Search thresholds 0.00 through 1.00, inclusive, at 0.01 intervals. Predict failure
when score >= threshold. Minimize validation cost subject to its alert cap; break
ties by fewer alerts then higher threshold. An infeasible policy raises an error.
The validation capacity limit is not guaranteed on later data; report violations.

## Run and inspect
Run python -m evals.evaluate_alert_policy after generating the synthetic database.
The ignored local report evals/results/alert_policy_evaluation.json contains the
validation curves, policy/config and dataset hashes, split sizes and test metrics.
Compare each policy with the fixed existing 0.26 threshold, always-alert and
never-alert decisions under the same costs. Constant-policy comparators are
reported without threshold selection. The never-alert comparator is an explicit
constant decision, not a threshold available to the deployment workflow.
A negative cost difference favors the policy on these records only.

Report precision, recall, FP, FN, alerts per 100 and cost together. Always-alert and
never-alert decisions may be cost-optimal under strong assumptions; inspect the
curve rather than choosing costs to make a model look better.

## Learning and next experiment
Change cost assumptions before inspecting a future holdout. Explain why each cost
ratio and capacity limit fits a hypothetical operator. Validate on independent
scenarios, add severity and investigation time, and compare with simple policies.
No confidence intervals, drift robustness, actual alert reductions, causal savings
or production readiness are established by this experiment.

Reference: https://scikit-learn.org/stable/modules/classification_threshold.html
