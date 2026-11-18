# Versioned workflow evaluation

## Question and scope

Does the deterministic workflow follow its declared routing, evidence and tracing
contracts when dependencies work, return no history, or fail? The unit of evaluation
is one traced workflow invocation. Eight known reference cases are regression
coverage, not eight independent samples from a real deployment.

Prediction is deliberately controlled (0.90 high risk, 0.01 low risk, or an injected
ValueError). This isolates orchestration behavior from predictive quality. The
actual graph, SQL investigation, incident ranking and trace writer execute.
Adapters point the two evidence tools at a temporary database; the partial-source
case injects a retrieval OperationalError. No model artifact or API is loaded.

## Reference design

The expectations in `evals/workflow_scenarios_v1.json` are specified separately
from graph execution. They are not derived from the returned result. Each case
starts with a new database containing five synthetic records:

| Record | Pipeline | Time | Label | Purpose |
|---|---|---|---|---|
| 1 | payment_etl | January 1 | Failed | Only eligible retrieved failure |
| 2 | payment_etl | January 2 | Successful | Prior-run denominator |
| 3 | payment_etl | January 3, exact cutoff | Failed | Boundary exclusion |
| 4 | payment_etl | January 4 | Failed | Future exclusion |
| 5 | another_pipeline | January 1 | Failed | Pipeline isolation |

All dates are in 2026; cutoff is 2026-01-03T00:00:00+00:00. Feature distances are
identical, so eligibility alone determines the one expected source ID. This suite
does not stress ranking quality; separate retrieval tests cover ranking behavior.

| Scenario | Expected result | Prediction / SQL / retrieval calls |
|---|---|---|
| Healthy high risk | risk_review; complete; 2 runs, 1 failure, source 1 | 1 / 1 / 1 |
| Low risk | normal_processing; not_requested | 1 / 0 / 0 |
| Unknown pipeline | complete; zero history and zero matches | 1 / 1 / 1 |
| Missing database | unavailable; no incident references | 1 / 1 / 0 |
| Missing table | unavailable; structured OperationalError | 1 / 1 / 0 |
| Retrieval error | partial; preserve SQL summary | 1 / 1 / 1 |
| Prediction error | failed at prediction; no evidence calls | 1 / 0 / 0 |
| Invalid request | failed at validate_input | 0 / 0 / 0 |

Each case also checks exact node order/status, shared cutoff arguments, trace
execution outcome, selected output facts and exclusions, and byte-for-byte database
preservation (or continued absence). The trace check looks for known forbidden
payload keys, the fixture path, and an injected exception-message sentinel. It is
a focused regression check, not a comprehensive privacy/security audit.

A case passes only if every check passes. The report's SHA-256 identifies reference
bytes; it is not an authenticity signature. Inspect reference changes in code review
rather than adjusting expectations solely to make a failing run green.

## Reproduce and interpret

From the repository root, using an environment with requirements installed:

```bash
python -m unittest discover -s tests -v
python -m evals.evaluate_workflow
```

The second command prints JSON: suite ID, reference hash, controlled prediction
mode, zero model calls, case totals, per-case observations and failed check names.
Temporary databases and trace files are removed after each case. To save a local
report, redirect stdout into the ignored runs directory after creating it.

Exit 0 means every case passed; 1 means at least one contract mismatch; 2 means
reference loading/validation failed. Unexpected evaluator programming errors also
exit nonzero. The GitHub workflow runs both commands after dependency checks.

Seven new evaluator tests bring the repository to 44 tests. Negative controls
change expected counts, break routing, move the SQL cutoff forward, and inject a
trace payload. They verify that the evaluator can detect these regressions. CLI
tests verify success, mismatch, and invalid-reference exit codes.

## What the result establishes

The local implementation satisfies these eight specified contracts. It does not
establish ML precision/recall, causal diagnosis, semantic answer groundedness,
production availability, latency under load, or LLM tool-selection quality. These
known cases have no held-out statistical interpretation. String checks are suitable
for today's fixed messages but may reject valid wording changes in a later generator.

Next, verify the full documented generate/train/infer/trace path in a clean clone.
Before introducing an LLM policy, define additional independent tasks and permitted
actions, and compare its outcomes with this deterministic baseline. Any stochastic
evaluation will need repeated trials and uncertainty reporting.
