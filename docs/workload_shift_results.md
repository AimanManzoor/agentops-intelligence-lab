# Workload-shift experiment

One experimental model; 50 reviews per 250-run batch.
Ten paired synthetic seeds; five random selections per batch.

| Scenario | Method | Mean failures caught | Precision at budget | Failure coverage |
|---|---|---:|---:|---:|
| baseline | risk_ranking | 33.30 | 66.6% | 34.9% |
| baseline | highest_workload | 25.10 | 50.2% | 26.2% |
| baseline | random_mean | 19.04 | 38.1% | 20.0% |
| higher_workload | risk_ranking | 37.80 | 75.6% | 32.9% |
| higher_workload | highest_workload | 27.50 | 55.0% | 23.9% |
| higher_workload | random_mean | 23.14 | 46.3% | 20.1% |

Workload rises by 3000, capped at 10000. Outcomes are redrawn
under the unchanged logistic probability rule. Paired scenarios
use the same outcome random numbers. Cause labels are omitted.

Metrics are averages, not confidence intervals. Failure prevalence
can change, so compare ranking methods within each scenario.
This is a synthetic covariate-shift stress test, not concept drift,
real-world validation or evidence of monetary savings.
The existing workflow threshold and saved model are unchanged.

Experiment code SHA-256: c30e3baf179adf5e4c5031d5856effff1b5c444d8a308e84a2767c5b75f9eef3

Generator SHA-256: 60f5b03b19d36c7b2ac2d6434f1faeda74da7bb14b80674645a0f4e13859620c
