# Review queue ranking results

Fixed budget: 50 of 250 runs per batch.
Evaluation: 10 independent generator seeds; 5 random draws per batch.

| Method | Mean reviewed | Mean failures caught | Mean precision at budget | Mean failure coverage |
|---|---:|---:|---:|---:|
| risk_ranking | 50 | 31.00 | 62.0% | 33.9% |
| highest_workload | 50 | 25.10 | 50.2% | 27.4% |
| random_mean | 50 | 19.16 | 38.3% | 20.8% |

Metrics are macro-averages across batches; random metrics first average the fixed draws within each batch. These are not confidence intervals.

## Paired comparisons

- Risk minus highest_workload: mean precision difference 11.80 percentage points; observed batch range -8.00 to 24.00; risk strictly better in 8 batches.
- Risk minus random_mean: mean precision difference 23.68 percentage points; observed batch range 10.80 to 40.00; risk strictly better in 10 batches.

## Limits
All data are synthetic draws from the same logistic generator, which favors logistic regression. New seeds do not establish domain-shift robustness, future-time validity, causality, production gains or real-world performance.
All scores in a batch are assumed available before selection. Score floor 0.0 is used only in this offline equal-budget experiment; the live workflow and its threshold remain unchanged.

Protocol SHA-256: 172e583a5ee6f14b30e3190c441c602c3495ebe7edcf346ba5f5192fcf004477

Generator SHA-256: 60f5b03b19d36c7b2ac2d6434f1faeda74da7bb14b80674645a0f4e13859620c

Versions: {"python": "3.12.7", "scikit_learn": "1.9.1"}

Reproduce: python -m evals.evaluate_review_queue --write-summary

See [frozen protocol and methodology](review_queue_evaluation.md).
