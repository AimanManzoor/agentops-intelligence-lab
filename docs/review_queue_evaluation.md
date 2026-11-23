# Review queue evaluation: frozen protocol v1

## Research question
At the same investigation budget, does ranking by learned failure probability
select more failed runs than highest-workload or seeded-random selection?

## Fixed design
The versioned configuration is evals/review_queue_protocol_v1.json.
Before the first full experiment run, freeze:
- Training: 3500 generated rows, seed 42, four pre-run features.
- Model: StandardScaler plus LogisticRegression(max_iter=1000, random_state=42).
- Evaluation: seeds 1001 through 1010; 250 rows per batch.
- Capacity: exactly 50 reviews (20%) per method in each batch.
- Random comparator: seeds 11, 22, 33, 44 and 55 per batch, each selecting 50.

The existing generator was extracted into a pure function; the original database
writer still generates exactly the same 5000 rows and 1886 failures at seed 42.
This experiment creates data in memory and trains a separate in-memory model.
It never reads or modifies the local project database or model artifact.

The same generator's logistic relationship favors logistic regression. New seeds
are independent simulated draws within that relationship, not new domains,
future calendar periods, independent real-world evidence or externally preregistered
research. The protocol was fixed before this experiment's full evaluation;
changing it after seeing results makes later comparisons exploratory.

## Selection and outcome separation
Model features are workload, CPU utilization, memory utilization and simulated
recent-failure rate. The generator's true failure probability, realized failure,
root-cause label and timestamps are excluded from model inputs.

For risk ranking, the existing queue receives only run ID and model-predicted
failure probability. Set its score floor to 0.0 only for this offline experiment
so every policy uses the full budget. This does not change the live 0.26 threshold.
Risk/workload ties resolve by ascending run ID. Random selection samples from
sorted IDs using a reproducible seed combining batch seed and random-draw seed.
Outcome labels are joined only after selection for evaluation.

The generator's recent-failure feature remains simulated, not a real historical
window. All batch scores are assumed available before selection; this is not a
streaming arrival-time policy.

## Metrics
- Precision at budget: failed selected runs / 50 reviews.
- Failure coverage: failed selected runs / all failed runs in the batch.
- Failures caught and missed: counts within that batch.
- Reviewed count: must equal 50 for all methods.

Average the five random draws within a batch first, then macro-average all
methods across batches. Coverage is undefined for a batch without failures;
undefined values are excluded from its mean and remain null if all are undefined.
Report paired risk-minus-baseline precision differences and their observed range.
Ranges describe these seeds; they are not confidence intervals or hypothesis tests.
No estimated monetary savings or causal effect is computed.

## Reproduce and publish
Run:

    python -m evals.evaluate_review_queue --write-summary

The complete JSON report stays in ignored evals/results/review_queue_evaluation.json.
The command writes a shareable aggregate table to docs/review_queue_results.md.
Commit that results document with the code, protocol and README.
The report includes protocol/generator hashes, library versions, per-batch counts,
selection IDs and random-draw metrics to support local inspection.

## Verification
Eleven new tests cover generation reproducibility and bounds, original writer
compatibility, protocol rejection, exclusion of oracle fields, baseline selection,
metric calculations, outcome-flip invariance, equal budgets, averaging and CLI errors.
Flipping evaluation outcomes and stored oracle probabilities must not change
selected IDs. Existing suite total: 72 Python tests; eight workflow scenarios
and nine dashboard tests remain separate checks.

No result assertion requires the learned model to win. A loss is a research
finding, not an excuse to change reference answers or tune on evaluation labels.

## Next boundary
Confirm on a new, prospectively specified data-generating process with drift,
severity and review effort; then assess whether ranking helps under those conditions.
This milestone does not deploy a service, change routing or establish production
reliability, security, causal benefit or real-world predictive performance.
