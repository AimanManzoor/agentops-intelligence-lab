# AgentOps Intelligence Lab

**A reproducible, evaluation-first prototype for predicting data-pipeline failures and building toward evidence-grounded AI incident investigation.**

**Current status:** Synthetic data pipeline, baseline predictive ML, validated evaluation workflow, inference tool, read-only SQL evidence retrieval, similar incident retrieval, and deterministic LangGraph risk routing implemented. The workflow reports incomplete investigations when evidence tools fail. Optional local JSON traces record node execution, timing, evidence status, and failures. Twenty-five regression tests cover evidence retrieval, risk routing, tool failures, and tracing. **Planned:** semantic incident retrieval/RAG, LLM-assisted investigation, broader agent evaluations, OpenTelemetry, API deployment, containerization, and deployment automation.

> **Scope and safety:** This is a portfolio/research prototype using synthetic data. Its risk probabilities and recommendations are **not validated for production decisions**. It does not currently take autonomous remediation actions.

## 1. Mission and problem

Production data platforms run ingestion, ETL, synchronization, and reporting jobs. Failures may stem from resource pressure, workload, or transient processing issues. Teams need not only alerts but also reproducible risk estimates, relevant evidence, and explanations that can be tested.

**Mission:** Build an end-to-end system that evolves from **data generation → failure-risk prediction → tool-based investigation → stateful agent orchestration → rigorous evaluations → observable deployment**.

**Questions this project explores:**
- Can pre-execution signals estimate failure risk without leaking information from after a run?
- How should a prediction service be wrapped as a safe, reusable tool?
- How can a graph direct high-risk cases to further review while preserving traceable state?
- How do we evaluate predictive models *and* agent behavior separately?

## 2. What is implemented today

| Layer | Implementation | Status |
|---|---|---|
| Data | Reproducible Python generators and local SQLite tables | Implemented |
| Synthetic scenarios | 5,000 `realistic_incidents` rows with simulated pre-run conditions and simplified root-cause labels | Implemented |
| Predictive ML | Majority-class baseline, Logistic Regression, Random Forest | Implemented |
| Evaluation | Chronological 70/15/15 split; select model on validation average precision; tune threshold on validation F1; final test metrics and figures | Implemented |
| Inference | `predict_failure()` validates inputs, loads the local trusted model artifact, returns structured risk output | Implemented |
| Workflow | LangGraph shared state, validation and prediction nodes, conditional high/low-risk routes | Implemented (deterministic) |
| SQL investigation | Read-only, parameterized SQLite queries; time cutoff; failure counts and recorded labels | Implemented |
| Similar incident retrieval | Fixed-scale feature distance; prior failures; traceable run IDs | Implemented |
| Workflow reliability | Explicit investigation status; structured storage errors; preserve available evidence | Implemented |
| Local run tracing | Per-run JSON records with node durations, routes, evidence status and error types | Implemented (local only) |
| Regression tests | SQL filtering, time cutoff, empty results, risk routing, and incomplete evidence | Implemented |
| Agent reasoning, RAG, external telemetry, deployment | Not yet implemented | Roadmap |

**Terminology:** The current LangGraph system is a deterministic tool-using workflow, **not** an autonomous LLM agent. The elevated-risk branch now retrieves SQL evidence before risk review. Semantic retrieval/RAG, LLM-driven decisions, and guarded remediation are future milestones.

## 3. Architecture

```mermaid
flowchart TD
    G[Seeded synthetic generator] --> DB[(Local SQLite: agentops.db)]
    DB --> EV[Training / validation / final-test evaluation]
    EV --> M[Validated model artifact: local .joblib]
    IN[Pipeline pre-run input] --> V[LangGraph: validate_input]
    V --> P[LangGraph: prediction_node]
    P --> T[predict_failure tool]
    M --> T
    T --> R{Risk at or above threshold?}
    R -->|Yes| I[Read-only SQL investigation]
    DB --> I
    I -->|SQL available| S[Retrieve nearest failures]
    I -->|SQL unavailable| H
    DB --> S
    S --> H[Risk review with evidence status]
    R -->|No| L[Standard checks response]
    H --> OUT[Structured output]
    L --> OUT
```

In the workflow, **state** carries the input, prediction, SQL evidence, retrieved incidents and route; **nodes** do bounded work; **edges** define execution order; the **conditional edge** selects a route. The ML model performs inference—it does not orchestrate the graph.

For the current graph definition, see [the LangGraph workflow diagram](docs/langgraph_workflow.md). For planned architecture and evaluation decisions, see [architecture](docs/architecture.md), [evaluation methodology](docs/evaluation_methodology.md), and the [model card](docs/model_card.md).

## 4. Data design

Two original tables (`pipeline_runs`, `prediction_features`) were used to practice ingestion, schema design, SQL joins, and label validation. A separate `realistic_incidents` table supports later ML work.

The synthetic incident generator uses a fixed seed and a specified probability function so outcomes are reproducible and have a relationship to **pre-run features**:

| Pre-run feature | Description |
|---|---|
| `expected_workload` | Anticipated number of records or workload units |
| `cpu_usage_pct` | Simulated CPU utilization |
| `memory_usage_pct` | Simulated memory utilization |
| `recent_failure_rate` | Simulated historical failure-rate signal |

**Labels / post-run evidence:** `target_failed` indicates the simulated observed outcome, while `root_cause` is a simplified assigned incident label. Neither is used as a model input. Realistic incident ground truth and temporal dependencies remain areas for improvement.

**Previously observed synthetic dataset:** 5,000 rows; 1,886 failed runs (37.72%). Root-cause categories included transient processing error, memory pressure, CPU saturation, and high workload. These are synthetic counts, not operational benchmarks.

## 5. ML training and evaluation

We compare a majority-class baseline, Logistic Regression, and Random Forest. In the improved evaluation harness, data is sorted by `started_at`, then split chronologically:

- Training: 3,500 rows (70%)
- Validation: 750 rows (15%) — model selection and threshold tuning
- Final test: 750 rows (15%) — evaluation of the frozen selection

Logistic Regression was selected by **validation average precision**; the decision threshold **0.26** was chosen using validation F1. The final reported metrics from the synthetic held-out test set were:

| Metric | Value |
|---|---:|
| Precision (failure class) | 0.4688 |
| Recall (failure class) | 0.8633 |
| F1 (failure class) | 0.6076 |
| Average precision | 0.6688 |
| ROC-AUC | 0.7537 |
| Brier score | 0.1890 |

**Interpretation:** A threshold tuned for greater failure recall catches many simulated failures but produces false alerts. Neither these numbers nor the threshold should be taken as production performance guarantees. Because failure labels were generated using a logistic probability function, Logistic Regression has a favorable inductive bias in this experiment.

### Evaluation figures

![Final test confusion matrix](assets/final_confusion_matrix.png)

![Precision-recall curve](assets/precision_recall_curve.png)

![Calibration curve](assets/calibration_curve.png)

See [model card](docs/model_card.md) for assumptions, intended use, and limitations.

## 6. Project structure

```text
agentops-intelligence-lab/
├── assets/                            # Evaluation charts
├── data/                              # Local SQLite data (not committed)
├── docs/                              # Architecture, methodology, model card, graph diagram
├── evals/
│   └── evaluate_model.py              # Validation-based model selection and final evaluation
├── models/                            # Local model artifacts (not committed)
├── src/
│   ├── ingestion/
│   │   ├── generate_dataset.py
│   │   ├── generate_prediction_features.py
│   │   └── generate_realistic_incidents.py
│   ├── prediction/
│   │   └── Train_model.py              # Initial ML baseline experiment
│   ├── tools/
│   │   ├── prediction_tool.py          # Validated inference interface
│   │   ├── sql_investigation_tool.py   # Read-only incident evidence
│   │   └── incident_retrieval_tool.py   # Similar failed runs and source IDs
│   ├── observability/
│   │   ├── tracing.py                 # Local per-run node instrumentation
│   │   └── run_workflow.py            # Traced runner and high/low-risk examples
│   └── agents/
│       └── graph.py                    # Current deterministic LangGraph workflow
├── tests/
│   ├── test_investigation.py           # SQL and workflow regression checks
│   ├── test_incident_retrieval.py       # Ranking, filtering, and shared cutoff
│   ├── test_workflow_reliability.py     # Missing sources, partial evidence, error boundaries
│   └── test_run_tracing.py             # Paths, timings, failed nodes, isolation and CLI
├── requirements.txt
└── README.md
```

## 7. Reproduce locally

**Prerequisites:** Python 3.12 (the project was developed using Python 3.12) and a shell. SQLite is accessed through Python's standard-library `sqlite3`; SQLiteStudio is not required.

```bash
# Clone the project
 git clone https://github.com/AimanManzoor/agentops-intelligence-lab.git
 cd agentops-intelligence-lab

# Isolated environment
 python3 -m venv .venv
 source .venv/bin/activate
 python -m pip install -r requirements.txt
 mkdir -p data

# Recreate local synthetic incidents and train/validate the model
 python src/ingestion/generate_realistic_incidents.py
 python evals/evaluate_model.py

# Run inference and the graph
 python src/tools/prediction_tool.py
 python src/agents/graph.py
```

**Current fresh-clone setup caveat:** `generate_realistic_incidents.py` uses the project's `data/agentops.db` path. If `data/` is absent in a fresh clone, create it first with `mkdir -p data`. The `data/` folder and `.db` files are intentionally ignored by Git. The `.joblib` model must be generated locally by the evaluation script before inference or graph execution.

**Do not load untrusted `.joblib` files:** Python pickle/joblib deserialization can execute code. Only use artifacts produced by code and environments you trust.

### SQL evidence and regression checks

Run the standalone evidence tool:

```bash
python src/tools/sql_investigation_tool.py
```

Run the regression tests:

```bash
python -m unittest discover -s tests -v
```

The elevated-risk path is **prediction → investigate → retrieve_incidents → risk_review**; the lower-risk path skips SQL. The investigation tool opens SQLite in read-only mode and uses parameterized queries. It returns the source table, synthetic-data marker, UTC cutoff, total runs, failed runs, historical failure rate, and recorded failure labels.

An optional graph input, `as_of_utc`, sets the evidence cutoff; otherwise the tool uses the current UTC time. Only records whose `started_at` is strictly before that cutoff are included. Naive timestamps are interpreted as UTC. The synthetic schema does not include completion timestamps, so this filter is based on run start time.

A pipeline with no matching history returns zero counts, a null historical failure rate, and an empty label list. This is a successful query with no matching records, not a source failure.

The graph handles `FileNotFoundError` and SQLite errors from the two evidence tools. If SQL fails, it skips similar-incident retrieval and returns `investigation_status="unavailable"` with null evidence and retrieval. If only similar-incident retrieval fails, it preserves the SQL summary and returns `investigation_status="partial"`. Successful high-risk investigations return `complete`; lower-risk cases return `not_requested` and skip both tools. Structured `tool_errors` identify the tool and exception type. Model errors, input-validation errors, and unexpected programming errors still propagate; there is no automatic repair or retry.

See [workflow reliability](docs/workflow_reliability.md) for response examples, failure tests, research interpretation, and limitations.

The historical failure rate is an observed fraction across the selected pipeline's recorded runs, not a replacement for the model's case-specific risk estimate. Recorded synthetic labels supply context and do not establish the cause of a new case.

The regression suite uses a temporary SQLite database and controlled prediction outputs. They check workflow and SQL behavior rather than predictive accuracy, and do not require a trained model artifact. The separate local smoke checks exercised the trained model's elevated-risk and lower-risk examples.

See [similar incident retrieval](docs/incident_retrieval.md) for ranking, source references, and limitations.

### Local run traces

With the local model artifact and database already generated, run both examples:

```bash
python -m src.observability.run_workflow --example both
```

Each invocation saves `runs/<run_id>.json` and prints its path, node order, elapsed time,
route, and investigation status. `runs/` is ignored by Git. The trace distinguishes
completed execution with incomplete evidence from an unhandled workflow failure.
It records node names, durations, statuses and error types without recording input
features, database rows, final messages or raw exception messages. The synchronous
runner saves a trace before re-raising ordinary workflow exceptions when trace storage
is available. The existing untraced graph remains available through `build_graph()`.

This is local instrumentation, not an OpenTelemetry deployment or a production
latency benchmark. See [local run tracing](docs/run_tracing.md) for the schema,
Python interface, timing definitions, tests, and limits.

## 8. Learning goals and engineering decisions

| Concept | Where it appears | Why it matters |
|---|---|---|
| Data ingestion | `src/ingestion/` | Repeatable upstream inputs |
| Reproducibility | Fixed random seed and `requirements.txt` | Repeatable development runs |
| Leakage prevention | Pre-run features only | Avoid misleading predictions |
| Honest evaluation | Validation/test separation | Avoid selection bias |
| Inference boundary | `prediction_tool.py` | Reusable, validated tool contract |
| Stateful orchestration | LangGraph `AgentState` | Predictable data flow between nodes |
| Conditional routing | `route_by_risk()` | Clear risk-dependent workflow |
| Version control | Git commits and public repository | Traceable milestones |

**What this project demonstrates today:** the ability to connect data generation, validation, ML evaluation, model inference, conditional graph routing, SQL evidence, and similar incident retrieval. **What comes next:** semantic incident retrieval/RAG and evaluated LLM-assisted investigation.

## 9. Roadmap

- [x] Seeded synthetic pipeline data and SQLite storage
- [x] Baseline model training and visual evaluation
- [x] Train/validation/test evaluation and threshold tuning
- [x] Validated inference tool
- [x] Deterministic LangGraph graph and conditional routing
- [x] Regression tests for SQL evidence retrieval and both graph branches
- [ ] Automated tests for inference input validation and model-artifact handling
- [x] Read-only SQL investigation tool and incident evidence retrieval
- [x] Structured incident retrieval with source IDs
- [x] Explicit incomplete-investigation responses and tool-failure regression tests
- [x] Local node-level run traces and traced high/low-risk runner
- [ ] Historical search / RAG and context selection with source attribution
- [ ] LLM-based investigation and tool-choice policies with safeguards
- [ ] Agent evaluation harness: tool accuracy, groundedness, latency, regression tests
- [ ] OpenTelemetry traces, real telemetry ingestion and drift checks
- [x] GitHub Actions regression workflow
- [ ] FastAPI, Docker, deployment automation and monitoring

## 10. Current limitations

1. The incident dataset is synthetic; results are not generalizable without real-world evaluation.
2. The `recent_failure_rate` is simulated, not derived from real historical windows.
3. Root-cause labels are rule-assigned and are not independently verified causal explanations.
4. There is no live source ingestion, semantic incident retrieval/RAG, LLM reasoning, human approval workflow, or deployed service yet.
5. The local SQLite database and model binaries are intentionally excluded from the repository.

---

**Portfolio focus:** practical AI engineering fundamentals—data quality, explicit tool contracts, reproducible evaluations, explainable stateful workflows, and incremental deployment readiness.



## Automated regression checks

Workflow: `.github/workflows/tests.yml`.

On pushes to main, pull requests, or manual runs, GitHub starts an Ubuntu
job using Python 3.12. It installs requirements.txt, checks dependency
compatibility, and runs the regression suite.

The job fails if no tests are discovered or any test fails.
No API keys, local database, or trained model artifact are required.
Tests use controlled prediction outputs and temporary fixtures.

Run locally in the project environment:

    python -m pip check
    python -m unittest discover -s tests -v

The current suite contains 25 tests. A clean-environment local run passed
before preparing this workflow. Check the first hosted run after pushing:

https://github.com/AimanManzoor/agentops-intelligence-lab/actions

Select AgentOps regression tests and inspect any failed step.

A passing workflow checks covered software behavior. It does not establish
predictive accuracy, groundedness, or production reliability.
This job does not train models or deploy a service.

Making this check mandatory before merging requires a separate repository
ruleset or branch-protection configuration.
