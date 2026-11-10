# Workflow reliability: incomplete evidence

## Problem and resulting behavior

Previously, a missing SQLite database or a storage error in either evidence tool
stopped an elevated-risk graph execution with an exception. A caller could not
receive the completed model prediction alongside an explicit statement that
historical evidence was unavailable.

The graph now catches `FileNotFoundError` and `sqlite3.Error` from SQL investigation
and similar-incident retrieval. It keeps the prediction unchanged and returns an
explicit evidence status. These handlers live at the graph's tool boundary; the
standalone tools still raise their original errors.

| Situation | Final `investigation_status` | Evidence behavior |
|---|---|---|
| Elevated risk; both tools succeed | `complete` | Return SQL summary and retrieval results |
| Elevated risk; no historical rows match | `complete` | Zero counts, null rate, empty matches; no tool error |
| SQL tool cannot read its source | `unavailable` | Set evidence and retrieval to null; skip retrieval |
| SQL succeeds but similar-incident retrieval fails | `partial` | Preserve SQL summary; set retrieval to null |
| Lower predicted risk | `not_requested` | Skip both tools |

`complete` means the requested tools completed; it does not certify that the
prediction is accurate, that the evidence is sufficient, or that a cause is known.
An unknown pipeline is treated as having no matching history under the current
data contract; there is no separate pipeline registry for validating its existence.

## Structured response examples

When the SQL database is missing, the result includes:

```json
{
  "route_taken": "risk_review",
  "investigation_status": "unavailable",
  "evidence": null,
  "retrieval": null,
  "tool_errors": [
    {"tool": "sql_investigation", "error_type": "FileNotFoundError"}
  ]
}
```

The final message keeps the model's case-specific probability, explicitly states
that SQL evidence is unavailable and retrieval was skipped, and recommends checking
source availability and investigating manually. It does not claim there were zero
historical failures. Pre-existing high-risk evidence fields supplied with the input
are cleared when the SQL call fails, so they cannot be reused as this review's evidence.

If SQL succeeded and retrieval failed, the returned SQL counts and historical rate
remain available. The final message reports retrieval unavailability rather than
claiming that no similar failures exist. It supplies no fabricated source IDs.

`tool_errors` contains only the tool name and exception type, not raw exception
messages or filesystem paths. `pending_retrieval` is an intermediate graph state
after successful SQL investigation, not a completed investigation status.

Input-validation errors, missing or invalid model artifacts, and unexpected
programming errors are outside these recovery handlers and still propagate.
The graph does not retry calls, repair a database, or take remediation actions.

## Verification

From the repository root, with project dependencies installed:

```bash
python -m unittest discover -s tests -v
```

The suite contains 15 tests: eight existing SQL/retrieval checks plus seven new
workflow reliability tests. The new tests cover healthy sources, empty history,
an actually missing temporary database, an actually missing SQL table, injected
retrieval storage errors, low-risk bypass, and errors that must still propagate.
They use temporary SQLite data, the real graph and read-only tools where applicable,
and controlled prediction outputs. They require no trained model artifact or API key.

The retrieval-failure test injects both a file-not-found error and a SQLite storage
error. The missing-database test also supplies stale evidence to verify that a failed
investigation cannot present it as newly retrieved evidence. Existing tests continue
to check cutoff sharing, pipeline filtering, ranking, and read-only behavior.

## Research and interview interpretation

**Question:** Can the workflow preserve an available prediction and accurately
communicate evidence availability when a dependent tool fails?

The intervention is bounded error handling at two tool boundaries. Controlled
failure scenarios specify the expected route, tool calls, retained evidence,
structured status, and final-message content. Healthy-source and empty-history
cases serve as controls. These checks support a claim about tested workflow
behavior, not a measured production failure rate or model-accuracy improvement.

The central distinction is between a prediction, observed historical evidence,
missing evidence, and a causal explanation. A model estimate remains an estimate
when evidence tools fail. A missing database cannot establish that no prior failures
exist. A nearby synthetic incident cannot establish the cause of a new run.

An accurate interview description is: "I added explicit partial and unavailable
investigation states to a LangGraph pipeline workflow. Storage failures preserve
the model output and any valid SQL evidence, skip dependent work when necessary,
and produce a response without inventing history. I verified this with controlled
failure tests alongside the existing regression suite."

Limitations remain: synthetic data, controlled model outputs in these tests,
start-time-only historical cutoffs, no completed-at availability timestamp, no
automated recovery, no external traces, and no validation with production incidents.
