# AgentOps LangGraph Workflow

```mermaid
flowchart TD
    V[Validate input] --> P[Predict failure risk]
    P -->|Elevated risk| I[Read SQL evidence]
    P -->|Lower risk| N[Standard checks]
    I -->|SQL available| R[Retrieve prior failures]
    I -->|SQL unavailable| H[Risk review]
    R -->|Results or explicit retrieval error| H
    H --> E[Return structured result]
    N --> E
```

Successful investigations return `complete`, including those with no matching
history. SQL storage failures return `unavailable` and skip retrieval. Retrieval
storage failures return `partial` while preserving SQL evidence. Lower-risk runs
return `not_requested` and skip both tools. Input/model errors and unexpected
programming errors still propagate.

See [workflow reliability](workflow_reliability.md) for the response contract and tests.

