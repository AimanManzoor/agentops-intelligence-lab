# Similar historical incident retrieval

The elevated-risk path retrieves the three nearest eligible failed runs.
Candidates must have the same pipeline name and a start timestamp strictly before the cutoff.
The SQL summary and retriever share the same normalized UTC cutoff.

Ranking uses Euclidean distance over four features divided by fixed scales:

| Feature | Scale |
|---|---:|
| Expected workload | 10000 |
| CPU usage | 100 |
| Memory usage | 100 |
| Recent failure rate | 1 |

Lower distance means closer operating conditions. Ties are ordered by run ID.
The workload scale is a reference value, not an enforced maximum.
Each result contains a source ID, run ID, timestamp, features, distance, and recorded synthetic label.
Labels are excluded from distance calculations and do not establish a new case's cause.

This is structured retrieval. Semantic RAG and LLM generation remain future work.
Distance is not probability or confidence. There is currently no relevance threshold.
The schema has start timestamps, without completion or availability timestamps.
Tests verify ranking and workflow behavior; empirical retrieval relevance has not been measured.

Run the tool: `python src/tools/incident_retrieval_tool.py`

Run all tests: `python -m unittest discover -s tests -v`
