# Dashboard export security boundary

## Scope
The dashboard uses committed synthetic data and browser-local calculations.
This milestone hardens CSV exports before accepting external telemetry.
It does not add an authenticated backend or establish production security.

## Controls
CSV fields are quoted and embedded quotes are doubled. Formula-like text,
including leading whitespace, control characters and full-width formula
prefixes, receives an apostrophe prefix. Actual numeric values retain their
representation. Source records are not modified.

Six security regression tests cover formula prefixes, whitespace/control
characters, ordinary values, field escaping, export integration/non-mutation
and empty exports. GitHub Actions runs these alongside existing checks.

Preview only the documentation directory on loopback:
`python3 -m http.server 8000 --bind 127.0.0.1 --directory docs`

## Limits
Tests verify serialization contracts, not spreadsheet execution or resistance
to every attack. Spreadsheet applications handle apostrophes differently;
saving and reopening CSV files may remove protections. These exports are
for spreadsheet viewing rather than lossless data interchange.

Publish only synthetic or approved data. Ignored files can still exist in
Git history. Only load locally generated, trusted joblib models: untrusted
pickle/joblib files can execute code before metadata validation. A checksum
stored alongside an artifact does not authenticate its source.

Before deploying a public inference API, add authentication, authorization,
request/resource limits, secret management, dependency vulnerability
monitoring and deployment-specific security tests.

## Research interpretation
Security tests, workflow contracts and predictive evaluation measure
different properties. Passing one does not establish the others.
Next evaluate false-alert costs and incident-retrieval relevance on
independent scenarios. Synthetic metrics do not establish real-world quality.

References:
- https://owasp.org/www-community/attacks/CSV_Injection
- https://scikit-learn.org/stable/model_persistence.html
