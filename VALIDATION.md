# Validation

Run `python -m unittest discover -s tests -v` from the repository root.

GitHub Actions runs the same checks on Python 3.11 for every push and pull request. Tests use local fixtures and do not spend API credits. Training quality and application correctness are evaluated separately. A passing test suite is not evidence of production readiness or a benchmark score.

Benchmark reports record their dataset, configuration, seed, environment and measurement scope. Synthetic inputs and offline providers are explicitly identified. Historical results without saved artifacts are not presented as reproduced measurements.
