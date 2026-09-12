# Contributing to RuntimeFit

RuntimeFit's recommendation must remain explainable and reproducible. Changes to
selection behavior should include a regression test that shows the evidence,
requirement, and expected decision.

## Development

Use Python 3.11 or newer:

```bash
python -m pip install -e '.[dev]'
ruff check src tests scripts
ruff format --check src tests scripts
mypy src/runtimefit
PYTHONPATH=src python -m unittest discover -s tests -v
```

Before opening a pull request:

- do not include private prompts, credentials, internal endpoints, or proprietary
  benchmark data;
- label synthetic and smoke-test evidence clearly;
- preserve raw provider evidence and its provenance for performance claims; and
- disclose runtime, model, workload, hardware, caching, and cost assumptions.

Bug reports should include a minimal sanitized configuration and the RuntimeFit
version. Security reports should follow [SECURITY.md](SECURITY.md).
