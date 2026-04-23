# Contributing
Thank you for your interest in improving **AI-Industrial Predictive Safety**.
This document describes how to propose changes to the project.
## Ways to contribute
- **Report bugs** via the [issue tracker](https://github.com/Allarayzer/ai-industrial-predictive-safety/issues)
  using the bug report template.
- **Suggest features** using the feature request template. Please describe
  the industrial use case, not just the code change.
- **Improve documentation** — clarifications, corrections, and additional
  examples are always welcome.
- **Submit pull requests** for bug fixes or new capabilities.
## Development setup
1. Fork the repository and clone your fork.
2. Create a virtual environment and install the package in editable mode
   with development dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Run the test suite to verify your setup:
   ```bash
   pytest tests/
   ```
## Pull request guidelines
- Create a topic branch off `main`: `git checkout -b my-feature`.
- Keep changes focused. A pull request should address one concern.
- Add tests that exercise new behavior; aim to keep coverage at or above
  the current level.
- Follow existing style. The project uses `ruff` for linting; run
  `ruff check src/ tests/` before submitting.
- Write clear commit messages in the imperative mood
  (e.g. "Add spectral features", not "Added spectral features").
- Update `CHANGELOG.md` under the `[Unreleased]` heading describing your
  change.
- If your change affects the public API, update `docs/api.md` accordingly.
## Code style
- Python 3.10+.
- Type hints on public functions.
- Docstrings in NumPy style (Parameters / Returns / Notes sections).
- Prefer small, composable modules; functions exceeding ~50 lines should
  be refactored.
## Scientific accuracy
Contributions that claim reproductions of published results must include:
- A reference to the original publication.
- Enough implementation detail for independent verification.
- Runnable benchmark scripts in `benchmarks/`.
## Reporting security issues
Please do not file public issues for security concerns. See
[`SECURITY.md`](SECURITY.md) for the reporting process.
## Code of conduct
By participating in this project you agree to abide by the
[Contributor Covenant](CODE_OF_CONDUCT.md).
