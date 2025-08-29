# Repository Guidelines

## Project Structure & Module Organization
- `src/stockforecast/`: Core library code and modules.
- `tests/`: Unit/integration tests mirroring `src` (e.g., `tests/module/test_api.py`).
- `scripts/`: One-off or CLI scripts for data prep/training.
- `configs/`: YAML/JSON configs for experiments and env-specific settings.
- `notebooks/`: Exploratory analysis; keep outputs cleared before commit.
- `data/`: Local datasets and artifacts (git-ignored). Add small fixtures under `tests/fixtures/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install (editable + dev): `pip install -e .[dev]`
- Run package: `python -m stockforecast` or a script in `scripts/`.
- Tests: `pytest -q` (use `-k <pattern>` to filter).
- Lint/format: `ruff check .` and `black .`
- Types: `mypy src`
- If a `Makefile` exists: `make install`, `make test`, `make lint`, `make run` provide the same.

## Coding Style & Naming Conventions
- Python 3.x, 4-space indent, UTF-8.
- Format with `black`, lint with `ruff`, organize imports with `isort` (or `ruff --fix`).
- Naming: modules/files `snake_case`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE_CASE`.
- Docstrings: concise Google-style; prefer type hints on all public APIs.

## Testing Guidelines
- Framework: `pytest` with `pytest-cov` if configured.
- Layout: mirror `src` paths; test files `test_*.py`; fixtures in `tests/conftest.py` or `tests/fixtures/`.
- Aim for meaningful coverage; add a failing test with every bug fix.
- Run with coverage: `pytest --cov=stockforecast --cov-report=term-missing`.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Keep messages imperative and scoped.
- Branches: `feature/<slug>`, `fix/<slug>`, `chore/<slug>`.
- PRs: link issues, describe changes and rationale, include before/after results (metrics/plots) when relevant, and note any migration steps.
- Checks: tests green, linters clean, no large artifacts in Git.

## Security & Configuration Tips
- Never commit secrets. Use `.env` (git-ignored) and add `example.env` for defaults.
- Keep `data/` and generated artifacts out of Git; use paths from configs/env vars.
- Pin dependencies and update thoughtfully to keep forecasts reproducible.
