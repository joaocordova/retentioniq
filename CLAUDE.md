# CLAUDE.md

## What is this project?
RetentionIQ — a prescriptive retention system for multi-location subscription businesses (fitness franchise with 250+ locations). Combines ML, causal inference, optimization, and LLM agents.

## Tech stack
Python 3.11, PostgreSQL 16 + pgvector, FastAPI, Dagster, MLflow, Feast, Docker, LangGraph, DoWhy, EconML, Pyomo, Evidently, Great Expectations, GitHub Actions.

## How to run
```bash
make setup    # install + start infra
make data     # generate synthetic data
make pipeline # run Dagster pipeline
make serve    # start API
make test     # run tests
```

## Code structure
- `src/data/` — medallion layers (bronze/silver/gold) + data quality
- `src/features/` — feature engineering + Feast definitions
- `src/models/` — survival, churn, LTV models
- `src/causal/` — DoWhy DAG, EconML estimation, causal forests
- `src/optimization/` — Pyomo budget optimizer
- `src/agents/` — LangGraph multi-agent system
- `src/api/` — FastAPI endpoints
- `src/monitoring/` — Evidently drift detection
- `tests/` — unit + integration tests
- `notebooks/` — exploration only, not production code
- `configs/` — all YAML config, no hardcoded params

## Key rules
- Type hints on all functions. Docstrings on public functions.
- ruff for formatting (line-length = 100)
- No random train/test splits — always temporal
- PII masked before any LLM call
- Every architecture decision documented as ADR in docs/ARCHITECTURE.md
- **No hardcoded secrets.** Use env vars (see `.env.example`). `.env` is gitignored.
- All configs in `configs/*.yaml` — never hardcode params in Python code
- structlog for all logging (structured JSON)
- Custom exceptions in `src/exceptions.py` — never bare `raise Exception`
