# Development Setup — RetentionIQ

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 16+ with pgvector extension (handled by docker-compose)
- Git + DVC

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/joaocordova/retentioniq.git
cd retentioniq

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Start infrastructure (PostgreSQL + pgvector + MLflow)
docker-compose up -d

# 5. Initialize database schema
python scripts/init_db.py

# 6. Generate synthetic franchise data
python scripts/generate_franchise_data.py --locations 250 --members 2000000 --months 18

# 7. Pull DVC-tracked data (if available)
dvc pull

# 8. Run the pipeline
dagster dev  # Opens Dagster UI at http://localhost:3000

# 9. Start the API (separate terminal)
uvicorn src.api.main:app --reload --port 8000

# 10. Run tests
make test
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your values. **Never commit `.env` to git.**

```bash
cp .env.example .env
# Edit .env with your values
```

The `.env.example` template contains all required variables:

```env
# Database (docker-compose defaults work for local dev)
POSTGRES_USER=retentioniq
POSTGRES_PASSWORD=changeme_in_production
POSTGRES_DB=retentioniq
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001

# LLM (for agent layer — required for agent endpoints)
ANTHROPIC_API_KEY=             # Claude API key
OPENAI_API_KEY=                # For embeddings (text-embedding-3-small)

# Feature Store
FEAST_REPO_PATH=./feature_store

# Monitoring
EVIDENTLY_DASHBOARD_PORT=8050
```

**Security note:** Docker Compose uses env var substitution with safe defaults for local dev. Production deployments must override `POSTGRES_PASSWORD` via environment or secrets management. No real credentials are stored in the codebase.

## Infrastructure (docker-compose)

```bash
docker-compose up -d     # Start all services
docker-compose down      # Stop all services
docker-compose logs -f   # Follow logs
```

Services:
- **postgres:** PostgreSQL 16 with pgvector on port `${POSTGRES_PORT:-5432}`
- **mlflow:** MLflow tracking server on port `${MLFLOW_PORT:-5001}`

All service credentials are injected via environment variables (see `.env.example`). Docker Compose uses safe defaults for local development.

## Makefile Commands

```bash
make setup       # Create venv + install deps + start docker
make data        # Generate synthetic data
make pipeline    # Run Dagster pipeline (full materialization)
make serve       # Start FastAPI server
make test        # Run all tests (unit + integration)
make test-unit   # Run unit tests only
make lint        # Run ruff + mypy
make format      # Auto-format with ruff
make monitor     # Start Evidently monitoring dashboard
make clean       # Remove artifacts, caches, temp files
make notebook    # Start Jupyter Lab
```

## Project Configuration

All configuration lives in `configs/` as YAML files. No hardcoded parameters.

```
configs/
├── data.yaml          # Data paths, schemas, quality thresholds
├── features.yaml      # Feature definitions, windows, aggregations
├── models.yaml        # Hyperparameters, training configs
├── causal.yaml        # DAG definition, treatment/outcome variables
├── optimization.yaml  # Budget constraints, scenario parameters
├── agents.yaml        # LLM model, temperature, tools, guardrails
└── monitoring.yaml    # Drift thresholds, alert channels
```

## Testing

```bash
# Unit tests (fast, no infra needed)
pytest tests/unit/ -v

# Integration tests (requires docker-compose running)
pytest tests/integration/ -v

# Data quality tests (Great Expectations)
python -m src.data.quality.run_all

# Agent evaluation
python -m src.agents.eval.run --scenarios configs/eval_scenarios.yaml

# Full test suite with coverage
pytest --cov=src --cov-report=html tests/
```

## Troubleshooting

**pgvector not found:** Ensure PostgreSQL container is running with the pgvector extension. Check with: `docker exec -it retentioniq-postgres psql -U retentioniq -c "CREATE EXTENSION IF NOT EXISTS vector;"`

**MLflow connection refused:** Verify MLflow container is running: `docker-compose ps`. Check logs: `docker-compose logs mlflow`

**DVC pull fails:** DVC remote may not be configured. For local development, generate data with `make data` instead.

**CUDA not available:** The project runs on CPU by default. GPU acceleration is optional and only impacts training speed for XGBoost models. Causal inference and optimization run on CPU.
