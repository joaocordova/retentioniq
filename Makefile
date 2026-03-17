.PHONY: setup data pipeline serve test test-unit test-integration lint format monitor clean notebook

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"
	docker-compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	python scripts/init_db.py
	@echo "✓ Setup complete. Activate venv: source .venv/bin/activate"

data:
	python scripts/generate_franchise_data.py \
		--locations 250 \
		--members 2000000 \
		--months 18 \
		--seed 42 \
		--output data/raw/
	@echo "✓ Synthetic data generated in data/raw/"

pipeline:
	dagster asset materialize --select "*" -m src.data

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v -m unit --tb=short

test-integration:
	pytest tests/integration/ -v -m integration --tb=short

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

monitor:
	python -m src.monitoring.dashboard --port 8050

clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist/ build/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Cleaned"

notebook:
	jupyter lab --no-browser --port 8888
