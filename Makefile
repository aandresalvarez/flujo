dev:
	poetry install --with dev

test:
	poetry run pytest --cov=pydantic_ai_orchestrator --cov-report=term-missing 