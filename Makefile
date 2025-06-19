.PHONY: help install quality test cov bandit cyclonedx

help:
	@echo "Commands:"
	@echo "  install    : Create a hatch environment and install dependencies."
	@echo "  quality    : Run all code quality checks (format, lint, types, security)."
	@echo "  test       : Run all tests with pytest."
	@echo "  cov        : Run tests and report code coverage."
	@echo "  bandit     : Run Bandit security scan."
	@echo "  cyclonedx  : Generate a CycloneDX SBOM."

install:
	@hatch env create

quality:
	@echo "✅ Running all quality checks..."
	@hatch run quality

test:
	@echo "🧪 Running tests..."
	@hatch run test

cov:
	@echo "📊 Running tests with coverage..."
	@hatch run cov

bandit:
	@echo "🔍 Running Bandit security scan..."
	@hatch run bandit-check

cyclonedx:
	@echo "📦 Generating CycloneDX SBOM..."
	@hatch run cyclonedx-py environment --pyproject pyproject.toml --output-file sbom.json --output-format JSON
