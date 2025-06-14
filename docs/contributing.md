# Contributing Guide

Thank you for your interest in contributing to `pydantic-ai-orchestrator`! This guide will help you get started.

## Development Setup

### 1. Prerequisites

- Python 3.11 or higher
- Git
- Make (optional, for using Makefile)
- Virtual environment (recommended)

### 2. Repository Setup

```bash
# Clone the repository
git clone https://github.com/aandresalvarez/rloop.git
cd rloop

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Development Tools

The project uses several development tools:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing
- **pre-commit**: Git hooks

Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### 1. Branching Strategy

- `main`: Production-ready code
- `develop`: Development branch
- Feature branches: `feature/description`
- Bug fix branches: `fix/description`
- Release branches: `release/version`

### 2. Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Make your changes:
   - Follow the [Coding Standards](#coding-standards)
   - Write tests for new features
   - Update documentation

3. Run tests and checks:
   ```bash
   # Run all checks
   make check

   # Or run individually
   make format    # Format code
   make lint      # Run linters
   make type      # Run type checker
   make test      # Run tests
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

5. Push and create a pull request:
   ```bash
   git push origin feature/your-feature
   ```

### 3. Pull Request Process

1. **Before Submitting**
   - Update documentation
   - Add tests
   - Run all checks
   - Update changelog

2. **Pull Request Template**
   - Description of changes
   - Related issues
   - Testing performed
   - Documentation updates

3. **Review Process**
   - Code review
   - CI checks
   - Documentation review
   - Final approval

## Coding Standards

### 1. Python Style

Follow [PEP 8](https://pep8.org/) and use Black for formatting:

```bash
# Format code
make format

# Check formatting
make format-check
```

### 2. Type Hints

Use type hints for all functions and classes:

```python
from typing import Optional, List

def process_items(
    items: List[str],
    limit: Optional[int] = None
) -> List[str]:
    """Process a list of items.
    
    Args:
        items: List of items to process
        limit: Optional limit on items
        
    Returns:
        Processed items
    """
    # Implementation
    return items[:limit] if limit else items
```

### 3. Documentation

Follow [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings:

```python
def calculate_score(
    items: List[float],
    weights: Optional[List[float]] = None
) -> float:
    """Calculate weighted score for items.
    
    Args:
        items: List of scores to process
        weights: Optional weights for each score
        
    Returns:
        Weighted average score
        
    Raises:
        ValueError: If weights length doesn't match items
    """
    # Implementation
    pass
```

### 4. Testing

Write tests for all new features:

```python
import pytest

def test_calculate_score():
    """Test score calculation."""
    items = [1.0, 2.0, 3.0]
    weights = [0.5, 0.3, 0.2]
    
    score = calculate_score(items, weights)
    assert score == pytest.approx(1.9)
    
    # Test without weights
    score = calculate_score(items)
    assert score == pytest.approx(2.0)
    
    # Test error case
    with pytest.raises(ValueError):
        calculate_score(items, [0.5, 0.5])
```

## Project Structure

```
pydantic-ai-orchestrator/
├── src/
│   └── pydantic_ai_orchestrator/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   ├── pipeline.py
│       │   └── agents.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── candidate.py
│       │   └── task.py
│       ├── tools/
│       │   ├── __init__.py
│       │   └── base.py
│       └── utils/
│           ├── __init__.py
│           ├── telemetry.py
│           └── validation.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_orchestrator.py
│   ├── test_pipeline.py
│   └── test_agents.py
├── docs/
│   ├── index.md
│   ├── installation.md
│   └── usage.md
├── examples/
│   ├── feature.py
│   └── advanced.py
├── pyproject.toml
├── setup.py
├── LICENSE
└── README.md
```

For more information about the project structure and examples, please refer to:
- [Examples directory](https://github.com/aandresalvarez/rloop/tree/main/examples)
- [Code of Conduct](https://github.com/aandresalvarez/rloop/blob/main/CODE_OF_CONDUCT.md)
- [License](https://github.com/aandresalvarez/rloop/blob/main/LICENSE)

## Testing

### 1. Running Tests

```