# Scoring Guide

This guide explains how to use and customize the scoring system in `flujo`.

## Overview

The orchestrator includes several scoring mechanisms:

- Ratio-based scoring
- Weighted scoring
- Model-based scoring
- Custom scoring functions

## Basic Scoring

### Ratio Score

The simplest scoring method counts passed checklist items:

```python
from flujo.domain.scoring import ratio_score

# Calculate a simple pass/fail ratio
score = ratio_score(checklist)
# Returns a float between 0.0 and 1.0
```

### Weighted Score

Assign different weights to checklist items:

```python
from flujo.domain.scoring import weighted_score

# Define weights for different criteria
weights = {
    "correctness": 0.5,
    "readability": 0.3,
    "efficiency": 0.2
}

# Calculate weighted score
score = weighted_score(checklist, weights)
```

## Advanced Scoring

### Model-Based Scoring

Use an AI model to evaluate quality:

```python
from flujo.infra.agents import make_agent_async
from flujo.domain.scoring import model_score

# Create a scoring agent
scorer_agent = make_agent_async(
    "openai:gpt-4",
    "You are a quality evaluator. Score the solution from 0 to 1.",
    float
)

# Use model-based scoring
score = model_score(checklist, scorer_agent)
```

### Custom Scoring

Create your own scoring function:

```python
def custom_scorer(checklist):
    """Calculate a custom score based on checklist items."""
    total_score = 0
    total_weight = 0

    for item in checklist.items:
        # Define custom weights based on item type
        weight = 1.0
        if "critical" in item.description.lower():
            weight = 2.0
        elif "optional" in item.description.lower():
            weight = 0.5

        # Add to total
        total_score += weight * (1.0 if item.passed else 0.0)
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0

# Use in pipeline
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(make_solution_agent())
    >> Step.validate(
        make_validator_agent(),
        scorer=custom_scorer
    )
)
```

## Scoring in Pipelines

### Step-Level Scoring

Configure scoring for individual steps:

```python
# Review step with custom scoring
review_step = Step.review(
    make_review_agent(),
    scorer=lambda c: weighted_score(c, {
        "completeness": 0.4,
        "clarity": 0.6
    })
)

# Validation step with model scoring
validate_step = Step.validate(
    make_validator_agent(),
    scorer=lambda c: model_score(c, scorer_agent)
)
```

### Pipeline-Level Scoring

Configure scoring for the entire pipeline:

```python
# Create a pipeline with custom scoring
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(make_solution_agent())
    >> Step.validate(make_validator_agent())
)

# Configure the runner with custom scoring
runner = Flujo(
    pipeline,
    scorer=lambda c: weighted_score(c, {
        "review_score": 0.3,
        "solution_score": 0.5,
        "validation_score": 0.2
    })
)
```

## Scoring Best Practices

### 1. Define Clear Criteria

```python
# Example checklist with clear criteria
checklist = Checklist(items=[
    ChecklistItem(
        description="Code follows PEP 8 style guide",
        category="style",
        critical=True
    ),
    ChecklistItem(
        description="Includes docstrings for all functions",
        category="documentation",
        critical=True
    ),
    ChecklistItem(
        description="Has unit tests",
        category="testing",
        critical=False
    )
])
```

### 2. Use Appropriate Weights

```python
# Example weights for code generation
code_weights = {
    "syntax": 0.3,      # Basic correctness
    "style": 0.2,       # Code style
    "documentation": 0.2,  # Documentation
    "testing": 0.2,     # Test coverage
    "performance": 0.1  # Performance considerations
}

# Example weights for content generation
content_weights = {
    "grammar": 0.3,     # Grammar and spelling
    "style": 0.3,       # Writing style
    "tone": 0.2,        # Appropriate tone
    "clarity": 0.2      # Clear communication
}
```

### 3. Implement Progressive Scoring

```python
def progressive_scorer(checklist):
    """Score that requires critical items to pass."""
    # First, check critical items
    critical_items = [i for i in checklist.items if i.critical]
    if not all(i.passed for i in critical_items):
        return 0.0  # Fail if any critical item fails

    # Then, calculate weighted score for remaining items
    non_critical = [i for i in checklist.items if not i.critical]
    return weighted_score(Checklist(items=non_critical), {
        "style": 0.4,
        "documentation": 0.3,
        "testing": 0.3
    })
```

### 4. Use Model Scoring Wisely

```python
# Create a specialized scoring agent
code_scorer = make_agent_async(
    "openai:gpt-4",
    """You are a code quality expert. Evaluate the code based on:
    1. Correctness (40%)
    2. Readability (30%)
    3. Efficiency (30%)
    Return a score between 0 and 1.""",
    float
)

# Use in pipeline
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(make_solution_agent())
    >> Step.validate(
        make_validator_agent(),
        scorer=lambda c: model_score(c, code_scorer)
    )
)
```

## Examples

### Code Generation Scoring

```python
from flujo import Step, Flujo
from flujo.plugins import (
    SQLSyntaxValidator,
    CodeStyleValidator
)

# Define code-specific weights
code_weights = {
    "syntax": 0.3,
    "style": 0.2,
    "documentation": 0.2,
    "testing": 0.2,
    "performance": 0.1
}

# Create a code generation pipeline
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(code_agent)
    >> Step.validate(
        make_validator_agent(),
        plugins=[
            SQLSyntaxValidator(),
            CodeStyleValidator()
        ],
        scorer=lambda c: weighted_score(c, code_weights)
    )
)
```

### Content Generation Scoring

```python
# Define content-specific weights
content_weights = {
    "grammar": 0.3,
    "style": 0.3,
    "tone": 0.2,
    "clarity": 0.2
}

# Create a content generation pipeline
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(writer_agent)
    >> Step.validate(
        make_validator_agent(),
        scorer=lambda c: weighted_score(c, content_weights)
    )
)
```

## Troubleshooting

### Common Issues

1. **Inconsistent Scores**
   - Check weight definitions
   - Verify checklist items
   - Review scoring function
   - Monitor model outputs

2. **Performance Issues**
   - Cache model scores
   - Use simpler scoring when possible
   - Batch evaluations
   - Monitor costs

3. **Quality Issues**
   - Review scoring criteria
   - Adjust weights
   - Update checklist items
   - Calibrate model scoring

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](../user_guide/usage.md)
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](../user_guide/use_cases.md)
