# Testing Guide

This guide covers testing strategies and best practices for Flujo workflows, from unit tests to end-to-end testing.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Structure](#test-structure)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Mocking Strategies](#mocking-strategies)
- [Test Utilities](#test-utilities)
- [Best Practices](#best-practices)
- [Debugging Tests](#debugging-tests)

## Testing Philosophy

Flujo follows these testing principles:

1. **Comprehensive Coverage**: Test all public APIs and critical paths
2. **Isolation**: Unit tests should be independent and fast
3. **Realistic Scenarios**: Integration tests should reflect real usage
4. **Maintainability**: Tests should be easy to understand and modify
5. **Performance**: Tests should run quickly and efficiently

## Test Structure

The test suite is organized as follows:

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for component interactions
├── e2e/              # End-to-end tests for complete workflows
├── benchmarks/       # Performance benchmarks
└── conftest.py       # Shared test fixtures and configuration
```

## Unit Testing

### Testing Agents

```python
import pytest
from flujo import make_agent_async
from flujo.testing import MockBackend

@pytest.mark.asyncio
async def test_custom_agent():
    # Create a mock backend for testing
    mock_backend = MockBackend()
    
    # Create agent with mock backend
    agent = make_agent_async(
        "mock:test-model",
        "You are a helpful assistant.",
        str,
        backend=mock_backend
    )
    
    # Test agent execution
    result = await agent.run("Hello")
    assert result == "Mock response"
    assert mock_backend.calls == [("test-model", "Hello")]
```

### Testing Pipeline Steps

```python
import pytest
from flujo import Step, Flujo
from flujo.testing import MockAgent

@pytest.mark.asyncio
async def test_pipeline_step():
    # Create mock agent
    mock_agent = MockAgent("Mock response")
    
    # Create pipeline with mock agent
    pipeline = Step.solution(mock_agent)
    runner = Flujo(pipeline)
    
    # Test pipeline execution
    result = runner.run("Test input")
    
    assert result.step_history[0].success
    assert result.step_history[0].output == "Mock response"
    assert mock_agent.calls == ["Test input"]
```

### Testing Recipes

```python
import pytest
from flujo.recipes import Default
from flujo.testing import MockAgent

@pytest.mark.asyncio
async def test_default_recipe():
    # Create mock agents
    review_agent = MockAgent("Review checklist")
    solution_agent = MockAgent("Generated solution")
    validator_agent = MockAgent("Validation result")
    
    # Create recipe with mock agents
    recipe = Default(
        review_agent=review_agent,
        solution_agent=solution_agent,
        validator_agent=validator_agent
    )
    
    # Test recipe execution
    result = await recipe.run_async("Test task")
    
    assert result is not None
    assert result.solution == "Generated solution"
    assert review_agent.calls == ["Test task"]
    assert solution_agent.calls == ["Test task"]
    assert validator_agent.calls == ["Test task"]
```

## Integration Testing

### Testing Pipeline DSL

```python
import pytest
from flujo import Step, Flujo, Pipeline
from flujo.testing import MockAgent

@pytest.mark.asyncio
async def test_pipeline_composition():
    # Create mock agents
    agent1 = MockAgent("Step 1 result")
    agent2 = MockAgent("Step 2 result")
    
    # Compose pipeline
    pipeline = Step.solution(agent1) >> Step.solution(agent2)
    
    # Test pipeline execution
    runner = Flujo(pipeline)
    result = runner.run("Input")
    
    # Verify step execution order
    assert len(result.step_history) == 2
    assert result.step_history[0].output == "Step 1 result"
    assert result.step_history[1].output == "Step 2 result"
    
    # Verify agent calls
    assert agent1.calls == ["Input"]
    assert agent2.calls == ["Step 1 result"]
```

### Testing Context Sharing

```python
import pytest
from flujo import Step, Flujo
from flujo.domain.models import PipelineContext
from flujo.testing import MockAgent

class TestContext(PipelineContext):
    counter: int = 0

@pytest.mark.asyncio
async def test_context_sharing():
    # Create agents that modify context
    def agent1(input_data: str, context: TestContext) -> str:
        context.counter += 1
        return f"Step 1: {input_data}"
    
    def agent2(input_data: str, context: TestContext) -> str:
        context.counter *= 2
        return f"Step 2: {input_data}"
    
    # Create pipeline
    pipeline = (
        Step.from_mapper(agent1) >> 
        Step.from_mapper(agent2)
    )
    
    # Test with context
    runner = Flujo(pipeline, context_model=TestContext)
    result = runner.run("test")
    
    # Verify context was shared and modified
    final_context = result.final_pipeline_context
    assert final_context.counter == 2  # (0+1)*2
```

### Testing Tools Integration

```python
import pytest
from flujo import Step, Flujo
from flujo.testing import MockTool

@pytest.mark.asyncio
async def test_tool_integration():
    # Create mock tool
    mock_tool = MockTool("Tool result")
    
    # Create agent with tool
    agent = make_agent_async(
        "mock:test-model",
        "Use the tool to process input.",
        str,
        tools=[mock_tool]
    )
    
    # Create pipeline
    pipeline = Step.solution(agent)
    runner = Flujo(pipeline)
    
    # Test execution
    result = runner.run("Process this")
    
    assert result.step_history[0].success
    assert mock_tool.calls == ["Process this"]
```

## End-to-End Testing

### Testing Complete Workflows

```python
import pytest
from flujo.recipes import Default, AgenticLoop
from flujo import Task

@pytest.mark.asyncio
async def test_default_workflow_e2e():
    """Test a complete Default recipe workflow."""
    recipe = Default()
    task = Task(prompt="Write a simple Python function")
    
    result = await recipe.run_async(task)
    
    # Verify result structure
    assert result is not None
    assert hasattr(result, 'solution')
    assert hasattr(result, 'score')
    assert hasattr(result, 'checklist')
    
    # Verify solution quality
    assert result.score > 0.0
    assert result.score <= 1.0
    assert "def" in result.solution.lower()

@pytest.mark.asyncio
async def test_agentic_loop_e2e():
    """Test a complete AgenticLoop workflow."""
    # Define simple tool
    async def simple_tool(query: str) -> str:
        return f"Tool result for: {query}"
    
    # Create AgenticLoop
    loop = AgenticLoop(
        planner_agent=make_agent_async("openai:gpt-4o", "Use the tool.", str),
        agent_registry={"simple_tool": simple_tool}
    )
    
    result = loop.run("Use the tool to get information")
    
    # Verify execution
    assert result.final_pipeline_context is not None
    assert len(result.final_pipeline_context.command_log) > 0
```

### Testing with Real APIs

```python
import pytest
import vcr
from flujo.recipes import Default
from flujo import Task

@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_with_real_openai():
    """Test with real OpenAI API (recorded with VCR)."""
    recipe = Default()
    task = Task(prompt="What is 2+2?")
    
    result = await recipe.run_async(task)
    
    assert result is not None
    assert "4" in result.solution
    assert result.score > 0.5

# VCR configuration
my_vcr = vcr.VCR(
    cassette_library_dir='tests/cassettes',
    record_mode='once',
    match_on=['uri', 'method', 'headers'],
    filter_headers=['authorization']
)
```

## Mocking Strategies

### Mock Backends

```python
from flujo.testing import MockBackend

# Simple mock backend
mock_backend = MockBackend()

# Mock with custom responses
mock_backend = MockBackend(responses={
    "test-input": "test-output",
    "another-input": "another-output"
})

# Mock with dynamic responses
def dynamic_response(input_data: str) -> str:
    return f"Processed: {input_data}"

mock_backend = MockBackend(response_func=dynamic_response)
```

### Mock Agents

```python
from flujo.testing import MockAgent

# Simple mock agent
mock_agent = MockAgent("Fixed response")

# Mock agent with different responses
mock_agent = MockAgent(responses=[
    "First response",
    "Second response",
    "Third response"
])

# Mock agent with custom logic
def custom_agent_logic(input_data: str) -> str:
    if "error" in input_data.lower():
        raise ValueError("Simulated error")
    return f"Processed: {input_data}"

mock_agent = MockAgent(response_func=custom_agent_logic)
```

### Mock Tools

```python
from flujo.testing import MockTool

# Simple mock tool
mock_tool = MockTool("Tool result")

# Mock tool with different responses
mock_tool = MockTool(responses={
    "query1": "result1",
    "query2": "result2"
})

# Mock tool with custom logic
def tool_logic(input_data: str) -> str:
    return f"Tool processed: {input_data.upper()}"

mock_tool = MockTool(response_func=tool_logic)
```

## Test Utilities

### Test Assertions

```python
from flujo.testing.assertions import assert_pipeline_success, assert_step_output

def test_pipeline_assertions():
    # Test pipeline success
    result = runner.run("test")
    assert_pipeline_success(result)
    
    # Test specific step output
    assert_step_output(result, 0, "expected output")
    
    # Test step count
    assert len(result.step_history) == 2
```

### Test Fixtures

```python
import pytest
from flujo.testing import MockAgent, MockBackend

@pytest.fixture
def mock_agents():
    """Provide mock agents for testing."""
    return {
        'review': MockAgent("Review checklist"),
        'solution': MockAgent("Generated solution"),
        'validator': MockAgent("Validation result")
    }

@pytest.fixture
def mock_backend():
    """Provide mock backend for testing."""
    return MockBackend()

@pytest.fixture
def sample_task():
    """Provide sample task for testing."""
    return Task(prompt="Test task", metadata={"test": True})
```

### Performance Testing

```python
import pytest
import time
from flujo.recipes import Default

@pytest.mark.benchmark
def test_pipeline_performance(benchmark):
    """Benchmark pipeline performance."""
    recipe = Default()
    task = Task(prompt="Simple task")
    
    def run_pipeline():
        return recipe.run_sync(task)
    
    result = benchmark(run_pipeline)
    assert result is not None
```

## Best Practices

### 1. Test Organization

- **Group related tests** in classes or modules
- **Use descriptive test names** that explain what is being tested
- **Keep tests focused** on a single behavior or component
- **Use fixtures** for common setup and teardown

### 2. Test Data Management

```python
# Use constants for test data
TEST_PROMPTS = [
    "Write a function",
    "Generate documentation",
    "Create a test"
]

# Use factories for complex objects
def create_test_task(prompt: str = "Test task") -> Task:
    return Task(
        prompt=prompt,
        metadata={"test": True, "timestamp": time.time()}
    )
```

### 3. Error Testing

```python
import pytest
from flujo.exceptions import FlujoError

def test_error_handling():
    """Test that errors are handled gracefully."""
    with pytest.raises(FlujoError):
        # Trigger an error condition
        problematic_operation()

def test_invalid_input():
    """Test behavior with invalid input."""
    with pytest.raises(ValueError):
        agent.run("")  # Empty input
```

### 4. Async Testing

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_operations():
    """Test async operations properly."""
    result = await async_operation()
    assert result is not None

@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test concurrent operations."""
    tasks = [async_operation() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
```

### 5. Test Isolation

```python
import pytest

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Setup
    yield
    # Teardown
    cleanup_global_state()

def test_isolated_operation():
    """Test that doesn't depend on global state."""
    # Test implementation
    pass
```

## Debugging Tests

### Verbose Output

```bash
# Run tests with verbose output
pytest -v tests/

# Run specific test with maximum verbosity
pytest -vvv tests/test_specific.py::test_function

# Show print statements
pytest -s tests/
```

### Test Debugging

```python
import pytest
import pdb

def test_with_debugging():
    """Test with debugging capabilities."""
    result = some_operation()
    
    if result is None:
        pdb.set_trace()  # Break into debugger
    
    assert result is not None
```

### Coverage Analysis

```bash
# Run tests with coverage
pytest --cov=flujo tests/

# Generate HTML coverage report
pytest --cov=flujo --cov-report=html tests/

# Show missing lines
pytest --cov=flujo --cov-report=term-missing tests/
```

### Test Profiling

```bash
# Profile test execution
pytest --durations=10 tests/

# Profile specific test
pytest --durations=0 tests/test_slow.py::test_slow_operation
```

This testing guide provides comprehensive coverage of testing strategies for Flujo. For more specific examples, see the test files in the `tests/` directory.
