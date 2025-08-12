# Tools Guide

This guide explains how to use and create tools that allow AI agents to interact with external systems in `flujo`.

## Overview

Tools enable agents to:

- Fetch data from APIs
- Execute code
- Interact with databases
- Call external services
- Access local resources

## Basic Usage

### Creating a Tool

```python
from pydantic_ai import Tool
from typing import Optional

def get_weather(city: str, country: Optional[str] = None) -> str:
    """Get current weather for a city.

    Args:
        city: The city name
        country: Optional country code

    Returns:
        A string describing the weather
    """
    # Implementation here
    return f"Weather in {city}: Sunny"

# Create a tool
weather_tool = Tool(get_weather)

# Give it to an agent
agent = make_agent_async(
    "openai:gpt-4",
    "You are a weather assistant.",
    str,
    tools=[weather_tool]
)
```

### Using Tools in a Pipeline

```python
from flujo import Step, Flujo

# Create a pipeline with tools
pipeline = (
    Step.review(make_review_agent())
    >> Step.solution(
        make_solution_agent(),
        tools=[weather_tool]
    )
    >> Step.validate(make_validator_agent())
)

# Run it
runner = Flujo(pipeline)
result = runner.run("What's the weather in Paris?")
```

## Tool Types

### 1. Function Tools

The simplest type of tool is a function:

```python
def calculate_total(items: list[float], tax_rate: float = 0.1) -> float:
    """Calculate total with tax.

    Args:
        items: List of prices
        tax_rate: Tax rate (default: 0.1)

    Returns:
        Total price including tax
    """
    subtotal = sum(items)
    tax = subtotal * tax_rate
    return subtotal + tax

# Create the tool
calculator_tool = Tool(calculate_total)
```

### 2. Class Tools

For more complex tools, use a class:

```python
from pydantic import BaseModel

class DatabaseTool:
    """Tool for database operations."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def query(self, sql: str) -> list[dict]:
        """Execute a SQL query.

        Args:
            sql: SQL query string

        Returns:
            List of result rows
        """
        # Implementation here
        return [{"id": 1, "name": "example"}]

    def insert(self, table: str, data: dict) -> int:
        """Insert a row into a table.

        Args:
            table: Table name
            data: Row data

        Returns:
            Inserted row ID
        """
        # Implementation here
        return 1

# Create the tool
db_tool = Tool(DatabaseTool("postgresql://..."))
```

### 3. Async Tools

For I/O-bound operations, use async tools:

```python
import aiohttp
from typing import Optional

async def fetch_data(url: str, timeout: Optional[int] = 30) -> dict:
    """Fetch data from a URL.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Response data as dict
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            return await response.json()

# Create the tool
fetch_tool = Tool(fetch_data)
```

## Tool Configuration

### Basic Configuration

```python
# Create a tool with configuration
tool = Tool(
    my_function,
    timeout=10,  # Tool timeout
    retries=2,   # Number of retries
    backoff_factor=1.5  # Backoff between retries
)
```

### Advanced Configuration

```python
from pydantic_ai import Tool, ToolConfig

# Create a tool with advanced configuration
config = ToolConfig(
    timeout=10,
    retries=2,
    backoff_factor=1.5,
    rate_limit=100,  # Calls per minute
    cache_ttl=300,   # Cache for 5 minutes
    validate_input=True,  # Validate input types
    validate_output=True  # Validate output type
)

tool = Tool(my_function, config=config)
```

## Best Practices

### 1. Error Handling

```python
from typing import Optional
from pydantic_ai import Tool, ToolError

def safe_api_call(endpoint: str, timeout: Optional[int] = None) -> dict:
    """Make a safe API call with error handling."""
    try:
        # Implementation here
        return {"status": "success"}
    except Exception as e:
        raise ToolError(f"API call failed: {str(e)}")

# Create the tool
api_tool = Tool(safe_api_call)
```

### 2. Input Validation

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=100)

def search_database(params: SearchParams) -> list[dict]:
    """Search database with validated parameters."""
    # Implementation here
    return [{"id": 1, "title": "example"}]

# Create the tool
search_tool = Tool(search_database)
```

### 3. Caching

```python
from functools import lru_cache
from pydantic_ai import Tool

@lru_cache(maxsize=100)
def get_cached_data(key: str) -> dict:
    """Get data with caching."""
    # Implementation here
    return {"key": key, "value": "cached"}

# Create the tool
cache_tool = Tool(get_cached_data)
```

### 4. Rate Limiting

```python
from pydantic_ai import Tool, ToolConfig
import time

class RateLimitedTool:
    """Tool with rate limiting."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.last_call = 0

    def call(self, param: str) -> str:
        """Make a rate-limited call."""
        # Implement rate limiting
        now = time.time()
        if now - self.last_call < 60 / self.calls_per_minute:
            time.sleep(60 / self.calls_per_minute)
        self.last_call = time.time()

        # Implementation here
        return f"Processed: {param}"

# Create the tool
rate_tool = Tool(
    RateLimitedTool(calls_per_minute=100).call,
    config=ToolConfig(rate_limit=100)
)
```

## Examples

### API Integration

```python
import aiohttp
from pydantic import BaseModel
from pydantic_ai import Tool

class WeatherParams(BaseModel):
    city: str
    country: str = "US"
    units: str = "metric"

async def get_weather(params: WeatherParams) -> dict:
    """Get weather from OpenWeatherMap API."""
    async with aiohttp.ClientSession() as session:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": f"{params.city},{params.country}",
            "units": params.units,
            "appid": os.getenv("OPENWEATHER_API_KEY")
        }
        async with session.get(url, params=params) as response:
            return await response.json()

# Create the tool
weather_tool = Tool(
    get_weather,
    config=ToolConfig(
        timeout=10,
        retries=2,
        cache_ttl=300  # Cache for 5 minutes
    )
)
```

### Database Operations

```python
from sqlalchemy import create_engine, text
from pydantic_ai import Tool

class DatabaseTool:
    """Tool for database operations."""

    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)

    def query(self, sql: str, params: dict = None) -> list[dict]:
        """Execute a SQL query."""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return [dict(row) for row in result]

    def insert(self, table: str, data: dict) -> int:
        """Insert a row into a table."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text(f"INSERT INTO {table} VALUES (:data)"),
                {"data": data}
            )
            return result.lastrowid

# Create the tool
db_tool = Tool(
    DatabaseTool("postgresql://...").query,
    config=ToolConfig(
        timeout=30,
        retries=3,
        validate_input=True
    )
)
```

### File Operations

```python
from pathlib import Path
from pydantic_ai import Tool

def read_file(path: str, encoding: str = "utf-8") -> str:
    """Read a file safely."""
    try:
        return Path(path).read_text(encoding=encoding)
    except Exception as e:
        raise ToolError(f"Failed to read file: {str(e)}")

def write_file(path: str, content: str, encoding: str = "utf-8") -> None:
    """Write to a file safely."""
    try:
        Path(path).write_text(content, encoding=encoding)
    except Exception as e:
        raise ToolError(f"Failed to write file: {str(e)}")

# Create the tools
read_tool = Tool(read_file)
write_tool = Tool(write_file)
```

## Troubleshooting

### Common Issues

1. **Tool Errors**
   - Check error messages
   - Verify input types
   - Test tool directly
   - Review configuration

2. **Performance Issues**
   - Implement caching
   - Use rate limiting
   - Optimize timeouts
   - Monitor usage

3. **Security Issues**
   - Validate inputs
   - Sanitize outputs
   - Use secure connections
   - Implement access control

### Getting Help

- Check the [Troubleshooting Guide](troubleshooting.md)
- Search [existing issues](https://github.com/aandresalvarez/flujo/issues)
- Create a new issue if needed

## Next Steps

- Read the [Usage Guide](../user_guide/usage.md)
- Explore [Advanced Topics](extending.md)
- Check out [Use Cases](../user_guide/use_cases.md)
