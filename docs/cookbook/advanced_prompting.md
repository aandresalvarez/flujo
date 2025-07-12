# Cookbook: Advanced Prompt Formatting

Build dynamic prompts using the built-in `format_prompt` utility. It supports simple substitutions, conditional blocks, loops, nested data access and escaping.

## Basic Usage

```python
from flujo import format_prompt

result = format_prompt("Hello {{ name }}!", name="World")
# result -> "Hello World!"
```

## Conditional Blocks

Include text only when a variable is provided and truthy.

```python
template = "User query: {{ query }}. {{#if feedback}}Previous feedback: {{ feedback }}{{/if}}"
print(format_prompt(template, query="a", feedback="It was wrong."))
# User query: a. Previous feedback: It was wrong.

print(format_prompt(template, query="a", feedback=None))
# User query: a.
```

## Iterating Over Lists

```python
template = "Consider:\n{{#each examples}}- {{ this }}\n{{/each}}"
print(format_prompt(template, examples=["A", "B"]))
# Consider:
# - A
# - B
```

## Nested Placeholders

```python
template = "User: {{ user.name }} ({{ user.email }})"
user = {"name": "Bob", "email": "b@example.com"}
print(format_prompt(template, user=user))
# User: Bob (b@example.com)
```

## Escaping

Use a backslash before the opening braces to emit literal braces.

```python
template = r"The syntax is \{{ variable_name }}."
print(format_prompt(template))
# The syntax is {{ variable_name }}.
```
