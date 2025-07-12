# Cookbook: Advanced Prompt Formatting

`flujo` provides an `AdvancedPromptFormatter` that allows you to create dynamic and flexible prompts using conditionals, loops, and nested data. This is particularly useful for generating complex prompts for language models.

## Basic Usage

You can use placeholders in your prompt templates, which will be replaced with the corresponding values from the `kwargs` you provide.

```python
from flujo.utils.prompting import format_prompt

template = "Hello, {{ name }}! Today is {{ day_of_week }}."
formatted_prompt = format_prompt(template, name="Alice", day_of_week="Thursday")

print(formatted_prompt)
# Hello, Alice! Today is Thursday.
```

## Conditionals (`{{#if ...}}...{{/if}}`)

You can include conditional blocks in your templates using `{{#if <condition>}}...{{/if}}`. The content inside the block will only be included if the condition evaluates to a truthy value.

```python
from flujo.utils.prompting import format_prompt

template = "{{#if is_admin}}Welcome, admin!{{/if}} Hello, {{ name }}."
formatted_prompt = format_prompt(template, name="Bob", is_admin=True)
print(formatted_prompt)
# Welcome, admin! Hello, Bob.

formatted_prompt = format_prompt(template, name="Charlie", is_admin=False)
print(formatted_prompt)
#  Hello, Charlie.
```

## Loops (`{{#each ...}}...{{/each}}`)

You can iterate over lists using `{{#each <list_name>}}...{{/each}}`. Inside the loop, `{{ this }}` refers to the current item in the list.

```python
from flujo.utils.prompting import format_prompt

template = "Items:\n{{#each items}}- {{ this }}\n{{/each}}"
formatted_prompt = format_prompt(template, items=["apple", "banana", "cherry"])

print(formatted_prompt)
# Items:
# - apple
# - banana
# - cherry
```

### Nested Loops and Objects

You can also use nested data structures and access their properties using dot notation.

```python
from flujo.utils.prompting import format_prompt

template = "Users:\n{{#each users}}- Name: {{ this.name }}, Age: {{ this.age }}\n{{/each}}"
users_data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 24},
]
formatted_prompt = format_prompt(template, users=users_data)

print(formatted_prompt)
# Users:
# - Name: Alice, Age: 30
# - Name: Bob, Age: 24
```

## Escaping Placeholders

If you need to include literal `{{` or `}}` in your template without them being interpreted as placeholders, you can escape the opening curly brace with a backslash: `\{{`.

```python
from flujo.utils.prompting import format_prompt

template = "This is a literal curly brace: \{{\nformatted_prompt = format_prompt(template)

print(formatted_prompt)
# This is a literal curly brace: {{
```

```
