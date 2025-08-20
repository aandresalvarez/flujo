# Externalized Prompt Management in Flujo YAML Blueprints

This example demonstrates how to use Flujo's externalized prompt management feature, which allows you to store system prompts in separate files instead of embedding them directly in your YAML pipeline definitions.

## Overview

The externalized prompt feature addresses several challenges with large, embedded prompts:
- **Poor readability** in YAML files
- **Difficult maintenance** and editing
- **Poor version control** diffing
- **Lack of reusability** across different pipelines

## How It Works

Instead of writing:
```yaml
agents:
  my_agent:
    model: "openai:gpt-4o"
    system_prompt: |
      This is a very long and complex prompt that spans
      many lines. Editing it here is cumbersome, and reviewing
      changes in a git diff is difficult.
    output_schema: { ... }
```

You can now write:
```yaml
agents:
  my_agent:
    model: "openai:gpt-4o"
    system_prompt:
      from_file: "./prompts/my_agent_prompt.md"
    output_schema: { ... }
```

## File Structure

```
examples/
├── externalized_prompt_demo.yaml    # Main pipeline definition
├── prompts/                         # Directory containing prompt files
│   ├── code_reviewer.md            # Code reviewer agent prompt
│   └── documentation_writer.md     # Documentation writer agent prompt
└── README_externalized_prompts.md  # This file
```

## Benefits

1. **Better Organization**: Prompts are stored in dedicated files with proper extensions
2. **Easier Editing**: Use your preferred text editor with syntax highlighting
3. **Better Version Control**: Git diffs show actual content changes, not just line numbers
4. **Reusability**: Share prompts across multiple pipelines
5. **Maintainability**: Update prompts without touching pipeline logic

## Security Features

- **Path Traversal Protection**: The system prevents accessing files outside the project directory
- **Sandboxed Loading**: All prompt files must be within the same directory as the pipeline YAML
- **Clear Error Messages**: Descriptive errors for missing files or security violations

## Usage Examples

### Basic Externalized Prompt
```yaml
system_prompt:
  from_file: "./prompts/simple_prompt.txt"
```

### Relative Path Resolution
```yaml
system_prompt:
  from_file: "./shared_prompts/agent_prompt.md"
```

Note: Paths are sandboxed to the blueprint's base_dir. If you need to reference a parent directory (e.g., `../shared_prompts/...`), ensure the compiler/loader is invoked with base_dir set to the project root so that the resolved absolute path still falls within the sandbox. Otherwise, the os.path.commonpath check will block traversal outside base_dir.

### Nested Directory Structure
```yaml
system_prompt:
  from_file: "./agents/prompts/specialized_agent.md"
```

## Error Handling

The system provides clear error messages for common issues:

- **File Not Found**: `Prompt file not found for agent 'agent_name': /path/to/file`
- **Path Traversal**: `Path traversal detected in from_file: ../secret.txt`
- **File Read Errors**: `Error reading prompt file '/path/to/file': [error details]`

## Best Practices

1. **Use Descriptive Filenames**: `code_reviewer.md` instead of `prompt1.md`
2. **Organize by Function**: Group related prompts in subdirectories
3. **Use Appropriate Extensions**: `.md` for markdown, `.txt` for plain text
4. **Keep Prompts Focused**: Each prompt file should serve a single, clear purpose
5. **Version Control**: Commit prompt files alongside pipeline definitions

## Testing

You can test this feature by running:

```bash
# Test the example pipeline
cd examples
python3 -c "
from flujo.domain.blueprint import load_pipeline_blueprint_from_yaml
with open('externalized_prompt_demo.yaml') as f:
    yaml_text = f.read()
pipeline = load_pipeline_blueprint_from_yaml(yaml_text, base_dir='.')
print(f'✅ Pipeline loaded with {len(pipeline.steps)} steps')
"
```

## Backward Compatibility

This feature is fully backward compatible. Existing YAML files with embedded `system_prompt` strings will continue to work exactly as before.

## Technical Details

- **File Encoding**: All prompt files are read as UTF-8
- **Path Resolution**: Relative paths are resolved from the pipeline YAML file location
- **Security**: Path traversal attempts are blocked using `os.path.commonpath` validation
- **Performance**: Prompt files are read once during compilation and cached
