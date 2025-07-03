# Cookbook: Redacting Secrets

`flujo` provides utilities for redacting sensitive information, such as API keys and passwords, from text. This is useful for preventing secrets from being accidentally logged or exposed.

## Redacting Strings

The `redact_string` function can be used to replace occurrences of a secret string with a `[REDACTED]` placeholder.

```python
from flujo.utils.redact import redact_string

my_secret = "my-super-secret-api-key"  # pragma: allowlist secret
text = f"Using the API key: {my_secret}"

redacted_text = redact_string(text, my_secret)

print(redacted_text)
# Using the API key: [REDACTED]
```

The `redact_string` function will also redact any string that starts with the first 8 characters of the secret, followed by at least 5 more plausible characters. This helps to prevent partial secrets from being exposed.

## Redacting Passwords in URLs

The `redact_url_password` function can be used to redact the password from a URL.

```python
from flujo.utils.redact import redact_url_password

url = "https://user:password@example.com"  # pragma: allowlist secret

redacted_url = redact_url_password(url)

print(redacted_url)
# https://[REDACTED]@example.com
```

## Summarizing and Redacting Prompts

The `summarize_and_redact_prompt` function can be used to create a truncated and redacted version of a prompt. This is useful for logging or displaying prompts without exposing any sensitive information.

```python
from flujo.utils.redact import summarize_and_redact_prompt

prompt = "Please use the following API key to access the service: my-super-secret-api-key" # pragma: allowlist secret

redacted_prompt = summarize_and_redact_prompt(prompt)

print(redacted_prompt)
# Please use the following API key to access the service: [REDACTED]
```

The `summarize_and_redact_prompt` function will automatically redact any API keys that are configured in the `flujo` settings.
