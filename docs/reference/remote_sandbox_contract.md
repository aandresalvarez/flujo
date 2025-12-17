---
title: Remote Sandbox Contract
---

# Remote Sandbox Contract

This document specifies the HTTP contract expected by Flujo’s `RemoteSandbox` implementation (`flujo/infra/sandbox/remote_sandbox.py`).

## Endpoint

- **Method**: `POST`
- **Path**: `/execute` (default)
- **Content-Type**: `application/json`
- **Auth (optional)**: `Authorization: Bearer <api_key>`

The base URL is configured via `FLUJO_SANDBOX_API_URL` (or TOML `settings.sandbox.api_url`). The default path is `/execute`.

## Request JSON

```json
{
  "code": "print('hello')",
  "language": "python",
  "files": { "requirements.txt": "requests==2.32.3\n" },
  "environment": { "FOO": "bar" },
  "arguments": ["--flag", "value"],
  "timeout_s": 60.0
}
```

Fields:
- `code` (string, required): Source code to execute.
- `language` (string, required): Language identifier (e.g., `"python"`).
- `files` (object, optional): Mapping of relative file path → file content (UTF-8 text).
- `environment` (object, optional): Mapping of env var name → value.
- `arguments` (array of strings, optional): CLI args passed to the program entrypoint by the provider.
- `timeout_s` (number, optional): Execution timeout in seconds (provider should enforce it).

## Response JSON

```json
{
  "stdout": "hello\n",
  "stderr": "",
  "exit_code": 0,
  "timed_out": false,
  "error": null,
  "sandbox_id": "abc123",
  "artifacts": {
    "report.json": { "base64": "eyJvayI6dHJ1ZX0=" }
  }
}
```

Fields:
- `stdout` (string): Captured stdout.
- `stderr` (string): Captured stderr.
- `exit_code` (integer): Process exit code.
- `timed_out` (boolean): Whether the provider timed out the execution.
- `error` (string|null): Human-readable error message (if any).
- `sandbox_id` (string|null): Provider run/container identifier (optional).
- `artifacts` (object|null): Optional mapping of artifact name → artifact value.

### Artifacts encoding

Flujo accepts any of the following per artifact entry:
- **Base64 string**: `"name": "<base64>"` (decoded as base64; if decode fails, treated as UTF-8 text)
- **Object with base64**: `"name": {"base64": "<base64>"}`
- **Object with inline data**: `"name": {"data": "<utf8 text>"}`
- **Object with URL**: `"name": {"url": "https://.../artifact.bin"}` (Flujo fetches the URL and stores the response bytes)

## HTTP status handling

- Flujo will attempt to parse JSON regardless of status.
- If `status_code >= 400` and the response does not include an `error`, Flujo sets `error` to `"Remote sandbox HTTP <status>"`.

## Timeouts and TLS

- Client timeout defaults to `FLUJO_SANDBOX_TIMEOUT_S` (or TOML `settings.sandbox.timeout_seconds`).
- TLS verification is controlled by `FLUJO_SANDBOX_VERIFY_SSL` (or TOML `settings.sandbox.verify_ssl`).

