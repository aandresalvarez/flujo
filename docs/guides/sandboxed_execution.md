# Sandboxed Execution (Code Interpreter)

Flujo includes a built-in skill `flujo.builtins.code_interpreter` to run generated code in an isolated sandbox implementation.

## Configure the Sandbox

### Remote sandbox (recommended for production)

```toml
[settings.sandbox]
mode = "remote"
api_url = "https://your-sandbox.example"
timeout_seconds = 60
verify_ssl = true
```

### Docker sandbox (local development)

Install the Docker SDK extra:

```bash
pip install "flujo[docker]"
```

```toml
[settings.sandbox]
mode = "docker"
docker_image = "python:3.13-slim"
docker_pull = true
docker_mem_limit = "512m"
docker_pids_limit = 256
docker_network_mode = "none"
```

Or configure via environment variables:

```bash
export FLUJO_SANDBOX_MODE=remote
export FLUJO_SANDBOX_API_URL=https://your-sandbox.example
export FLUJO_SANDBOX_TIMEOUT_S=60
```

## Using the Skill

When the sandbox is configured, pipelines/agents can call the tool by id:

- `flujo.builtins.code_interpreter`

It returns a JSON object containing `stdout`, `stderr`, `exit_code`, `timed_out`, `error`, and `artifacts` (when supported).

## Notes / Limitations

- Docker sandbox currently focuses on `language="python"`.
- Remote sandbox expects an HTTP API compatible with the `RemoteSandbox` request shape (code + language + files + env + args).

## Remote Sandbox Contract

See `docs/reference/remote_sandbox_contract.md` for the provider-facing request/response schema (including artifact handling).
