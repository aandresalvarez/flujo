# SARIF Output for CI/CD

`flujo validate --format sarif` emits SARIF 2.1.0 compatible JSON that can be uploaded to code scanning tools.

Example:

```
uv run flujo validate pipeline.yaml --format sarif > findings.sarif
```

Basic structure includes:
- `runs[].tool.driver.rules`: rule metadata with ids (e.g., V-T1) and help URIs.
- `runs[].results[]`: results with `ruleId`, `level` (error|warning), `message`, and `locations` (filename, line, column when available).

GitHub Actions usage:
- Use an action to upload SARIF, e.g., `github/codeql-action/upload-sarif@v3`.

```
- name: Validate YAML
  run: uv run flujo validate pipeline.yaml --format sarif > findings.sarif

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: findings.sarif
```

