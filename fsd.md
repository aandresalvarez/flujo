 

### **FSD-021 Addendum: Developer Debugging and Verification Workflow**

This section provides a standardized workflow for debugging the `flujo create` command. Following these steps will ensure a consistent environment for reproducing the bug and verifying the fix.

#### **Objective**

To provide a simple, repeatable process for any developer to:
1.  Set up a clean Flujo project.
2.  Run the Architect pipeline (`flujo create`) in debug mode.
3.  Observe the precise point of failure and the data flowing between steps.
4.  Verify that code changes have successfully fixed the issue.

#### **Prerequisites**

- A working Flujo development environment from this repo.
- The latest code from the feature branch containing the FSD-021 fixes.
- Local CLI installed via the project toolchain (see Step 0).

##### **Step 0: Install and use the local CLI**

Run from the repository root to ensure you’re running the dev version of the CLI.

```bash
# From repo root
make install              # sets up .venv via uv and syncs deps
uv run flujo --version    # sanity-check the CLI is wired to this repo
```

#### **Step-by-Step Debugging Procedure**

##### **Step 1: Create a Clean Test Directory**

Every debugging session should start in a fresh, isolated directory to avoid conflicts with other projects or cached state.

```bash
# Navigate to your development area
cd ~/dev

# Create and enter a new directory for this test run
mkdir flujo-create-debug
cd flujo-create-debug
```

##### **Step 2: Initialize a Flujo Project**

Initialize a standard Flujo project. This creates the necessary `flujo.toml`, `.flujo/` directory, and local state database, ensuring the CLI operates in a project-aware context.

```bash
# Initialize a new project.
flujo init

# If you are re-running the test in the same directory, use --force.
# flujo init --force
# > This directory already has Flujo project files... Re-initialize? [y/N]: y
```
This step is critical because it ensures the command uses the project-local state backend, which is essential for certain features and telemetry.

##### **Step 3: Run the Architect with a Test Goal and Debug Flag**

Now, execute the `flujo create` command with a clear, non-trivial goal. The `--debug` flag is essential as it will enable the verbose logging and `DEBUG` print statements we've added to the built-in functions.

Execute the following command from within your `flujo-create-debug` directory:

```bash
# Recommended: run CLI through uv to use the local repo environment
uv run flujo create --debug \
  --goal "Take the content from the Flujo GitHub README, get the first 500 characters, and save it to a file named readme_snippet.txt."
```
- **`--debug`**: Enables verbose CLI logging and ensures helpful diagnostics print clearly.
- **`--goal "..."`**: Provides a realistic, repeatable multi-step goal.

Optional non-interactive run (safer for repeatability and logs to a dedicated dir):

```bash
uv run flujo create --debug \
  --non-interactive \
  --output-dir ./output \
  --goal "Take the content from the Flujo GitHub README, get the first 500 characters, and save it to a file named readme_snippet.txt."
```
Notes:
- In `--non-interactive` mode, `--output-dir` is required.
- If the generated pipeline references side-effect skills, non-interactive runs will fail unless `--allow-side-effects` is provided (see Step 5).

##### **Step 4: Analyze the Output**

With the proposed fixes in place, you should observe the following in your terminal output:

1.  **Initial Pipeline Execution:** You will see the normal output from the pipeline's initial steps (DecomposeGoal, MapStepsToTools, etc.).

2.  **Crucial Debug Logs:** You should see the output from our hardened `extract_yaml_text` function. This is the key to verification.

    ```
    # You are looking for these lines specifically:
    DEBUG [extract_yaml_text]: Received type: <class 'pydantic_ai.agent.AgentResult'>
    DEBUG [extract_yaml_text]: Received value (first 200 chars): YamlWriter(generated_yaml='version: "0.1"\nname: github_readme_snippet\nsteps:\n- name: fetch_readme\n  agent:\n    id: flujo.builtins.http_get\n    params:\n      url: https://github.com/aandresalvarez/fl
    DEBUG [extract_yaml_text]: Successfully extracted YAML (first 100 chars): version: "0.1"
    name: github_readme_snippet
    steps:
    - name: fetch_readme
      agent:
        id: flujo.bu
    ```
    - **If you see this:** Extraction is working. The bug is fixed.
    - **If the "Successfully extracted YAML" line shows garbage or is empty:** The extraction logic needs refinement. Use the "Received value" dump to adjust the parsing shape.

3.  **Final Pipeline Output:** After the pipeline completes, you should see the confirmation that the final YAML file was written.

    ```
    [green]Wrote: /Users/yourname/dev/flujo-create-debug/pipeline.yaml[/green]
    ```

4.  **Verification of `pipeline.yaml`:**
    - Open the created `pipeline.yaml` in your editor.
    - Confirm steps reflect the goal (expected: `http_get`, processing, `fs_write_file`).
    - Optionally validate the spec:
      ```bash
      uv run flujo validate --file pipeline.yaml
      ```

##### **Step 5: Side-Effect Policy and Safety Flags**

- Non-interactive runs block pipelines that reference side-effect skills by default.
- If your generated YAML includes side-effect skills and you want to proceed in CI-like or scripted runs, add:
  ```bash
  --allow-side-effects
  ```
- Overwrite existing output safely when re-running:
  ```bash
  --force
  ```

##### **Step 6: Troubleshooting Quick Hits**

- No debug lines visible:
  - Ensure you used `--debug` and are running `uv run flujo ...` from the repo root after `make install`.
- Error: `--output-dir is required`:
  - Provide `--output-dir` in `--non-interactive` runs (by design).
- Warning: `Extracted text does not look like a valid Flujo YAML`:
  - Check the earlier "Received value" dump. If the YAML is embedded in a wrapper object or JSON string, update the extraction logic to target the right field.
- Architect blueprint not found error:
  - Ensure you’re running the CLI from this repo’s environment (`uv run flujo ...`) so bundled resources are discoverable.

##### **Step 7: Clean Resets Between Iterations**

When debugging changes across runs, ensure a clean slate:

```bash
# From the test directory (e.g., ~/dev/flujo-create-debug)
rm -rf .flujo/ output/ pipeline.yaml flujo.toml || true
uv run flujo init --force --yes
```

##### **Step 8: Test-Assisted Verification (Optional)**

For fast feedback on safety flows and output behavior, run the targeted tests:

```bash
# From repo root
make test-fast -k create
```

This hits scenarios such as:
- Blocking side-effects in `--non-interactive` mode unless `--allow-side-effects` is provided.
- Overwriting behavior via `--force`.

##### **Step 9: Capture Artifacts**

- The CLI writes the most recent exception to `output/last_run_error.txt` on failures.
- Save terminal logs that include the `DEBUG [extract_yaml_text]` lines alongside `pipeline.yaml` to document repros and fixes.

By following this workflow, any developer can quickly reproduce the issue, validate the behavior of `extract_yaml_text`, and confirm end-to-end success of the `flujo create` command using the local development CLI. It keeps runs deterministic, safe, and easy to iterate.
