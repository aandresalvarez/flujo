Test Skip and Opt-In Policy

This repository intentionally skips a small set of tests in default local/CI runs. The goals are to keep the core test loop fast and stable across platforms while making it easy to opt into heavier or environment‑specific coverage when needed.

Categories

- Deprecated governor tests:
  - Background: The old UsageGovernor has been removed in favor of proactive quota reservations (pure quota mode).
  - Action: Legacy tests that targeted UsageGovernor are replaced with quota‑mode assertions. Any remaining governor‑specific skips are safe to ignore and can be removed in future cleanup.

- Platform‑specific tests:
  - Some tests are Windows‑only or Unix‑only and are skipped on other platforms to avoid false failures.

- Pending redesign / expected failures:
  - A few integration tests around fallback chain behavior are marked xfail, tracked for redesign. They document known gaps without breaking the build.

- E2E and benchmarks (opt‑in):
  - E2E tests and performance benchmarks are disabled by default. These are intended for special pipelines or workflows and can be enabled explicitly.

How to enable opt‑in suites

- E2E golden transcript:
  - Set `CI_E2E_RUN=1` to include tests under `tests/e2e/`.
  - Some tests require `vcrpy` and real API keys; follow comments in the test files.

- Benchmarks:
  - Performance tests are placeholders or long‑running by nature. Enable and run them manually as needed.

Notes

- If you see “UsageGovernor removed in pure quota mode” in a skip message, it’s expected — the new quota model is covered by quota‑mode tests.
- If you need stricter coverage in CI, consider flipping specific xfails to skips with links to tracking issues, or gate them behind environment flags.

