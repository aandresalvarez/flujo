"""
04_batch_processing.py
----------------------
Reads a CSV file with one prompt per row, runs the orchestrator for each,
and writes results to `results.csv`.

CSV schema:
    prompt,text
"""

import csv
import pathlib
import time
from flujo import (
    Orchestrator,
    Task,
    review_agent,
    solution_agent,
    validator_agent,
    reflection_agent,
)

INPUT = pathlib.Path("prompts.csv")
OUTPUT = pathlib.Path("results.csv")

# Create orchestrator with the required agents
orch = Orchestrator(review_agent, solution_agent, validator_agent, reflection_agent)

with INPUT.open() as f_in, OUTPUT.open("w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = ["prompt", "score", "solution"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        prompt = row["prompt"]
        start = time.perf_counter()
        candidate = orch.run_sync(Task(prompt=prompt))
        dur = time.perf_counter() - start
        print(f"✓ {prompt[:40]:40} | score={candidate.score:.2f} | {dur:.1f}s")
        writer.writerow(
            {
                "prompt": prompt,
                "score": candidate.score,
                "solution": candidate.solution.replace("\n", " "),
            }
        )

print(f"\nSaved results ➜ {OUTPUT}")
