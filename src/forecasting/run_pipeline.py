"""
run_pipeline.py
===============
Full end-to-end Capstone pipeline runner.

Steps:
    1. Ensemble Forecast     — src/forecasting/ensemble_model.py
    2. Forecast → JSON       — src/optimization/build_jobs_json_from_forecast.py
    3. Optimization Solver   — src/optimization/solver.py

Run from the project root:
    python run_pipeline.py

Optional flags:
    python run_pipeline.py --skip-forecast   # skip Step 1 (reuse existing forecast)
    python run_pipeline.py --skip-solver     # skip Step 3
    python run_pipeline.py --forecast-only   # run Steps 1-2 only
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

STEPS = [
    {
        "name":   "Ensemble Forecast (Scenario B)",
        "script": ROOT / "src" / "forecasting"  / "ensemble_model.py",
        "flag":   "--skip-forecast",
    },
    {
        "name":   "Forecast → Optimizer JSON",
        "script": ROOT / "src" / "optimization" / "build_jobs_json_from_forecast.py",
        "flag":   None,
    },
    {
        "name":   "Optimization Solver",
        "script": ROOT / "src" / "optimization" / "solver.py",
        "flag":   "--skip-solver",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(text: str, width: int = 60) -> None:
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_step(name: str, script: Path) -> bool:
    if not script.exists():
        print(f"  [SKIP] {script} not found — skipping.")
        return True

    print(f"\n  Running: python {script.relative_to(ROOT)}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAIL] {name} exited with code {result.returncode}.")
        return False

    print(f"\n  [OK] {name} completed in {elapsed:.1f}s")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = set(sys.argv[1:])
    forecast_only = "--forecast-only" in args

    banner("CAPSTONE PIPELINE — full run")
    print("  Project root:", ROOT)

    results = {}
    for i, step in enumerate(STEPS, 1):
        # Apply flag skips
        if step["flag"] and step["flag"] in args:
            print(f"\n  [SKIP] Step {i}: {step['name']} (flag: {step['flag']})")
            results[step["name"]] = "skipped"
            continue

        # --forecast-only: stop after step 2
        if forecast_only and i == 3:
            print(f"\n  [SKIP] Step {i}: {step['name']} (--forecast-only)")
            results[step["name"]] = "skipped"
            continue

        banner(f"Step {i}/{len(STEPS)}: {step['name']}")
        ok = run_step(step["name"], step["script"])
        results[step["name"]] = "ok" if ok else "FAILED"

        if not ok:
            print("\n  Pipeline halted. Fix the error above and re-run.")
            _print_summary(results)
            sys.exit(1)

    _print_summary(results)


def _print_summary(results: dict) -> None:
    banner("PIPELINE SUMMARY")
    for name, status in results.items():
        icon = "✓" if status == "ok" else ("–" if status == "skipped" else "✗")
        print(f"  {icon}  {name:<45}  {status}")
    print()


if __name__ == "__main__":
    main()
