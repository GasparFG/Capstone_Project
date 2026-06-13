"""
run_robustness_suite.py
=======================
Iterates over every case in test_manifest.json and runs the
optimization pipeline for each one, writing results to a
timestamped output folder.

Usage:
    python run_robustness_suite.py \
        --manifest data/robustness_tests/test_manifest.json \
        --output-root outputs/robustness \
        --time-limit 300 \
        --mip-gap 0.05
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",    required=True)
    parser.add_argument("--output-root", default="outputs/robustness")
    parser.add_argument("--time-limit",  type=int,   default=300)
    parser.add_argument("--mip-gap",     type=float, default=0.05)
    parser.add_argument("--start-from",  type=int,   default=0,
                        help="Skip the first N cases (resume after crash).")
    args = parser.parse_args()

    manifest   = json.loads(Path(args.manifest).read_text())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_log_path = output_root / "run_log.jsonl"
    total  = len(manifest)
    passed = failed = 0

    print(f"Starting robustness suite: {total} cases, "
          f"time_limit={args.time_limit}s, mip_gap={args.mip_gap}")
    print(f"Skipping first {args.start_from} cases.\n")

    for idx, case in enumerate(manifest):
        if idx < args.start_from:
            continue

        name        = case["case_name"]
        case_outdir = output_root
        case_outdir.mkdir(parents=True, exist_ok=True)

        print(f"[{idx+1:>3}/{total}] {name} ... ", end="", flush=True)
        t0 = time.time()

        cmd = [
            sys.executable, "-m", "src.optimization.cli",   # adjust to your module name
            "--jobs-json-input",   case["jobs_file"],
            "--server-json-input", case["server_file"],
            "--output-root",       str(case_outdir),
            "--time-limit",        str(args.time_limit),
            "--mip-gap",           str(args.mip_gap),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=args.time_limit + 60,   # hard wall = solver limit + 60s
            )
            elapsed = round(time.time() - t0, 1)
            ok = result.returncode == 0
            status = "PASS" if ok else "FAIL"
            (passed if ok else failed).__class__   # just increment below
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"{status}  ({elapsed}s)")
            if not ok:
                print(f"       stderr: {result.stderr[-300:]}")

        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - t0, 1)
            failed += 1
            status = "TIMEOUT"
            result = None
            print(f"TIMEOUT  ({elapsed}s)")

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            failed += 1
            status = "ERROR"
            result = None
            print(f"ERROR  {e}")

        # Append one line to the run log regardless of outcome
        log_entry = {
            "idx":        idx,
            "case_name":  name,
            "status":     status,
            "elapsed_s":  elapsed,
            "n_servers":  case["n_servers"],
            "n_jobs":     case["n_jobs"],
            "k_slots":    case["k_slots"],
            "psi_stage":  case["psi_stage"],
            "timestamp":  datetime.utcnow().isoformat(),
            "stdout_tail": result.stdout[-500:] if result else "",
            "stderr_tail": result.stderr[-500:] if result else "",
        }
        with open(run_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\nDone.  {passed} passed / {failed} failed / {total} total")
    print(f"Run log: {run_log_path}")


if __name__ == "__main__":
    main()