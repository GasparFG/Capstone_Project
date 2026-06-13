import json, subprocess, sys
from pathlib import Path

manifest = json.loads(Path("data/robustness/test_manifest.json").read_text())
output_root = Path("outputs/baseline_only")

for case in manifest:
    case_outdir = output_root / case["case_name"]
    case_outdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "baseline_round_robin.py",
        "--jobs-json-input",   case["jobs_file"],
        "--server-json-input", case["server_file"],
        "--output-root",       str(case_outdir),
    ])