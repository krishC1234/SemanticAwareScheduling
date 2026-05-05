#!/usr/bin/env python3
"""Size-aware baseline — assign GPUs based on model parameter count.

GPU assignment tiers:
  250M+ params  → 8 GPUs
  30-250M params → 4 GPUs
  1-30M params  → 2 GPUs
  0-1M params   → 1 GPU

Param count is extracted from the script's docstring.

Usage:
    python3 -m evaluation.test_scripts.size_aware_baseline
    python3 -m evaluation.test_scripts.size_aware_baseline --seed 42 --max-delay 60
"""

import argparse
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from evaluation.metrics import MetricsCollector, parse_job_runtime
from evaluation.report import report

EVAL_JOBS_DIR = Path(__file__).parent.parent / "jobs"
POLL_INTERVAL = 10

LOGS_DIR = EVAL_JOBS_DIR.parent.parent / "logs" / "size_aware_output"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def extract_param_count(script_path):
    """Extract approximate param count from script docstring (e.g. '~25.6M params')."""
    text = script_path.read_text()
    match = re.search(r"~([\d.]+)([MBK])\s*params", text, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    if unit == "B":
        return value * 1e9
    elif unit == "M":
        return value * 1e6
    elif unit == "K":
        return value * 1e3
    return value


def assign_gpus(param_count):
    """Assign GPUs based on param count tiers."""
    if param_count is None:
        return 4  # default
    if param_count >= 250e6:
        return 8
    elif param_count >= 30e6:
        return 4
    elif param_count >= 1e6:
        return 2
    else:
        return 1


def sbatch_submit(script_path, gpu_count):
    """Submit a DDP script to SLURM via sbatch + torchrun."""
    log_file = LOGS_DIR / f"{script_path.stem}_{gpu_count}gpu.log"
    sbatch_script = (
        "#!/bin/bash\n"
        f"#SBATCH --gres=gpu:{gpu_count}\n"
        f"#SBATCH --job-name={script_path.stem}\n"
        f"#SBATCH --output={log_file}\n"
        f"#SBATCH --error={log_file}\n"
        f"torchrun --standalone --nproc_per_node={gpu_count} {script_path.resolve()}\n"
    )
    try:
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=sbatch_script,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip(), time.time()
        else:
            print(f"sbatch error: {result.stderr.strip()}")
            return None, None
    except Exception as e:
        print(f"sbatch exception: {e}")
        return None, None


def get_active_job_ids():
    """Return set of SLURM job IDs that are RUNNING or PENDING."""
    try:
        out = subprocess.run(
            ["squeue", "--Format=jobid", "--noheader",
             "--states=RUNNING,PENDING"],
            capture_output=True, text=True, timeout=10,
        )
        return {line.strip() for line in out.stdout.strip().splitlines() if line.strip()}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Size-aware baseline: GPUs assigned by param count tiers")
    parser.add_argument("--max-delay", type=float, default=600,
                        help="Maximum delay between submissions (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Shared output directory for this run")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scripts = sorted(EVAL_JOBS_DIR.glob("*.py"))
    if not scripts:
        print(f"No .py files found in {EVAL_JOBS_DIR}")
        sys.exit(1)

    rng.shuffle(scripts)
    total_jobs = len(scripts)

    # Pre-compute GPU assignments
    assignments = {}
    for s in scripts:
        pc = extract_param_count(s)
        gpus = assign_gpus(pc)
        assignments[s] = gpus
        pc_str = f"{pc/1e6:.1f}M" if pc else "unknown"
        print(f"  {s.stem:30s}  {pc_str:>10s}  → {gpus} GPU(s)")

    print(f"\n=== Size-aware baseline: {total_jobs} jobs, "
          f"delays 0-{args.max_delay}s ===\n")

    collector = MetricsCollector(interval=2)
    collector.pending = total_jobs
    collector.start()

    tracked = {}

    for i, script in enumerate(scripts):
        gpus = assignments[script]
        enqueue_time = time.time()
        collector.record_submission(enqueue_time)
        print(f"[{i+1}/{total_jobs}] Submitting {script.stem}...", end=" ")
        slurm_id, submit_time = sbatch_submit(script, gpus)
        if slurm_id:
            log_file = LOGS_DIR / f"{script.stem}_{gpus}gpu.log"
            tracked[slurm_id] = {
                "name": script.stem,
                "gpus": gpus,
                "submit_time": submit_time,
                "enqueue_time": enqueue_time,
                "log_file": log_file,
            }
            print(f"SLURM job {slurm_id} ({gpus} GPUs)")
        else:
            print("failed")
            collector.pending -= 1

        if i < total_jobs - 1:
            delay = rng.uniform(0, args.max_delay)
            print(f"         waiting {delay:.1f}s before next submission...")
            time.sleep(delay)

    print(f"\n=== All jobs submitted, polling for completions ===\n")

    seen = set()
    while collector.pending > 0:
        active = get_active_job_ids()

        for slurm_id, info in list(tracked.items()):
            if slurm_id not in active and slurm_id not in seen:
                completion_time = time.time()
                run_time = parse_job_runtime(info["log_file"])
                if run_time is None:
                    if Path(info["log_file"]).exists():
                        print(f"  FAILED: {info['name']} (crashed, no results)")
                        seen.add(slurm_id)
                        collector.pending -= 1
                    continue
                wait_time = completion_time - info["enqueue_time"] - run_time
                collector.record_job(
                    name=info["name"],
                    gpus=info["gpus"],
                    run_time=run_time,
                    wait_time=max(0, wait_time),
                )
                seen.add(slurm_id)

        print(f"  [{collector.pending} pending, "
              f"{len(active & set(tracked.keys()))} running]")

        if collector.pending > 0:
            time.sleep(POLL_INTERVAL)

    summary = collector.stop()
    report(summary, "size_aware_baseline", max_delay=args.max_delay, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
