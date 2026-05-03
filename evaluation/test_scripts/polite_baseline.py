#!/usr/bin/env python3
"""Polite baseline — every job requests exactly 1 GPU.

SLURM can pack multiple jobs concurrently (up to N_GPUs jobs at once),
but each job only uses a single GPU.

Usage:
    python3 -m evaluation.test_scripts.polite_baseline
    python3 -m evaluation.test_scripts.polite_baseline --seed 42 --max-delay 60
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from evaluation.metrics import MetricsCollector
from evaluation.report import report

EVAL_JOBS_DIR = Path(__file__).parent.parent / "jobs"
POLL_INTERVAL = 30
GPUS_PER_JOB = 1


def sbatch_submit(script_path):
    """Submit a DDP script to SLURM via sbatch + torchrun with 1 GPU."""
    sbatch_script = (
        "#!/bin/bash\n"
        f"#SBATCH --gres=gpu:{GPUS_PER_JOB}\n"
        f"#SBATCH --job-name={script_path.stem}\n"
        "#SBATCH --output=/dev/null\n"
        f"torchrun --standalone --nproc_per_node={GPUS_PER_JOB} {script_path.resolve()}\n"
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


def get_job_runtime(slurm_id):
    """Query sacct for elapsed wall time of a completed SLURM job."""
    try:
        out = subprocess.run(
            ["sacct", "-j", slurm_id, "--format=Elapsed",
             "--noheader", "--parsable2"],
            capture_output=True, text=True, timeout=10,
        )
        for line in out.stdout.strip().splitlines():
            line = line.strip()
            if not line or line == "Elapsed":
                continue
            parts = line.split("-")
            if len(parts) == 2:
                days, hms = int(parts[0]), parts[1]
            else:
                days, hms = 0, parts[0]
            h, m, s = hms.split(":")
            return days * 86400 + int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Polite baseline: every job gets 1 GPU")
    parser.add_argument("--max-delay", type=float, default=600,
                        help="Maximum delay between submissions (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scripts = sorted(EVAL_JOBS_DIR.glob("*.py"))
    if not scripts:
        print(f"No .py files found in {EVAL_JOBS_DIR}")
        sys.exit(1)

    rng.shuffle(scripts)
    total_jobs = len(scripts)

    print(f"=== Polite baseline: {total_jobs} jobs, {GPUS_PER_JOB} GPU/job, "
          f"delays 0-{args.max_delay}s ===\n")
    print("Submission order:")
    for i, s in enumerate(scripts, 1):
        print(f"  {i:2d}. {s.stem}")
    print()

    collector = MetricsCollector(interval=2)
    collector.pending = total_jobs
    collector.start()

    tracked = {}

    for i, script in enumerate(scripts):
        enqueue_time = time.time()
        collector.record_submission(enqueue_time)
        print(f"[{i+1}/{total_jobs}] Submitting {script.stem}...", end=" ")
        slurm_id, submit_time = sbatch_submit(script)
        if slurm_id:
            tracked[slurm_id] = {
                "name": script.stem,
                "gpus": GPUS_PER_JOB,
                "submit_time": submit_time,
                "enqueue_time": enqueue_time,
            }
            print(f"SLURM job {slurm_id} ({GPUS_PER_JOB} GPU)")
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
                run_time = get_job_runtime(slurm_id)
                if run_time == 0.0:
                    continue
                wait_time = info["submit_time"] - info["enqueue_time"]
                collector.record_job(
                    name=info["name"],
                    gpus=info["gpus"],
                    run_time=run_time,
                    wait_time=wait_time,
                )
                seen.add(slurm_id)

        print(f"  [{collector.pending} pending, "
              f"{len(active & set(tracked.keys()))} running]")

        if collector.pending > 0:
            time.sleep(POLL_INTERVAL)

    summary = collector.stop()
    report(summary, "polite_baseline_1gpu")


if __name__ == "__main__":
    main()