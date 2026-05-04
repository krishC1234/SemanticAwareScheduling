#!/usr/bin/env python3
"""Equal-share baseline — divide available GPUs evenly among pending jobs.

Manages its own FIFO queue. When GPUs are free, divides them equally
among all waiting jobs and submits the batch. Each job gets at least 1
GPU; if there are more jobs than GPUs, only the first N are submitted
and the rest wait.

Usage:
    python3 -m evaluation.test_scripts.equal_share_baseline
    python3 -m evaluation.test_scripts.equal_share_baseline --seed 42 --max-delay 60
"""

import argparse
import random
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from evaluation.metrics import MetricsCollector
from evaluation.report import report
from scheduler.slurm_monitor import get_total_gpus, get_available_gpus

EVAL_JOBS_DIR = Path(__file__).parent.parent / "jobs"
POLL_INTERVAL = 10


def sbatch_submit(script_path, gpu_count):
    """Submit a DDP script to SLURM via sbatch + torchrun."""
    sbatch_script = (
        "#!/bin/bash\n"
        f"#SBATCH --gres=gpu:{gpu_count}\n"
        f"#SBATCH --job-name={script_path.stem}\n"
        "#SBATCH --output=/dev/null\n"
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


def get_running_job_ids():
    """Return set of SLURM job IDs that are actually RUNNING (not pending)."""
    try:
        out = subprocess.run(
            ["squeue", "--Format=jobid", "--noheader",
             "--states=RUNNING"],
            capture_output=True, text=True, timeout=10,
        )
        return {line.strip() for line in out.stdout.strip().splitlines() if line.strip()}
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Equal-share baseline: divide GPUs evenly among pending jobs")
    parser.add_argument("--max-delay", type=float, default=600,
                        help="Maximum delay between submissions (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    total_gpus = get_total_gpus()
    rng = random.Random(args.seed)
    scripts = sorted(EVAL_JOBS_DIR.glob("*.py"))
    if not scripts:
        print(f"No .py files found in {EVAL_JOBS_DIR}")
        sys.exit(1)

    rng.shuffle(scripts)
    total_jobs = len(scripts)

    print(f"=== Equal-share baseline: {total_jobs} jobs, {total_gpus} total GPUs, "
          f"delays 0-{args.max_delay}s ===\n")
    print("Submission order:")
    for i, s in enumerate(scripts, 1):
        print(f"  {i:2d}. {s.stem}")
    print()

    collector = MetricsCollector(interval=2)
    collector.pending = total_jobs
    collector.start()

    # Internal FIFO queue: {"script": Path, "enqueue_time": float}
    wait_queue = deque()
    tracked = {}
    seen = set()
    start_times = {}

    # Pre-compute arrival delays
    arrival_delays = [0.0] + [rng.uniform(0, args.max_delay) for _ in range(total_jobs - 1)]
    arrival_idx = 0
    next_arrival = time.time()
    arrivals_done = False

    print("Starting job arrivals and queue management...\n")

    while collector.pending > 0:
        now = time.time()

        # Handle new arrivals
        while arrival_idx < total_jobs and now >= next_arrival:
            script = scripts[arrival_idx]
            enqueue_time = time.time()
            collector.record_submission(enqueue_time)
            wait_queue.append({"script": script, "enqueue_time": enqueue_time})
            print(f"[{arrival_idx+1}/{total_jobs}] Arrived: {script.stem} (queued)")

            arrival_idx += 1
            if arrival_idx < total_jobs:
                next_arrival = time.time() + arrival_delays[arrival_idx]

        if arrival_idx >= total_jobs and not arrivals_done:
            arrivals_done = True
            print(f"\n=== All {total_jobs} jobs arrived, draining queue ===\n")

        # Track when jobs transition to RUNNING
        active = get_active_job_ids()
        running = get_running_job_ids()
        for slurm_id in running:
            if slurm_id in tracked and slurm_id not in start_times:
                start_times[slurm_id] = time.time()

        # Check for completed jobs
        for slurm_id, info in list(tracked.items()):
            if slurm_id not in active and slurm_id not in seen:
                start = start_times.get(slurm_id, info["submit_time"])
                run_time = time.time() - start
                wait_time = start - info["enqueue_time"]
                collector.record_job(
                    name=info["name"],
                    gpus=info["gpus"],
                    run_time=run_time,
                    wait_time=wait_time,
                )
                seen.add(slurm_id)
                print(f"  Completed: {info['name']} "
                      f"({info['gpus']} GPUs, run={run_time:.1f}s, "
                      f"wait={wait_time:.1f}s)")

        # Divide available GPUs equally among waiting jobs
        available = get_available_gpus()
        if wait_queue and available > 0:
            n_pending = len(wait_queue)
            # How many jobs can we run? At least 1 GPU each.
            n_to_submit = min(n_pending, available)
            gpus_each = available // n_to_submit

            print(f"  Allocating: {available} GPUs / {n_to_submit} jobs "
                  f"= {gpus_each} GPU(s) each")

            for _ in range(n_to_submit):
                entry = wait_queue.popleft()
                script = entry["script"]
                slurm_id, submit_time = sbatch_submit(script, gpus_each)
                if slurm_id:
                    tracked[slurm_id] = {
                        "name": script.stem,
                        "gpus": gpus_each,
                        "submit_time": submit_time,
                        "enqueue_time": entry["enqueue_time"],
                    }
                    print(f"  Submitted: {script.stem} -> SLURM job {slurm_id} "
                          f"({gpus_each} GPUs)")
                else:
                    print(f"  Failed: {script.stem}")
                    collector.pending -= 1

        if collector.pending > 0:
            time.sleep(POLL_INTERVAL)

    summary = collector.stop()
    report(summary, "equal_share_baseline", max_delay=args.max_delay)


if __name__ == "__main__":
    main()