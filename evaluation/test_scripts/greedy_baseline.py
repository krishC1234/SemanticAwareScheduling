#!/usr/bin/env python3
"""Greedy baseline — every job requests ALL GPUs on the node.

Jobs run one-at-a-time since each claims the entire node.

Usage:
    python3 -m evaluation.test_scripts.greedy_baseline
    python3 -m evaluation.test_scripts.greedy_baseline --seed 42 --max-delay 60
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

from evaluation.metrics import MetricsCollector, parse_job_runtime
from evaluation.report import report
from scheduler.slurm_monitor import get_total_gpus

EVAL_JOBS_DIR = Path(__file__).parent.parent / "jobs"
POLL_INTERVAL = 30


LOGS_DIR = EVAL_JOBS_DIR.parent.parent / "logs" / "greedy_output"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


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
        description="Greedy baseline: every job gets ALL GPUs")
    parser.add_argument("--max-delay", type=float, default=600,
                        help="Maximum delay between submissions (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Shared output directory for this run")
    args = parser.parse_args()

    total_gpus = get_total_gpus()
    rng = random.Random(args.seed)
    scripts = sorted(EVAL_JOBS_DIR.glob("*.py"))
    if not scripts:
        print(f"No .py files found in {EVAL_JOBS_DIR}")
        sys.exit(1)

    rng.shuffle(scripts)
    total_jobs = len(scripts)

    print(f"=== Greedy baseline: {total_jobs} jobs, {total_gpus} GPU(s)/job, "
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
        slurm_id, submit_time = sbatch_submit(script, total_gpus)
        if slurm_id:
            log_file = LOGS_DIR / f"{script.stem}_{total_gpus}gpu.log"
            tracked[slurm_id] = {
                "name": script.stem,
                "gpus": total_gpus,
                "submit_time": submit_time,
                "enqueue_time": enqueue_time,
                "log_file": log_file,
            }
            print(f"SLURM job {slurm_id} ({total_gpus} GPUs)")
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
    report(summary, f"greedy_baseline_{total_gpus}gpu", max_delay=args.max_delay, run_dir=args.run_dir)


if __name__ == "__main__":
    main()