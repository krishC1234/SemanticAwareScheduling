#!/usr/bin/env python3
"""Run all evaluation scripts sequentially with the same seed.

Starts the semantic-aware scheduler server automatically for the
scheduler eval, then shuts it down before running the baselines.

Usage:
    python3 -m evaluation.run_all
    python3 -m evaluation.run_all --seed 42 --max-delay 120
"""

import argparse
import subprocess
import sys
import time
import signal
import os


SEED = 42
MAX_DELAY = 600
SCHEDULER_PORT = 9321


def run_script(module, seed, max_delay, extra_args=None):
    """Run a test script as a subprocess and wait for it to finish."""
    cmd = [
        sys.executable, "-m", module,
        "--seed", str(seed),
        "--max-delay", str(max_delay),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'#'*60}")
    print(f"  Running: {module}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'#'*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nWARNING: {module} exited with code {result.returncode}")
    return result.returncode


def start_scheduler(port):
    """Start the scheduler server as a background process."""
    print(f"Starting scheduler server on port {port}...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "scheduler.main", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Give it a moment to bind the port
    time.sleep(3)
    if proc.poll() is not None:
        print(f"ERROR: scheduler failed to start")
        print(proc.stderr.read().decode())
        sys.exit(1)
    print(f"Scheduler server started (PID {proc.pid})")
    return proc


def stop_scheduler(proc):
    """Gracefully stop the scheduler server."""
    print(f"Stopping scheduler server (PID {proc.pid})...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("Scheduler server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run all eval scripts sequentially")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed for all scripts (default: {SEED})")
    parser.add_argument("--max-delay", type=float, default=MAX_DELAY,
                        help=f"Max delay between job submissions (default: {MAX_DELAY})")
    parser.add_argument("--port", type=int, default=SCHEDULER_PORT,
                        help=f"Scheduler server port (default: {SCHEDULER_PORT})")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  FULL EVALUATION SUITE")
    print(f"  seed={args.seed}  max_delay={args.max_delay}s  port={args.port}")
    print(f"{'='*60}")

    # 1. Scheduler eval (needs the server running)
    scheduler_proc = start_scheduler(args.port)
    try:
        run_script(
            "evaluation.test_scripts.scheulder_eval",
            args.seed, args.max_delay,
            extra_args=["--port", str(args.port)],
        )
    finally:
        stop_scheduler(scheduler_proc)

    # 2. Baselines (no server needed)
    run_script("evaluation.test_scripts.greedy_baseline", args.seed, args.max_delay)
    run_script("evaluation.test_scripts.polite_baseline", args.seed, args.max_delay)
    run_script("evaluation.test_scripts.equal_share_baseline", args.seed, args.max_delay)

    print(f"\n{'='*60}")
    print(f"  ALL EVALUATIONS COMPLETE")
    print(f"  Results saved to evaluation/test_results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()