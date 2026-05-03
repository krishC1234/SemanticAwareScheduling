#!/usr/bin/env python3
"""Submit eval jobs to the scheduler in randomized order with random delays,
then poll for completions and collect GPU/timing metrics.

Usage:
    python3 -m eval_data.run_eval                     # defaults
    python3 -m eval_data.run_eval --max-delay 60 --seed 42

Requires the scheduler server to be running:
    python3 -m scheduler.main
"""

import argparse
import json
import random
import socket
import sys
import time
from pathlib import Path

from eval_data.metrics import MetricsCollector

EVAL_JOBS_DIR = Path(__file__).parent / "jobs"
HOST = "localhost"
PORT = 9321
POLL_INTERVAL = 30


def send_to_scheduler(msg, host=HOST, port=PORT):
    """Send a JSON message to the scheduler and return the parsed response."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        print(f"Error: scheduler not running on {host}:{port}")
        print(f"Start it with: python3 -m scheduler.main")
        sys.exit(1)
    sock.sendall(json.dumps(msg).encode())
    response = sock.recv(65536).decode()
    sock.close()
    return json.loads(response)


def submit_job(script_path, port=PORT):
    """Submit a single job to the scheduler."""
    return send_to_scheduler({"scripts": [str(script_path.resolve())]}, port=port)


def query_status(port=PORT):
    """Query the scheduler for completed/running/queued status."""
    return send_to_scheduler({"query": "status"}, port=port)


def main():
    parser = argparse.ArgumentParser(description="Submit eval jobs with random delays")
    parser.add_argument("--max-delay", type=float, default=600,
                        help="Maximum delay between submissions (seconds)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--port", type=int, default=PORT,
                        help="Scheduler server port")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scripts = sorted(EVAL_JOBS_DIR.glob("*.py"))
    if not scripts:
        print(f"No .py files found in {EVAL_JOBS_DIR}")
        sys.exit(1)

    rng.shuffle(scripts)
    total_jobs = len(scripts)

    print(f"=== Eval workload: {total_jobs} jobs, delays 0-{args.max_delay}s ===\n")
    print("Submission order:")
    for i, s in enumerate(scripts, 1):
        print(f"  {i:2d}. {s.stem}")
    print()

    # Start metrics collection
    collector = MetricsCollector(interval=2)
    collector.pending = total_jobs
    collector.start()

    # Phase 1: Submit jobs with random delays
    for i, script in enumerate(scripts):
        print(f"[{i+1}/{total_jobs}] Submitting {script.stem}...", end=" ")
        results = submit_job(script, port=args.port)
        for r in results:
            if r["status"] == "queued":
                print(f"queued (k={r['k']:.3f})")
            else:
                print(f"failed: {r['error']}")
                collector.pending -= 1

        if i < total_jobs - 1:
            delay = rng.uniform(0, args.max_delay)
            print(f"         waiting {delay:.1f}s before next submission...")
            time.sleep(delay)

    print(f"\n=== All {total_jobs} jobs submitted, polling for completions ===\n")

    # Phase 2: Poll until all jobs complete
    while collector.pending > 0:
        status = query_status(port=args.port)
        for job in status["completed"]:
            if job["name"] not in collector._seen_jobs:
                collector.record_job(
                    name=job["name"],
                    gpus=job["gpus"],
                    run_time=job["run_time"],
                    wait_time=job["wait_time"],
                )
        running = status.get("running", [])
        queued = status.get("queued", 0)
        print(f"  [{collector.pending} pending, {len(running)} running, {queued} queued]")

        if collector.pending > 0:
            time.sleep(POLL_INTERVAL)

    # Phase 3: Stop collection and print summary
    summary = collector.stop()

    print(f"\n{'='*50}")
    print(f"  EVAL SUMMARY")
    print(f"{'='*50}")
    print(f"  Jobs completed:    {summary['num_jobs']}")
    print(f"  Wall-clock time:   {summary['wall_time']:.1f}s ({summary['wall_time']/60:.1f}m)")
    print(f"  Total run time:    {summary.get('total_run_time', 0):.1f}s")
    print(f"  Total wait time:   {summary.get('total_wait_time', 0):.1f}s")
    print(f"  Avg run time:      {summary.get('avg_run_time', 0):.1f}s")
    print(f"  Avg wait time:     {summary.get('avg_wait_time', 0):.1f}s")

    if "gpu" in summary:
        gpu = summary["gpu"]
        print(f"\n  GPU Utilization:")
        print(f"    Avg GPU util:    {gpu['avg_gpu_util_pct']}%")
        print(f"    Avg mem util:    {gpu['avg_mem_util_pct']}%")
        print(f"    Peak mem used:   {gpu['peak_mem_used_mb']} MB")
        print(f"    Avg mem used:    {gpu['avg_mem_used_mb']} MB")
        print(f"    Avg power draw:  {gpu['avg_power_w']} W")
        print(f"    Samples:         {gpu['num_samples']}")

    print(f"\n  Per-job breakdown:")
    for j in summary["jobs"]:
        print(f"    {j['name']:30s}  {j['gpus']} GPUs  "
              f"run={j['run_time']:7.1f}s  wait={j['wait_time']:6.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
