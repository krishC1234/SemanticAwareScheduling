#!/usr/bin/env python3
"""Submit eval jobs to the scheduler in randomized order with random delays.

Simulates a realistic workload where jobs arrive over time — some finish
before others even enter the queue, forcing the scheduler to make
allocation decisions with a changing job mix.

Usage:
    python3 -m eval_data.run_eval                     # defaults
    python3 -m eval_data.run_eval --min-delay 5 --max-delay 30
    python3 -m eval_data.run_eval --seed 42

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

EVAL_JOBS_DIR = Path(__file__).parent / "jobs"
HOST = "localhost"
PORT = 9100


def submit_to_scheduler(script_path, host=HOST, port=PORT):
    """Send a single job to the running scheduler server."""
    msg = json.dumps({"scripts": [str(script_path.resolve())]})
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        print(f"Error: scheduler not running on {host}:{port}")
        print(f"Start it with: python3 -m scheduler.main")
        sys.exit(1)
    sock.sendall(msg.encode())
    response = sock.recv(8192).decode()
    sock.close()
    return json.loads(response)


def main():
    parser = argparse.ArgumentParser(description="Submit eval jobs with random delays")
    parser.add_argument("--min-delay", type=float, default=300,
                        help="Minimum delay between submissions (seconds)")
    parser.add_argument("--max-delay", type=float, default=1500,
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
    wall_start = time.time()

    print(f"=== Eval workload: {len(scripts)} jobs, "
          f"delays {args.min_delay}-{args.max_delay}s ===\n")
    print(f"Submission order:")
    for i, s in enumerate(scripts, 1):
        print(f"  {i:2d}. {s.stem}")
    print()

    for i, script in enumerate(scripts):
        # Submit
        print(f"[{i+1}/{len(scripts)}] Submitting {script.stem}...", end=" ")
        results = submit_to_scheduler(script, port=args.port)
        for r in results:
            if r["status"] == "queued":
                print(f"queued (k={r['k']:.3f})")
            else:
                print(f"failed: {r['error']}")

        # Wait before next submission (skip delay after last job)
        if i < len(scripts) - 1:
            delay = rng.uniform(args.min_delay, args.max_delay)
            print(f"         waiting {delay:.1f}s before next submission...")
            time.sleep(delay)

    wall_elapsed = time.time() - wall_start
    print(f"\n=== All {len(scripts)} jobs submitted ===")
    print(f"Total wall-clock time: {wall_elapsed:.1f}s ({wall_elapsed/60:.1f}m)")
    print(f"Monitor progress in the scheduler terminal.")


if __name__ == "__main__":
    main()