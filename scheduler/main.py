"""Semantic-Aware Scheduler — server process.

Usage:
    python3 -m scheduler.main                      # start server
    python3 -m scheduler.main --poll 10 --lam 0.01 # tune params

Listens on localhost:9100 for job submissions from the submit client.
Ctrl-C to shut down gracefully.
"""

import json
import socket
import threading
import time
from pathlib import Path

from scheduler.job_profiler import Intake
from scheduler.queue import Queue
from scheduler.slurm_monitor import get_available_gpus, get_running_job_ids
from scheduler.sbatch_wrapper import submit_allocation

POLL_INTERVAL = 5
HOST = "localhost"
PORT = 9100


class Scheduler:
    def __init__(self, queue, intake, poll_interval=POLL_INTERVAL):
        self.queue = queue
        self.intake = intake
        self.poll_interval = poll_interval
        self.running = {}       # {slurm_id: job}
        self.completed = []
        self.lock = threading.Lock()
        self._stop = threading.Event()

    def submit_script(self, path):
        """Classify and enqueue a single script. Thread-safe."""
        path = Path(path)
        with self.lock:
            job = self.intake.submit(path)
        print(f"  queued: {job}")
        return job

    def listen(self, host=HOST, port=PORT):
        """Listen for submit requests over TCP."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(10)
        server.settimeout(1)  # so we can check _stop periodically
        print(f"  listening on {host}:{port}")

        while not self._stop.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            threading.Thread(
                target=self._handle_client, args=(conn,), daemon=True
            ).start()
        server.close()

    def _handle_client(self, conn):
        """Handle a single submit request."""
        try:
            data = conn.recv(4096).decode().strip()
            msg = json.loads(data)
            paths = msg.get("scripts", [])
            results = []
            for p in paths:
                try:
                    job = self.submit_script(p)
                    results.append({"path": p, "status": "queued", "k": round(job.k, 4)})
                except Exception as e:
                    results.append({"path": p, "status": "error", "error": str(e)})
            conn.sendall(json.dumps(results).encode())
        except Exception as e:
            conn.sendall(json.dumps({"error": str(e)}).encode())
        finally:
            conn.close()

    def scheduler_loop(self):
        """Main loop: poll SLURM, check completions, allocate, submit."""
        while not self._stop.is_set():
            # Check for completed jobs
            if self.running:
                active_ids = get_running_job_ids()
                finished = [sid for sid in self.running if sid not in active_ids]
                for sid in finished:
                    job = self.running.pop(sid)
                    elapsed = time.time() - job.submit_time
                    self.completed.append(job)
                    print(f"  completed: {job.model_name} (SLURM {sid}, "
                          f"{job.assigned_gpus} GPUs, {elapsed:.1f}s)")

            # Allocate idle GPUs to waiting jobs
            with self.lock:
                if len(self.queue) > 0:
                    available = get_available_gpus()
                    if available > 0:
                        print(f"\n  {available} GPU(s) available, allocating...")
                        allocation = self.queue.allocate(available)
                        if allocation:
                            submitted = submit_allocation(allocation)
                            self.running.update(submitted)

            self._stop.wait(self.poll_interval)

    def run(self):
        """Start the scheduler. Blocks until Ctrl-C."""
        print(f"=== Scheduler running (poll={self.poll_interval}s) ===\n")

        # Start TCP listener for job submissions
        listener = threading.Thread(
            target=self.listen, daemon=True
        )
        listener.start()

        # Run scheduler loop in main thread
        try:
            self.scheduler_loop()
        except KeyboardInterrupt:
            print("\n\n=== Shutting down ===")
            self._stop.set()
            print(f"  {len(self.completed)} job(s) completed")
            print(f"  {len(self.running)} job(s) still running in SLURM")
            print(f"  {len(self.queue)} job(s) still in queue")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic-Aware GPU Scheduler Server")
    parser.add_argument("--lam", type=float, default=0.001, help="Starvation factor")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL, help="Poll interval (sec)")
    parser.add_argument("--port", type=int, default=PORT, help="Listen port")
    args = parser.parse_args()

    queue = Queue(lam=args.lam)
    intake = Intake(queue)
    scheduler = Scheduler(queue, intake, poll_interval=args.poll)
    global PORT
    PORT = args.port
    scheduler.run()


if __name__ == "__main__":
    main()
