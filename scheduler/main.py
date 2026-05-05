"""Semantic-Aware Scheduler — server process.

Usage:
    python3 -m scheduler.main                      # start server
    python3 -m scheduler.main --poll 10 --lam 0.01 # tune params

Listens on localhost:9321 for job submissions from the submit client.
Ctrl-C to shut down gracefully.
"""

import csv
import json
import re
import socket
import threading
import time
import argparse
from pathlib import Path

from scheduler.job_profiler import JobProfiler
from scheduler.logger import logger
from scheduler.queue import Queue
from scheduler.slurm_monitor import get_available_gpus, get_running_job_ids
from scheduler.sbatch_wrapper import submit_allocation

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_CSV = ROOT / "train_data" / "benchmark.csv"

POLL_INTERVAL = 5
HOST = "localhost"
PORT = 9321

class Scheduler:
    def __init__(self, queue, job_profiler, poll_interval=POLL_INTERVAL):
        self.queue = queue
        self.job_profiler = job_profiler
        self.poll_interval = poll_interval
        self.running = {}       # {slurm_id: job}
        self.completed = []
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._next_batch_id = 0
        self._batches = {}      # {batch_id: {"total": N, "done": 0, "total_time": 0, "start": time}}

    def submit_script(self, path, batch_id=None):
        """Classify and enqueue a single script. Thread-safe."""
        path = Path(path)
        with self.lock:
            job = self.job_profiler.submit(path)
            job.batch_id = batch_id
        logger.info(f"queued: {job}")
        return job

    def listen(self, host=HOST, port=None):
        """Listen for submit requests over TCP."""
        if port is None:
            port = PORT
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(10)
        server.settimeout(1)  # so we can check _stop periodically
        logger.info(f"listening on {host}:{port}")

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
        """Handle a single submit or status query request."""
        try:
            data = conn.recv(4096).decode().strip()
            msg = json.loads(data)

            if "query" in msg:
                self._handle_query(conn, msg)
            else:
                self._handle_submit(conn, msg)
        except Exception as e:
            logger.error(f"client handler error: {e}")
            conn.sendall(json.dumps({"error": str(e)}).encode())
        finally:
            conn.close()

    def _handle_query(self, conn, msg):
        """Handle status queries from eval harness."""
        if msg["query"] == "status":
            response = {
                "completed": [
                    {"name": j.model_name, "gpus": j.assigned_gpus,
                     "run_time": round(j.run_time, 1),
                     "wait_time": round(j.wait_time, 1),
                     "k": round(j.k, 4)}
                    for j in self.completed
                ],
                "running": [j.model_name for j in self.running.values()],
                "queued": len(self.queue),
            }
            conn.sendall(json.dumps(response).encode())
        else:
            conn.sendall(json.dumps({"error": f"unknown query: {msg['query']}"}).encode())

    def _handle_submit(self, conn, msg):
        """Handle job submission requests."""
        paths = msg.get("scripts", [])
        logger.info(f"received submission: {len(paths)} script(s)")

        batch_id = self._next_batch_id
        self._next_batch_id += 1
        self._batches[batch_id] = {
            "total": len(paths),
            "done": 0,
            "total_time": 0.0,
            "start": time.time(),
        }
        logger.info(f"created batch {batch_id} with {len(paths)} job(s)")

        results = []
        for p in paths:
            try:
                job = self.submit_script(p, batch_id=batch_id)
                results.append({"path": p, "status": "queued", "k": round(job.k, 4)})
            except Exception as e:
                logger.error(f"failed to submit {p}: {e}")
                results.append({"path": p, "status": "error", "error": str(e)})
        conn.sendall(json.dumps(results).encode())

    def scheduler_loop(self):
        """Main loop: poll SLURM, check completions, allocate, submit."""
        while not self._stop.is_set():
            # Check for completed jobs
            if self.running:
                active_ids = get_running_job_ids()
                finished = [sid for sid in self.running if sid not in active_ids]
                if finished:
                    logger.debug(f"poll: {len(finished)} job(s) finished, " f"{len(active_ids)} still active")
                for sid in finished:
                    job = self.running.pop(sid)
                    completion_time = time.time()
                    # Use actual runtime from job output if available
                    parsed_time = self._parse_job_runtime(job)
                    if parsed_time is not None:
                        job.run_time = parsed_time
                        job.wait_time = completion_time - job.start_time - parsed_time
                    else:
                        job.run_time = completion_time - job.submit_time
                        job.wait_time = job.submit_time - job.start_time
                    self.completed.append(job)
                    logger.info(f"completed: {job.model_name} (SLURM {sid}, "
                               f"{job.assigned_gpus} GPUs, {job.run_time:.1f}s, "
                               f"waited {job.wait_time:.1f}s in queue)")
                    self._append_benchmark(job)

                    if job.batch_id is not None and job.batch_id in self._batches:
                        batch = self._batches[job.batch_id]
                        batch["done"] += 1
                        batch["total_time"] += job.run_time
                        logger.debug(f"batch {job.batch_id}: {batch['done']}/{batch['total']} done")
                        if batch["done"] == batch["total"]:
                            wall = time.time() - batch["start"]
                            logger.info(f"=== Batch {job.batch_id} complete ({batch['total']} jobs) ===")
                            logger.info(f"  Sum of job times: {batch['total_time']:.1f}s")
                            logger.info(f"  Wall-clock time:  {wall:.1f}s")
                            logger.info(f"===========================")
                            del self._batches[job.batch_id]

            # Allocate idle GPUs to waiting jobs
            with self.lock:
                queue_len = len(self.queue)
                if queue_len > 0:
                    available = get_available_gpus()
                    logger.debug(f"allocation check: {queue_len} queued, "
                                 f"{available} GPUs free, {len(self.running)} running")
                    if available > 0:
                        logger.info(f"{available} GPU(s) available, allocating...")
                        allocation = self.queue.allocate(available)
                        if allocation:
                            for job, gpus in allocation:
                                logger.info(f"allocated {gpus} GPU(s) to {job.model_name} (k={job.k:.3f})")
                            submitted = submit_allocation(allocation)
                            self.running.update(submitted)
                            logger.info(f"state: {len(self.running)} running, "
                                       f"{len(self.queue)} queued, "
                                       f"{len(self.completed)} completed")

            self._stop.wait(self.poll_interval)

    def _parse_job_runtime(self, job):
        """Parse total_time_sec from job's ###RESULTS### block."""
        log_file = getattr(job, "log_file", None)
        if not log_file or not Path(log_file).exists():
            return None
        try:
            text = Path(log_file).read_text()
            match = re.search(r"###RESULTS###\s*\n(.+?)\n\s*###END_RESULTS###", text)
            if match:
                results = json.loads(match.group(1))
                return float(results.get("total_time_sec", 0))
        except Exception:
            pass
        return None

    def _append_benchmark(self, job):
        """Parse job output log and append a row to benchmark.csv."""
        log_file = getattr(job, "log_file", None)
        if not log_file or not Path(log_file).exists():
            logger.warning(f"benchmark: no log file for {job.model_name}, skipping CSV append")
            return

        try:
            text = Path(log_file).read_text()
        except Exception as e:
            logger.error(f"benchmark: failed to read {log_file}: {e}")
            return

        # Parse ###RESULTS### JSON block
        match = re.search(r"###RESULTS###\s*\n(.+?)\n\s*###END_RESULTS###", text)
        if not match:
            logger.warning(f"benchmark: no RESULTS block in {log_file}")
            return

        try:
            results = json.loads(match.group(1))
        except json.JSONDecodeError as e:
            logger.error(f"benchmark: bad JSON in {log_file}: {e}")
            return

        # Parse per-job GPU stats
        peak_vram, avg_sm, avg_mem_bw = 0, 0, 0
        gpu_stats_file = getattr(job, "gpu_stats_file", None)
        if gpu_stats_file and Path(gpu_stats_file).exists():
            try:
                lines = Path(gpu_stats_file).read_text().strip().splitlines()
                if lines:
                    vrams, sms, mem_bws = [], [], []
                    for line in lines:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            vrams.append(float(parts[0]))
                            sms.append(float(parts[1]))
                            mem_bws.append(float(parts[2]))
                    if vrams:
                        peak_vram = round(max(vrams))
                        avg_sm = round(sum(sms) / len(sms), 1)
                        avg_mem_bw = round(sum(mem_bws) / len(mem_bws), 1)
            except Exception as e:
                logger.warning(f"benchmark: failed to parse GPU stats for {job.model_name}: {e}")

        row = {
            "model": job.model_name,
            "config": job.model_name,
            "batch_size": results.get("batch_size", 0),
            "param_count": results.get("param_count", 0),
            "gpu_count": job.assigned_gpus,
            "total_time_sec": results.get("total_time_sec", round(job.run_time, 2)),
            "avg_throughput": results.get("avg_throughput", 0),
            "peak_vram_mb": peak_vram,
            "avg_sm_util_pct": avg_sm,
            "avg_mem_bw_pct": avg_mem_bw,
        }

        header = list(row.keys())
        write_header = not BENCHMARK_CSV.exists() or BENCHMARK_CSV.stat().st_size == 0
        with BENCHMARK_CSV.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
            w.writerow(row)
        logger.info(f"benchmark: appended {job.model_name} ({job.assigned_gpus} GPUs, "
                     f"{row['total_time_sec']}s) to {BENCHMARK_CSV.name}")

    def run(self):
        """Start the scheduler. Blocks until Ctrl-C."""
        logger.info(f"=== Scheduler running (poll={self.poll_interval}s) ===")

        # Start TCP listener for job submissions
        listener = threading.Thread(
            target=self.listen, daemon=True
        )
        listener.start()

        # Run scheduler loop in main thread
        try:
            self.scheduler_loop()
        except KeyboardInterrupt:
            logger.info("=== Shutting down ===")
            self._stop.set()
            logger.info(f"{len(self.completed)} job(s) completed")
            logger.info(f"{len(self.running)} job(s) still running in SLURM")
            logger.info(f"{len(self.queue)} job(s) still in queue")


def main():
    global PORT
    parser = argparse.ArgumentParser(description="Semantic-Aware GPU Scheduler Server")
    parser.add_argument("--lam", type=float, default=0.001, help="Starvation factor")
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL, help="Poll interval (sec)")
    parser.add_argument("--port", type=int, default=PORT, help="Listen port")
    args = parser.parse_args()

    logger.info(f"config: lam={args.lam}, poll={args.poll}s, port={args.port}")
    queue = Queue(lam=args.lam)
    job_profiler = JobProfiler(queue)
    scheduler = Scheduler(queue, job_profiler, poll_interval=args.poll)
    PORT = args.port
    scheduler.run()


if __name__ == "__main__":
    main()
