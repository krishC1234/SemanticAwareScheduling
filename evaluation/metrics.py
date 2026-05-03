"""Metrics collector for eval runs — GPU sampling + job timing in one place."""

import logging
import subprocess
import threading
import time

logger = logging.getLogger("scheduler")


class MetricsCollector:
    def __init__(self, interval=2):
        self.interval = interval
        self.gpu_samples = []
        self.jobs = []
        self.pending = 0
        self._seen_jobs = set()
        self._first_submit_time = None
        self._last_completion_time = None
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Begin background GPU sampling."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("MetricsCollector: started GPU sampling")

    def stop(self):
        """Stop sampling and return full summary."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        wall_time = time.time() - self._start_time
        logger.info("MetricsCollector: stopped GPU sampling")
        return self._summarize(wall_time)

    def record_submission(self, submit_time=None):
        """Record that a job was submitted. Call once per job at submission."""
        t = submit_time or time.time()
        if self._first_submit_time is None or t < self._first_submit_time:
            self._first_submit_time = t

    def record_job(self, name, gpus, run_time, wait_time, k=None):
        """Record a completed job's stats."""
        self._last_completion_time = time.time()
        self.jobs.append({
            "name": name,
            "gpus": gpus,
            "run_time": run_time,
            "wait_time": wait_time,
            "k": k,
        })
        self._seen_jobs.add(name)
        self.pending -= 1
        logger.info(f"MetricsCollector: recorded {name} "
                     f"(gpus={gpus}, run={run_time:.1f}s, wait={wait_time:.1f}s) "
                     f"— {self.pending} pending")

    def _sample_loop(self):
        """Background thread: sample nvidia-smi periodically."""
        while not self._stop_event.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,utilization.memory,"
                     "memory.used,memory.total,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0:
                    gpu_util_sum = 0
                    mem_util_sum = 0
                    mem_used_sum = 0
                    mem_total_sum = 0
                    power_sum = 0
                    count = 0
                    for line in out.stdout.strip().splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 5:
                            gpu_util_sum += float(parts[0])
                            mem_util_sum += float(parts[1])
                            mem_used_sum += float(parts[2])
                            mem_total_sum += float(parts[3])
                            power_sum += float(parts[4])
                            count += 1
                    if count > 0:
                        self.gpu_samples.append({
                            "gpu_util_pct": gpu_util_sum / count,
                            "mem_util_pct": mem_util_sum / count,
                            "mem_used_mb": mem_used_sum / count,
                            "mem_total_mb": mem_total_sum / count,
                            "power_w": power_sum / count,
                        })
            except Exception:
                pass
            self._stop_event.wait(self.interval)

    def _summarize(self, wall_time):
        """Build the full summary dict."""
        # Makespan: first submission to last completion
        if self._first_submit_time and self._last_completion_time:
            makespan = round(self._last_completion_time - self._first_submit_time, 1)
        else:
            makespan = None

        summary = {
            "wall_time": round(wall_time, 1),
            "makespan": makespan,
            "num_jobs": len(self.jobs),
            "jobs": self.jobs,
        }

        # Job timing aggregates
        if self.jobs:
            total_run = sum(j["run_time"] for j in self.jobs)
            total_wait = sum(j["wait_time"] for j in self.jobs)
            summary["total_run_time"] = round(total_run, 1)
            summary["total_wait_time"] = round(total_wait, 1)
            summary["avg_wait_time"] = round(total_wait / len(self.jobs), 1)
            summary["avg_run_time"] = round(total_run / len(self.jobs), 1)

        # GPU aggregates
        if self.gpu_samples:
            n = len(self.gpu_samples)
            summary["gpu"] = {
                "avg_gpu_util_pct": round(sum(s["gpu_util_pct"] for s in self.gpu_samples) / n, 1),
                "avg_mem_util_pct": round(sum(s["mem_util_pct"] for s in self.gpu_samples) / n, 1),
                "peak_mem_used_mb": round(max(s["mem_used_mb"] for s in self.gpu_samples)),
                "avg_mem_used_mb": round(sum(s["mem_used_mb"] for s in self.gpu_samples) / n),
                "avg_power_w": round(sum(s["power_w"] for s in self.gpu_samples) / n, 1),
                "num_samples": n,
            }

        return summary
