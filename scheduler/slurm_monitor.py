"""Polls SLURM to track available GPUs and running jobs."""

import subprocess
import re
import time
from scheduler.logger import logger

total_gpus = None


def get_total_gpus():
    """Query sinfo for total GPU count on the node."""
    try:
        out = subprocess.run(["sinfo", "--Node", "--Format=gres:50", "--noheader"], capture_output=True, text=True, timeout=10,)
        # Parse lines like "gpu:8" or "gpu:nvidia:8"
        total = 0
        for line in out.stdout.strip().splitlines():
            m = re.search(r"gpu(?::\w+)?:(\d+)", line)
            if m:
                total += int(m.group(1))
        result = total if total > 0 else 8
        logger.info(f"SLURM: detected {result} total GPUs")
        return result
    except Exception as e:
        logger.warning(f"SLURM: sinfo failed ({e}), defaulting to 8 GPUs")
        return 8


def get_used_gpus():
    """Query squeue for GPUs currently allocated to running jobs."""
    try:
        out = subprocess.run(
            ["squeue", "--Format=tres-alloc:80", "--noheader", "--states=RUNNING"],
            capture_output=True, text=True, timeout=10,
        )
        used = 0
        for line in out.stdout.strip().splitlines():
            m = re.search(r"gpu=(\d+)", line)
            if m:
                used += int(m.group(1))
        return used
    except Exception:
        return 0


def get_available_gpus():
    """Return number of idle GPUs on the cluster."""
    global total_gpus
    if total_gpus is None: total_gpus = get_total_gpus()
    return total_gpus - get_used_gpus()


def get_running_job_ids():
    """Return set of SLURM job IDs that are currently running."""
    try:
        out = subprocess.run(
            ["squeue", "--Format=jobid", "--noheader", "--states=RUNNING"],
            capture_output=True, text=True, timeout=10,
        )
        return {line.strip() for line in out.stdout.strip().splitlines() if line.strip()}
    except Exception:
        return set()