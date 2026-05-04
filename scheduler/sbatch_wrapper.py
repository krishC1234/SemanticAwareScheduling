"""Submits allocated jobs to SLURM via sbatch/torchrun."""

import subprocess
import time
from pathlib import Path
from scheduler.logger import logger

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs" / "scheduler_jobs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def submit_job(job, gpu_count):
    """Submit a job to SLURM with the given GPU allocation.

    Wraps the DDP script in a torchrun call via sbatch.
    Returns the SLURM job ID on success, None on failure.
    """
    log_file = LOGS_DIR / f"{job.model_name}_{gpu_count}gpu.log"
    gpu_stats_file = LOGS_DIR / f"{job.model_name}_{gpu_count}gpu_stats.csv"
    sbatch_script = (
        "#!/bin/bash\n"
        f"#SBATCH --gres=gpu:{gpu_count}\n"
        f"#SBATCH --job-name={job.model_name}\n"
        f"#SBATCH --output={log_file}\n"
        f"nvidia-smi --query-gpu=memory.used,utilization.gpu,utilization.memory "
        f"--format=csv,noheader,nounits -l 1 > {gpu_stats_file} 2>/dev/null &\n"
        f"MONITOR_PID=$!\n"
        f"torchrun --standalone --nproc_per_node={gpu_count} {job.path}\n"
        f"kill $MONITOR_PID 2>/dev/null || true\n"
    )

    try:
        result = subprocess.run(
            ["sbatch", "--parsable"],
            input=sbatch_script,
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            slurm_id = result.stdout.strip()
            job.log_file = log_file
            job.gpu_stats_file = gpu_stats_file
            logger.info(f"submitted {job.model_name} -> SLURM job {slurm_id} " f"({gpu_count} GPUs, k={job.k:.3f})")
            return slurm_id
        else:
            logger.error(f"sbatch failed for {job.model_name}: {result.stderr.strip()}")
            return None
    except Exception as e:
        logger.error(f"sbatch error for {job.model_name}: {e}")
        return None


def submit_allocation(allocation):
    """Submit a list of (job, gpu_count) pairs from queue.allocate().

    Returns dict {slurm_job_id: job} for successfully submitted jobs.
    """
    submitted = {}
    for job, gpu_count in allocation:
        slurm_id = submit_job(job, gpu_count)
        if slurm_id:
            job.slurm_id = slurm_id
            job.submit_time = time.time()
            submitted[slurm_id] = job
    return submitted