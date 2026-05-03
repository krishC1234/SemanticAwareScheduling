"""Shared reporting for all eval scripts.

Formats a MetricsCollector summary, prints it, and writes it to
evaluation/test_results/<label>.txt.
"""

from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "test_results"


def report(summary, label):
    """Print and save an eval summary.

    Args:
        summary: dict returned by MetricsCollector.stop()
        label: identifier used in the header and filename
               (e.g. "scheduler", "greedy_baseline", "polite_baseline")
    """
    lines = []

    def emit(s=""):
        lines.append(s)

    emit(f"\n{'='*50}")
    emit(f"  {label.upper()} SUMMARY")
    emit(f"{'='*50}")
    emit(f"  Jobs completed:    {summary['num_jobs']}")
    emit(f"  Wall-clock time:   {summary['wall_time']:.1f}s ({summary['wall_time']/60:.1f}m)")
    makespan = summary.get("makespan")
    if makespan is not None:
        emit(f"  Makespan:          {makespan:.1f}s ({makespan/60:.1f}m)")
    else:
        emit(f"  Makespan:          N/A")
    emit(f"  Total run time:    {summary.get('total_run_time', 0):.1f}s")
    emit(f"  Total wait time:   {summary.get('total_wait_time', 0):.1f}s")
    emit(f"  Avg run time:      {summary.get('avg_run_time', 0):.1f}s")
    emit(f"  Avg wait time:     {summary.get('avg_wait_time', 0):.1f}s")
    avg_jct = summary.get('avg_run_time', 0) + summary.get('avg_wait_time', 0)
    emit(f"  Avg Job Completion Time:           {avg_jct:.1f}s")
    if summary.get("jobs"):
        max_wait = max(j["wait_time"] for j in summary["jobs"])
        emit(f"  Max wait time:     {max_wait:.1f}s")

    # Jain's fairness index on wait times: J = (Σx)² / (n · Σx²)
    # J = 1.0 means perfectly fair (all jobs wait equally).
    if summary.get("jobs"):
        waits = [j["wait_time"] for j in summary["jobs"]]
        n = len(waits)
        sum_x = sum(waits)
        sum_x2 = sum(w * w for w in waits)
        jfi = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0
        emit(f"  Jain's fairness:   {jfi:.4f}")

    # Slowdown: (wait + run) / run per job. 1.0 = no waiting.
    if summary.get("jobs"):
        slowdowns = []
        for j in summary["jobs"]:
            if j["run_time"] > 0:
                slowdowns.append((j["wait_time"] + j["run_time"]) / j["run_time"])
        if slowdowns:
            avg_slowdown = sum(slowdowns) / len(slowdowns)
            max_slowdown = max(slowdowns)
            emit(f"  Avg slowdown:      {avg_slowdown:.2f}x")
            emit(f"  Max slowdown:      {max_slowdown:.2f}x")

    if "gpu" in summary:
        gpu = summary["gpu"]
        emit(f"\n  GPU Utilization:")
        emit(f"    Avg GPU util:    {gpu['avg_gpu_util_pct']}%")
        emit(f"    Avg mem util:    {gpu['avg_mem_util_pct']}%")
        emit(f"    Peak mem used:   {gpu['peak_mem_used_mb']} MB")
        emit(f"    Avg mem used:    {gpu['avg_mem_used_mb']} MB")
        emit(f"    Avg power draw:  {gpu['avg_power_w']} W")
        emit(f"    Samples:         {gpu['num_samples']}")

    # GPU-hours wasted: total GPU-seconds allocated minus useful GPU-seconds.
    # Uses per-job GPU allocations for total, and avg utilization for useful fraction.
    if summary.get("jobs") and "gpu" in summary:
        total_gpu_sec = sum(j["gpus"] * j["run_time"] for j in summary["jobs"])
        idle_frac = 1.0 - (summary["gpu"]["avg_gpu_util_pct"] / 100.0)
        wasted_gpu_sec = total_gpu_sec * idle_frac
        emit(f"\n  GPU-hours wasted:  {wasted_gpu_sec / 3600:.2f}h "
             f"({total_gpu_sec / 3600:.2f}h allocated, "
             f"{(total_gpu_sec - wasted_gpu_sec) / 3600:.2f}h useful)")

    # GPU efficiency: g^(k-1) per job. 1.0 = perfect linear scaling.
    efficiencies = []
    if summary.get("jobs"):
        for j in summary["jobs"]:
            if j.get("k") is not None and j["gpus"] > 0:
                eff = j["gpus"] ** (j["k"] - 1)
                efficiencies.append(eff)
        if efficiencies:
            avg_eff = sum(efficiencies) / len(efficiencies)
            emit(f"\n  Avg GPU efficiency: {avg_eff:.2%}")

    emit(f"\n  Per-job breakdown:")
    eff_idx = 0
    for j in summary["jobs"]:
        eff_str = ""
        if j.get("k") is not None and j["gpus"] > 0:
            eff = j["gpus"] ** (j["k"] - 1)
            eff_str = f"  eff={eff:.0%}"
        emit(f"    {j['name']:30s}  {j['gpus']} GPUs  "
             f"run={j['run_time']:7.1f}s  wait={j['wait_time']:6.1f}s"
             f"with eff: {eff_str}")
    emit(f"{'='*50}")

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{label}_{timestamp}.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nResults saved to {out_path}")