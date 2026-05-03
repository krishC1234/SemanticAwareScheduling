# CLAUDE.md

This new file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

Research project ("Semantic-Aware RL Batch Job Scheduling on Clusters") whose current contents are the **benchmark data-collection harness** that produces the dataset the scheduler will be trained on. The PDF in the repo root is the writeup. There is no scheduler/RL code in this repo yet — only the workload corpus and the runner that benchmarks each workload across GPU counts.

## Repository layout

- `data/jobs/<model>/<config>.py` — 57 model families × 4 configs each (~228 PyTorch DDP training scripts using synthetic data). Each script is a self-contained `torchrun`-launchable workload.
- `data/setup.sh` — provisions a single-node SLURM cluster (munge + slurmctld + slurmd) on a fresh Ubuntu host and auto-detects CPUs/RAM/GPUs to write `slurm.conf` and `gres.conf`.
- `data/run_benchmark.sh` — iterates every config × `GPU_COUNTS=(1 2 4 8)`, runs it via `torchrun --standalone --nproc_per_node=$N`, samples `nvidia-smi` once per second in parallel, and appends one CSV row per run.
- `data/benchmark.csv` — aggregated results (one row per model/config/gpu_count). Columns: `model,config,batch_size,param_count,gpu_count,total_time_sec,avg_throughput,peak_vram_mb,avg_sm_util_pct,avg_mem_bw_pct`.
- `data/results/` — per-run logs (`gpuN.log`) and raw GPU samples (`gpu_stats_N.csv`). Gitignored.

## Running the benchmark

Both shell scripts hardcode `/workspace/jobs` and `/workspace/results` — they are designed to run inside a container/VM where this repo's `data/` is mounted at `/workspace`. Don't "fix" the paths to `data/...` unless the user is restructuring; instead bind-mount or symlink.

```bash
bash data/setup.sh           # one-time SLURM bring-up (needs root, apt, nvidia-smi)
bash data/run_benchmark.sh   # full sweep; safe to re-run — appends to existing CSV
```

To run a single workload directly (the inner loop the runner executes):

```bash
torchrun --standalone --nproc_per_node=4 data/jobs/alexnet/alexnet_128_small.py
```

## Job script contract

Every script under `data/jobs/` is expected to follow the same protocol — `run_benchmark.sh` parses stdout and breaks if you deviate:

1. Filename encodes the swept axes: `<model>_<batch_size>_<size_variant>.py` (e.g. `bert_16_40M.py`, `alexnet_128_small.py`). The runner derives `config` from the filename stem.
2. Top of file has two clearly marked blocks: `# === VARYING ===` (the two axes that differ across this model's 4 configs) and `# === FIXED ===` (`EPOCHS`, `NUM_SAMPLES`, etc. — keep these consistent across a model family so configs are comparable).
3. Uses `torch.distributed` with NCCL + `DistributedDataParallel`; rank 0 does all printing.
4. Uses a synthetic `Dataset` (no disk/network I/O) so runs are reproducible and bandwidth-independent.
5. Emits two JSON sentinel blocks on rank 0 — the runner greps for these:
   - `###FEATURES###\n{"model_type":..., "batch_size":..., "param_count":...}\n###END_FEATURES###` (printed before training)
   - `###RESULTS###\n{"batch_size":..., "param_count":..., "gpu_count":..., "total_time_sec":..., "avg_throughput":...}\n###END_RESULTS###` (printed after training)
6. Calls `dist.destroy_process_group()` before exit.

When adding a new model, copy an existing family (e.g. `data/jobs/alexnet/`) and keep all four config files structurally identical aside from the two `# === VARYING ===` constants.
