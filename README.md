# Semantic-Aware Resource Scheduling on Clusters

A GPU cluster scheduler that predicts each job's multi-GPU scaling efficiency from its training script source code and uses this prediction to make informed GPU allocation decisions.

## Overview

Modern GPU schedulers like SLURM require users to manually specify GPU counts, leading to systematic over- or under-provisioning. This system removes that burden by:

1. **LLM Feature Extraction** -- Claude Sonnet reads submitted training scripts and extracts model family, batch size, and parameter count
2. **Scaling Exponent Prediction** -- A trained RandomForest regressor predicts the scaling exponent *k* from these features
3. **Greedy Marginal-Gain Allocation** -- GPUs are distributed across concurrent jobs to maximize cluster-wide throughput based on each job's predicted scaling behavior

## Repository Structure

```
.
├── scheduler/                  # Semantic-aware scheduler server
│   ├── main.py                 # Server process (TCP listener + scheduling loop)
│   ├── job_profiler.py         # LLM feature extraction + k prediction
│   ├── job.py                  # Job data class
│   ├── queue.py                # Priority queue with greedy GPU allocator
│   ├── scorer.py               # Marginal-gain scoring function
│   ├── slurm_monitor.py        # SLURM state polling (sinfo/squeue)
│   ├── sbatch_wrapper.py       # Generates and submits sbatch scripts
│   ├── submit.py               # Client for submitting jobs to the scheduler
│   └── logger.py               # Structured logging
├── model/                      # Scaling exponent predictor
│   ├── model.py                # Train/evaluate k-predictor (Ridge, RF, GB, MLP)
│   ├── best_model.joblib       # Saved RandomForest model
│   ├── feature_columns.json    # Feature schema
│   └── results.txt             # Cross-validation results
├── train_data/                 # Benchmark data collection
│   ├── jobs/                   # 196 PyTorch DDP training scripts (57 model families x 4 configs)
│   ├── benchmark.csv           # Aggregated benchmark results
│   └── scaling_dataset.csv     # Fitted scaling exponents per config
├── evaluation/                 # Evaluation framework
│   ├── jobs/                   # 21 held-out eval workloads
│   ├── test_scripts/           # Scheduling algorithms (scheduler, greedy, polite, FCFS-split, size-aware)
│   ├── metrics.py              # GPU sampling + job timing metrics
│   ├── report.py               # Results reporting and file output
│   └── test_results/           # Output results per run
├── shell_scripts/              # Automation
│   ├── setup.sh                # SLURM cluster provisioning
│   ├── run_eval.sh             # Single evaluation run
│   ├── run_full_eval.sh        # Multi-delay evaluation sweep
│   ├── run_contention_eval.sh  # Multi-contention evaluation sweep
│   ├── quickstart.sh           # One-command setup + run
│   ├── fetch_eval.sh           # Pull results from remote machine
│   └── run_benchmark.sh        # Benchmark data collection
├── paper/                      # Research paper (LaTeX)
│   ├── main.tex
│   └── references.bib
├── build_scaling_dataset.py    # Fits k from benchmark.csv
├── model_families.json         # Model family taxonomy (13 categories)
└── requirements.txt
```

## Quick Start

### Prerequisites
- Linux with NVIDIA GPUs
- SLURM (installed by `setup.sh`)
- Python 3.10+ with PyTorch, NCCL
- Anthropic API key (for LLM feature extraction)

### Setup and Run

```bash
# Clone and setup
git clone https://github.com/krishC1234/SemanticAwareScheduling.git
cd SemanticAwareScheduling
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
bash shell_scripts/setup.sh

# Run evaluation
nohup bash shell_scripts/run_contention_eval.sh 9100 > eval_contention.log 2>&1 &
```

Or use the quickstart script:
```bash
bash shell_scripts/quickstart.sh
```

### Submit Jobs to the Scheduler

```bash
# Start the scheduler server
python3 -m scheduler.main --port 9100

# Submit training scripts (in another terminal)
python3 -m scheduler.submit my_training_script.py
python3 -m scheduler.submit path/to/scripts/
```

Users submit training scripts without specifying GPU counts. The scheduler automatically determines the optimal allocation.

## Scaling Model

Training time scales with GPU count as:

```
T(g) = a * g^(-k)
```

where *k* is the scaling exponent (0 to 1). The greedy allocator assigns GPUs one at a time to the job with the highest marginal gain:

```
marginal_gain(g, k) = g^(-k) - (g+1)^(-k)
```

This naturally gives more GPUs to jobs that benefit from them (high *k*) and fewer to jobs with diminishing returns (low *k*).

## Evaluation

The evaluation framework compares the scheduler against four baselines:

| Strategy | Description |
|----------|-------------|
| **Greedy** | All GPUs per job, sequential execution |
| **Polite** | 1 GPU per job, maximum concurrency |
| **FCFS-Split** | Equal GPU share among pending jobs |
| **Size-Aware** | GPU tiers based on parameter count |

Evaluated across varying contention levels (3, 5, 12, 22 jobs) and inter-arrival delays (0s, 15s, 30s).

## Benchmark Corpus

- 57 model families across 13 categories (CNN, Transformer, GAN, GNN, RL, Diffusion, etc.)
- 4 configurations per family (2x batch size x 2x model size)
- All models sourced from [PyTorch Benchmark](https://github.com/pytorch/benchmark) and adapted for DDP
- 787 total benchmark runs across GPU counts {1, 2, 4, 8}

## Results

At low contention (3 jobs, 8 GPUs), the scheduler achieves:
- **50% lower makespan** vs polite baseline
- **28% lower avg JCT** vs polite baseline
- **0.998 Jain's fairness** (near-perfect)
