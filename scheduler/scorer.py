"""Scoring math for the greedy GPU allocator.

Scaling model:
    T(g) = a * g^(-k)    →  normalized: T(g)/T(1) = g^(-k)

Marginal gain of going from g to g+1 GPUs:
    g^(-k) - (g+1)^(-k)

The queue calls marginal_gain() on each pop/push cycle to re-score jobs
as their assigned GPU count changes.
"""


def marginal_gain(current_gpus: int, k: float) -> float:
    """Normalized time saved by going from current_gpus to current_gpus + 1.

    If current_gpus == 0, the job goes from not-running to running on 1 GPU.
    The gain is 1.0 (a full single-GPU runtime saved — the maximum unit).
    """
    if current_gpus == 0:
        return 1.0
    return current_gpus ** (-k) - (current_gpus + 1) ** (-k)