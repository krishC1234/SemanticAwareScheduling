import heapq
import time
from scheduler.scorer import marginal_gain
from scheduler.poller import get_total_gpus

MAX_GPU_PER_JOB = get_total_gpus()

class Queue:
    def __init__(self, lam=0.001):
        self.lam = lam
        self.heap = []
        self.counter = 0  # tiebreaker so heapq never compares Job objects

    def add_job(self, job):
        score = self._score(job)
        heapq.heappush(self.heap, (-score, self.counter, job))
        self.counter += 1

    def allocate(self, available_gpus):
        """Distribute available_gpus across queued jobs using the greedy scorer.

        Pops the best job, gives it +1 GPU, re-scores, pushes it back.
        Repeats until no GPUs left or all jobs are maxed out.

        Returns a list of (job, gpu_count) for jobs that got >= 1 GPU.
        """

        assigned = {}
        for _ in range(available_gpus):
            job = self._pop_valid()
            if job is None:
                break 
            job.assigned_gpus += 1
            assigned[id(job)] = job

            if job.assigned_gpus < MAX_GPU_PER_JOB:
                new_score = self._score(job)
                heapq.heappush(self.heap, (-new_score, self.counter, job))
                self.counter += 1

        # Mark allocated jobs so stale heap entries get skipped on future pops
        for job in assigned.values():
            job.submitted = True

        return [(job, job.assigned_gpus) for job in assigned.values()]

    def _pop_valid(self):
        """Pop entries, skipping any that were already submitted (lazy delete)."""
        while self.heap:
            neg_score, cnt, job = heapq.heappop(self.heap)
            if not getattr(job, "submitted", False):
                return job
        return None

    def _score(self, job):
        wait = time.time() - job.start_time
        return marginal_gain(job.assigned_gpus, job.k) + self.lam * wait

    def __len__(self):
        return len(self.heap)