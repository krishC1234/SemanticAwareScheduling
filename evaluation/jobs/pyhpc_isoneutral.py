#!/usr/bin/env python3
"""PyHPC Isoneutral Mixing - ocean physics simulation, batch=256, ~0.05M params

Isoneutral diffusion is a key computation in ocean general circulation
models. This implements the diffusion pre-computation step as a learnable
surrogate: a small MLP that maps grid-cell physical state (masks,
coordinates, temperature, salinity) to diffusion coefficients.

The original benchmark is a pure compute kernel (no learnable params).
Here we wrap it as a trainable MLP to match the DDP training pattern
while preserving the compute-heavy, memory-light profile of HPC workloads.

Reference: Häfner et al., "PyHPC Benchmarks", 2021
"""
import time, json, math, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 256

# === FIXED ===
EPOCHS = 50
NUM_SAMPLES = 10000
INPUT_DIM = 23           # physical state: masks, coords, temp, salt, etc.
HIDDEN = 128
OUTPUT_DIM = 4           # diffusion coefficients (Ai_ez, Ai_nz, Ai_bx, Ai_by)
GRID_POINTS = 512        # points per sample (spatial grid)

# ---------------------------------------------------------------------------
# Surrogate MLP for isoneutral diffusion
# ---------------------------------------------------------------------------
class IsoneutralMLP(nn.Module):
    """Small MLP mapping physical grid state → diffusion coefficients.
    ~0.05M trainable parameters.

    Input per grid point: 23 features (masks, coordinates, T, S, K values)
    Output per grid point: 4 diffusion coefficient values
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, OUTPUT_DIM),
        )

    def forward(self, x):
        """x: (B, grid_points, input_dim) → (B, grid_points, output_dim)"""
        return self.net(x)


class SyntheticOceanDataset(Dataset):
    """Synthetic ocean grid data mimicking isoneutral mixing inputs.
    Each sample = a patch of GRID_POINTS ocean cells with physical state."""

    def __init__(self, size):
        self.size = size

    def __len__(self): return self.size

    def __getitem__(self, _):
        # 23 input features per grid point: masks (4), coordinates (9),
        # temperature/salinity (6), diffusion coeffs (4)
        x = torch.randn(GRID_POINTS, INPUT_DIM)
        # Target: synthetic diffusion coefficients
        y = torch.randn(GRID_POINTS, OUTPUT_DIM)
        return x, y


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = IsoneutralMLP().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "mlp", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"pyhpc_isoneutral | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticOceanDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            optim.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            optim.step()
        tsp += len(ds)
        if rank == 0:
            print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | "
                  f"throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | "
              f"Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###")
        print(json.dumps({"batch_size": BATCH_SIZE, "param_count": pc,
                           "gpu_count": ws, "total_time_sec": round(tt, 2),
                           "avg_throughput": round(tsp / tt, 1)}))
        print("###END_RESULTS###")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()