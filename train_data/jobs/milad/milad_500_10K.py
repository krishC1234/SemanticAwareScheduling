#!/usr/bin/env python3
"""MILAD (Molecular Dynamics Force Field) - batch=500, ~10K params"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from torch.func import jacrev, vmap
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 500
HIDDEN_DIM = 16
N_LAYERS = 4
EPOCHS = 3
NUM_SAMPLES = 10000
LR = 1e-3

# Lennard-Jones parameters
SIGMA = 0.5
EPSILON = 4.0

def lennard_jones(r):
    return EPSILON * ((SIGMA / r) ** 12 - (SIGMA / r) ** 6)

def lennard_jones_force(r):
    return -EPSILON * ((-12 * SIGMA**12 / r**13) + (6 * SIGMA**6 / r**7))

class ForceFieldMLP(nn.Module):
    """MLP for learning interatomic potentials"""
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, HIDDEN_DIM), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers.extend([nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()])
        layers.append(nn.Linear(HIDDEN_DIM, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def make_prediction(model, drs):
    """Compute energies and forces using vmap+jacrev"""
    norms = torch.linalg.norm(drs, dim=1).reshape(-1, 1)
    energies = model(norms)
    # Use vmap+jacrev to compute per-sample Jacobians
    network_derivs = vmap(jacrev(model))(norms).squeeze(-1)
    forces = -network_derivs * drs / norms
    return energies, forces

def loss_fn(energies, forces, pred_energies, pred_forces):
    return F.mse_loss(energies, pred_energies) + 0.01 * F.mse_loss(forces, pred_forces) / 3

class MILADDataset(Dataset):
    """Dataset of interatomic distances and LJ energies/forces"""
    def __init__(self, sz, device):
        self.sz = sz
        # Generate random distances in valid LJ range
        r = torch.linspace(0.5, 2 * SIGMA, steps=sz)
        # Create 3D displacement vectors (pointing along random directions)
        directions = F.normalize(torch.randn(sz, 3), dim=1)
        self.drs = directions * r.unsqueeze(1)
        # Compute ground truth energies and forces
        norms = r.reshape(-1, 1)
        self.energies = torch.stack([lennard_jones(n) for n in norms]).reshape(-1, 1)
        self.forces = torch.stack([lennard_jones_force(n) * d for n, d in zip(norms, self.drs)])
    def __len__(self): return self.sz
    def __getitem__(self, i):
        return self.drs[i], self.energies[i], self.forces[i]

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = ForceFieldMLP().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"milad_forcefield","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"milad_500_10K | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = MILADDataset(NUM_SAMPLES, dev)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for drs, energies, forces in loader:
            drs, energies, forces = drs.to(dev), energies.to(dev), forces.to(dev)
            opt.zero_grad()
            pred_e, pred_f = make_prediction(model.module, drs)
            loss = loss_fn(energies, forces, pred_e, pred_f)
            loss.backward()
            opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
