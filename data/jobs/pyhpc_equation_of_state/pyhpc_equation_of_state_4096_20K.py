#!/usr/bin/env python3
"""PyHPC Equation of State (PINN) - batch=4096, ~20K params"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 4096
HIDDEN_DIM = 64
N_LAYERS = 4
EPOCHS = 3
NUM_SAMPLES = 100000
LR = 1e-3

# TEOS-10 polynomial coefficients (simplified GSW)
V00 = 1.0769995862e-3
V01 = -6.0799143809e-5
V02 = 9.9856169219e-6
V03 = -1.1309361437e-6
V04 = 6.4464345422e-8
V05 = -3.1849756977e-9

def gsw_rho(s, t, p):
    """Simplified seawater density from TEOS-10"""
    t2 = t * t
    t3 = t2 * t
    s_root = torch.sqrt(torch.clamp(s, min=0))
    rho = (999.842594 + 6.793952e-2 * t - 9.095290e-3 * t2 + 
           1.001685e-4 * t3 - 1.120083e-6 * t2 * t2 +
           s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t2) +
           s_root * s * (-5.72466e-3 + 1.0227e-4 * t) +
           p * (4.8314e-4 * s - 1.6546e-6 * t))
    return rho

def gsw_dHdT(s, t, p):
    """Derivative of enthalpy w.r.t temperature (specific heat capacity)"""
    cp0 = 4217.4
    t2 = t * t
    s_root = torch.sqrt(torch.clamp(s, min=0))
    cp = (cp0 - 7.6444 * t + 0.1779 * t2 - 0.0045 * t2 * t +
          s * (-9.31 + 0.32 * t - 0.004 * t2) +
          s_root * s * (0.15 - 0.001 * t) +
          p * (0.0004 - 0.000002 * t))
    return cp

class EOSMLP(nn.Module):
    """MLP to learn equation of state"""
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(3, HIDDEN_DIM), nn.SiLU()]
        for _ in range(N_LAYERS - 1):
            layers.extend([nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.SiLU()])
        layers.append(nn.Linear(HIDDEN_DIM, 1))
        self.net = nn.Sequential(*layers)
        # Normalization constants
        self.register_buffer('s_mean', torch.tensor(5.0))
        self.register_buffer('s_std', torch.tensor(3.0))
        self.register_buffer('t_mean', torch.tensor(4.0))
        self.register_buffer('t_std', torch.tensor(10.0))
        self.register_buffer('p_mean', torch.tensor(500.0))
        self.register_buffer('p_std', torch.tensor(300.0))
        self.register_buffer('y_mean', torch.tensor(4000.0))
        self.register_buffer('y_std', torch.tensor(200.0))
    def forward(self, s, t, p):
        # Normalize inputs
        s_n = (s - self.s_mean) / self.s_std
        t_n = (t - self.t_mean) / self.t_std
        p_n = (p - self.p_mean) / self.p_std
        x = torch.stack([s_n.flatten(), t_n.flatten(), p_n.flatten()], dim=-1)
        y_n = self.net(x)
        # Denormalize output
        y = y_n * self.y_std + self.y_mean
        return y.view_as(s)

class EOSDataset(Dataset):
    """Dataset of (s, t, p) inputs with ground truth dH/dT"""
    def __init__(self, sz):
        self.sz = sz
        # Generate oceanographic parameter ranges
        self.s = torch.rand(sz) * 10 + 0.01  # Salinity: 0.01-10 g/kg
        self.t = torch.rand(sz) * 32 - 12    # Temperature: -12 to 20 C
        self.p = torch.rand(sz) * 1000       # Pressure: 0-1000 dbar
        # Compute ground truth
        self.target = gsw_dHdT(self.s, self.t, self.p)
    def __len__(self): return self.sz
    def __getitem__(self, i):
        return self.s[i], self.t[i], self.p[i], self.target[i]

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = EOSMLP().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"pyhpc_eos_pinn","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"pyhpc_equation_of_state_4096_20K | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = EOSDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for s, t, p, target in loader:
            s, t, p, target = s.to(dev), t.to(dev), p.to(dev), target.to(dev)
            opt.zero_grad()
            pred = model(s, t, p)
            loss = F.mse_loss(pred, target)
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
