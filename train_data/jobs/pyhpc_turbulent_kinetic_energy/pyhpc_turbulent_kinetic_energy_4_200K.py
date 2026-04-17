#!/usr/bin/env python3
"""PyHPC Turbulent Kinetic Energy (Neural Surrogate) - batch=4, ~200K params"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 4
HIDDEN_CH = 64
N_LAYERS = 6
GRID_SIZE = 32
EPOCHS = 3
NUM_SAMPLES = 200
LR = 1e-3

def compute_tke_tendency(u, v, w, tke, kappaM, dz=1.0):
    dudz = torch.diff(u, dim=-1, prepend=u[..., :1]) / dz
    dvdz = torch.diff(v, dim=-1, prepend=v[..., :1]) / dz
    shear_sq = dudz**2 + dvdz**2
    production = kappaM * shear_sq
    c_eps = 0.7
    mxl = torch.clamp(tke, min=1e-6) ** 0.5 * 10
    dissipation = c_eps * torch.clamp(tke, min=0) ** 1.5 / torch.clamp(mxl, min=1e-6)
    d2tke_dz2 = (torch.roll(tke, -1, dims=-1) + torch.roll(tke, 1, dims=-1) - 2*tke) / dz**2
    diffusion = kappaM * d2tke_dz2
    dtke = production - dissipation + diffusion
    return dtke

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
    def forward(self, x):
        return F.silu(self.norm(self.conv(x)))

class TKENet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList([
            Conv3DBlock(5, HIDDEN_CH),
            *[Conv3DBlock(HIDDEN_CH, HIDDEN_CH) for _ in range(N_LAYERS - 2)],
            nn.Conv3d(HIDDEN_CH, 1, 3, padding=1)
        ])
    def forward(self, u, v, w, tke, kappaM):
        x = torch.stack([u, v, w, tke, kappaM], dim=1)
        for layer in self.encoder[:-1]:
            x = layer(x)
        x = self.encoder[-1](x)
        return x.squeeze(1)

class TKEDataset(Dataset):
    def __init__(self, sz):
        self.sz = sz
        self.cache = {}
    def __len__(self): return self.sz
    def __getitem__(self, i):
        if i not in self.cache:
            u = torch.randn(GRID_SIZE, GRID_SIZE, GRID_SIZE) * 0.1
            v = torch.randn(GRID_SIZE, GRID_SIZE, GRID_SIZE) * 0.1
            w = torch.randn(GRID_SIZE, GRID_SIZE, GRID_SIZE) * 0.01
            tke = torch.rand(GRID_SIZE, GRID_SIZE, GRID_SIZE) * 1e-4 + 1e-6
            kappaM = torch.rand(GRID_SIZE, GRID_SIZE, GRID_SIZE) * 0.1 + 0.01
            dtke = compute_tke_tendency(u, v, w, tke, kappaM)
            self.cache[i] = (u, v, w, tke, kappaM, dtke)
        return self.cache[i]

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = TKENet().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"pyhpc_tke_surrogate","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"pyhpc_turbulent_kinetic_energy_4_200K | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = TKEDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for u, v, w, tke, kappaM, dtke_gt in loader:
            u, v, w = u.to(dev), v.to(dev), w.to(dev)
            tke, kappaM, dtke_gt = tke.to(dev), kappaM.to(dev), dtke_gt.to(dev)
            opt.zero_grad()
            dtke_pred = model(u, v, w, tke, kappaM)
            loss = F.mse_loss(dtke_pred, dtke_gt)
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
