#!/usr/bin/env python3
"""Super-SloMo (Video Frame Interpolation) - batch=8, large params (~20M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 8
BASE_CHANNELS = 64
EPOCHS = 3
NUM_SAMPLES = 500
IMG_SIZE = 128
IN_CHANNELS = 3
LR = 1e-4

def conv(ic, oc, k=3, s=1, p=1): return nn.Sequential(nn.Conv2d(ic, oc, k, s, p), nn.LeakyReLU(0.1, True))

class Encoder(nn.Module):
    def __init__(self, ic, bc):
        super().__init__()
        self.c1 = nn.Sequential(conv(ic, bc, 7, 1, 3), conv(bc, bc, 7, 1, 3))
        self.c2 = nn.Sequential(conv(bc, bc*2, 5, 2, 2), conv(bc*2, bc*2, 5, 1, 2))
        self.c3 = nn.Sequential(conv(bc*2, bc*4, 3, 2, 1), conv(bc*4, bc*4, 3, 1, 1))
        self.c4 = nn.Sequential(conv(bc*4, bc*8, 3, 2, 1), conv(bc*8, bc*8, 3, 1, 1))
        self.c5 = nn.Sequential(conv(bc*8, bc*16, 3, 2, 1), conv(bc*16, bc*16, 3, 1, 1))
    def forward(self, x):
        s1 = self.c1(x); s2 = self.c2(s1); s3 = self.c3(s2); s4 = self.c4(s3); s5 = self.c5(s4)
        return s1, s2, s3, s4, s5

class Decoder(nn.Module):
    def __init__(self, bc, oc):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d4a, self.d4b = conv(bc*16, bc*8), conv(bc*16, bc*8)
        self.d3a, self.d3b = conv(bc*8, bc*4), conv(bc*8, bc*4)
        self.d2a, self.d2b = conv(bc*4, bc*2), conv(bc*4, bc*2)
        self.d1a, self.d1b = conv(bc*2, bc), conv(bc*2, bc)
        self.out = nn.Conv2d(bc, oc, 3, 1, 1)
    def forward(self, s1, s2, s3, s4, s5):
        x = self.d4b(torch.cat([self.d4a(self.up(s5)), s4], 1))
        x = self.d3b(torch.cat([self.d3a(self.up(x)), s3], 1))
        x = self.d2b(torch.cat([self.d2a(self.up(x)), s2], 1))
        x = self.d1b(torch.cat([self.d1a(self.up(x)), s1], 1))
        return self.out(x)

class FlowNet(nn.Module):
    def __init__(self, bc):
        super().__init__()
        self.enc = Encoder(6, bc)
        self.dec = Decoder(bc, 4)
    def forward(self, I0, I1):
        flow = self.dec(*self.enc(torch.cat([I0, I1], 1)))
        return flow[:, :2], flow[:, 2:4]

class InterpNet(nn.Module):
    def __init__(self, bc):
        super().__init__()
        self.enc = Encoder(16, bc)
        self.dec = Decoder(bc, 5)
    def forward(self, I0, I1, Ft0, Ft1, g0, g1):
        out = self.dec(*self.enc(torch.cat([I0, I1, Ft0, Ft1, g0, g1], 1)))
        return out[:, :2], out[:, 2:4], torch.sigmoid(out[:, 4:5])

def backwarp(img, flow):
    B, C, H, W = img.shape
    gy, gx = torch.meshgrid(torch.arange(H, device=img.device), torch.arange(W, device=img.device), indexing='ij')
    grid = torch.stack([gx, gy], -1).float().unsqueeze(0).expand(B, -1, -1, -1)
    fg = grid + flow.permute(0, 2, 3, 1)
    fg[..., 0] = 2.0 * fg[..., 0] / (W - 1) - 1.0
    fg[..., 1] = 2.0 * fg[..., 1] / (H - 1) - 1.0
    return F.grid_sample(img, fg, mode='bilinear', padding_mode='border', align_corners=True)

class SuperSloMo(nn.Module):
    def __init__(self, bc=BASE_CHANNELS):
        super().__init__()
        self.flownet = FlowNet(bc)
        self.interpnet = InterpNet(bc)
    def forward(self, I0, I1, t=0.5):
        F01, F10 = self.flownet(I0, I1)
        Ft0 = -(1 - t) * t * F01 + t * t * F10
        Ft1 = (1 - t) * (1 - t) * F01 - t * (1 - t) * F10
        g0, g1 = backwarp(I0, Ft0), backwarp(I1, Ft1)
        dFt0, dFt1, V = self.interpnet(I0, I1, Ft0, Ft1, g0, g1)
        Ft0r, Ft1r = Ft0 + dFt0, Ft1 + dFt1
        g0r, g1r = backwarp(I0, Ft0r), backwarp(I1, Ft1r)
        It = (V * g0r + (1 - V) * g1r)
        return It, F01, F10, Ft0r, Ft1r

def compute_loss(I0, I1, It_pred, It_gt, F01, F10, Ft0, Ft1):
    l_rec = F.l1_loss(It_pred, It_gt)
    l_warp = F.l1_loss(backwarp(I0, Ft0), It_gt) + F.l1_loss(backwarp(I1, Ft1), It_gt)
    def tv(f): return torch.mean(torch.abs(f[:,:,:,:-1] - f[:,:,:,1:])) + torch.mean(torch.abs(f[:,:,:-1,:] - f[:,:,1:,:]))
    l_smooth = tv(F01) + tv(F10)
    return l_rec + 0.4 * l_warp + 0.01 * l_smooth

class VideoDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i):
        base = torch.rand(IN_CHANNELS, IMG_SIZE, IMG_SIZE)
        I0 = torch.clamp(base - torch.rand_like(base) * 0.3, 0, 1)
        I1 = torch.clamp(base + torch.rand_like(base) * 0.3, 0, 1)
        return I0, I1, base

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = SuperSloMo().to(dev); pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"video_interpolation_superslomo","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"superslomo_8_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = VideoDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for I0, I1, It_gt in loader:
            I0, I1, It_gt = I0.to(dev), I1.to(dev), It_gt.to(dev)
            opt.zero_grad()
            It_pred, F01, F10, Ft0, Ft1 = model(I0, I1)
            loss = compute_loss(I0, I1, It_pred, It_gt, F01, F10, Ft0, Ft1)
            loss.backward(); opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
