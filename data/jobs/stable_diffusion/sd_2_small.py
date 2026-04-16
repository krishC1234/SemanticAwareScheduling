#!/usr/bin/env python3
"""Stable Diffusion UNet - batch=2, small params (~50M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 2
BASE_CHANNELS = 128
CHANNEL_MULT = (1, 2, 4, 4)
ATTN_RESOLUTIONS = (16, 8)
NUM_RES_BLOCKS = 1
EPOCHS = 3
NUM_SAMPLES = 200
LATENT_SIZE = 64
LATENT_CHANNELS = 4
CONTEXT_DIM = 768
TIME_EMB_DIM = 256

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(32, in_ch), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2, self.conv2 = nn.GroupNorm(32, out_ch), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x))) + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        return self.conv2(F.silu(self.norm2(h))) + self.skip(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -0.5
        self.to_q, self.to_k, self.to_v = nn.Linear(query_dim, inner_dim, bias=False), nn.Linear(context_dim, inner_dim, bias=False), nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    def forward(self, x, context):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        q, k, v = self.to_q(x_flat), self.to_k(context), self.to_v(context)
        q, k, v = [t.view(b, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2) for t in (q, k, v)]
        out = torch.matmul((torch.matmul(q, k.transpose(-1, -2)) * self.scale).softmax(dim=-1), v)
        return self.to_out(out.transpose(1, 2).reshape(b, -1, q.shape[-1] * self.heads)).permute(0, 2, 1).view(b, c, h, w)

class SpatialTransformer(nn.Module):
    def __init__(self, ch, ctx_dim, heads=8):
        super().__init__()
        self.norm, self.proj_in, self.attn, self.proj_out = nn.GroupNorm(32, ch), nn.Conv2d(ch, ch, 1), CrossAttention(ch, ctx_dim, heads), nn.Conv2d(ch, ch, 1)
    def forward(self, x, ctx): return x + self.proj_out(self.attn(self.proj_in(self.norm(x)), ctx))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, num_res, attn, ctx_dim, down):
        super().__init__()
        self.res = nn.ModuleList([ResBlock(in_ch if i == 0 else out_ch, out_ch, t_dim) for i in range(num_res)])
        self.attn = nn.ModuleList([SpatialTransformer(out_ch, ctx_dim) if attn else nn.Identity() for _ in range(num_res)])
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if down else None
    def forward(self, x, t, ctx):
        outs = []
        for r, a in zip(self.res, self.attn): x = r(x, t); x = a(x, ctx) if not isinstance(a, nn.Identity) else x; outs.append(x)
        if self.down: x = self.down(x); outs.append(x)
        return x, outs

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, t_dim, num_res, attn, ctx_dim, up):
        super().__init__()
        self.res = nn.ModuleList([ResBlock((in_ch if i == 0 else out_ch) + skip_ch, out_ch, t_dim) for i in range(num_res)])
        self.attn = nn.ModuleList([SpatialTransformer(out_ch, ctx_dim) if attn else nn.Identity() for _ in range(num_res)])
        self.up = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1) if up else None
    def forward(self, x, skips, t, ctx):
        for r, a in zip(self.res, self.attn): x = r(torch.cat([x, skips.pop()], 1), t); x = a(x, ctx) if not isinstance(a, nn.Identity) else x
        return self.up(x) if self.up else x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(LATENT_CHANNELS, BASE_CHANNELS, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.Linear(BASE_CHANNELS, TIME_EMB_DIM), nn.SiLU(), nn.Linear(TIME_EMB_DIM, TIME_EMB_DIM))
        chs = [BASE_CHANNELS * m for m in CHANNEL_MULT]
        self.downs, in_ch, res = nn.ModuleList(), BASE_CHANNELS, LATENT_SIZE
        for i, out_ch in enumerate(chs):
            self.downs.append(DownBlock(in_ch, out_ch, TIME_EMB_DIM, NUM_RES_BLOCKS, res in ATTN_RESOLUTIONS, CONTEXT_DIM, i < len(chs)-1))
            in_ch = out_ch
            if i < len(chs)-1: res //= 2
        self.mid1, self.mid_attn, self.mid2 = ResBlock(chs[-1], chs[-1], TIME_EMB_DIM), SpatialTransformer(chs[-1], CONTEXT_DIM), ResBlock(chs[-1], chs[-1], TIME_EMB_DIM)
        self.ups = nn.ModuleList()
        for i, out_ch in enumerate(reversed(chs)):
            self.ups.append(UpBlock(chs[-1] if i == 0 else chs[-i], out_ch, out_ch, TIME_EMB_DIM, NUM_RES_BLOCKS, res in ATTN_RESOLUTIONS, CONTEXT_DIM, i < len(chs)-1))
            if i < len(chs)-1: res *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, BASE_CHANNELS), nn.SiLU(), nn.Conv2d(BASE_CHANNELS, LATENT_CHANNELS, 3, padding=1))

    def forward(self, x, t, ctx):
        t_emb = self.time_mlp(timestep_embedding(t, BASE_CHANNELS))
        x = self.conv_in(x)
        skips = []
        for d in self.downs: x, s = d(x, t_emb, ctx); skips.extend(s)
        x = self.mid2(self.mid_attn(self.mid1(x, t_emb), ctx), t_emb)
        for u in self.ups: x = u(x, skips, t_emb, ctx)
        return self.out(x)

class SyntheticDiffusionDataset(Dataset):
    def __init__(self, size): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, i):
        return torch.randn(LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), torch.randn(LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), torch.rand(1), torch.randn(77, CONTEXT_DIM)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    model = UNet().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"diffusion_unet","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"sd_2_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticDiffusionDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim, mse = torch.optim.AdamW(model.parameters(), lr=1e-4), nn.MSELoss()
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for lat, noise, t, ctx in loader:
            lat, noise, t, ctx = lat.to(dev), noise.to(dev), t.to(dev).squeeze(), ctx.to(dev)
            optim.zero_grad(); mse(model(lat + noise * t[:, None, None, None], t, ctx), noise).backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
