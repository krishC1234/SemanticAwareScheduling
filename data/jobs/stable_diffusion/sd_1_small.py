#!/usr/bin/env python3
"""Stable Diffusion UNet - batch=1, small params (~50M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1
BASE_CHANNELS = 128
CHANNEL_MULT = (1, 2, 4, 4)
ATTN_RESOLUTIONS = (16, 8)
NUM_RES_BLOCKS = 1

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 200  # Smaller due to heavy model
LATENT_SIZE = 64  # 512 // 8
LATENT_CHANNELS = 4
CONTEXT_DIM = 768  # Text encoder dim
TIME_EMB_DIM = 256
NUM_HEADS = 8

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        context = context if context is not None else x
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = [t.view(b, -1, self.heads, t.shape[-1] // self.heads).transpose(1, 2) for t in (q, k, v)]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, -1, q.shape[-1] * self.heads)
        out = self.to_out(out)
        return out.permute(0, 2, 1).view(b, c, h, w)

class SpatialTransformer(nn.Module):
    def __init__(self, channels, context_dim, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.attn = CrossAttention(channels, context_dim, heads)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        h = self.proj_in(self.norm(x))
        h = self.attn(h, context)
        return x + self.proj_out(h)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_res, attn=False, context_dim=None, downsample=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res):
            self.res_blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, time_emb_dim))
            self.attn_blocks.append(SpatialTransformer(out_ch, context_dim) if attn else nn.Identity())
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1) if downsample else None

    def forward(self, x, t_emb, context):
        outputs = []
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x, t_emb)
            x = attn(x, context) if not isinstance(attn, nn.Identity) else x
            outputs.append(x)
        if self.downsample:
            x = self.downsample(x)
            outputs.append(x)
        return x, outputs

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch, time_emb_dim, num_res, attn=False, context_dim=None, upsample=True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res):
            ch_in = in_ch + skip_ch if i == 0 else out_ch + skip_ch
            self.res_blocks.append(ResBlock(ch_in, out_ch, time_emb_dim))
            self.attn_blocks.append(SpatialTransformer(out_ch, context_dim) if attn else nn.Identity())
        self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1) if upsample else None

    def forward(self, x, skips, t_emb, context):
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = torch.cat([x, skips.pop()], dim=1)
            x = res(x, t_emb)
            x = attn(x, context) if not isinstance(attn, nn.Identity) else x
        if self.upsample:
            x = self.upsample(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch, ch_mult, num_res_blocks, attn_resolutions, context_dim, time_emb_dim):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_ch, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim)
        )
        channels = [base_ch * m for m in ch_mult]
        self.down_blocks = nn.ModuleList()
        in_ch_curr = base_ch
        curr_res = LATENT_SIZE
        for i, out_ch in enumerate(channels):
            attn = curr_res in attn_resolutions
            self.down_blocks.append(DownBlock(in_ch_curr, out_ch, time_emb_dim, num_res_blocks, attn, context_dim, i < len(channels) - 1))
            in_ch_curr = out_ch
            if i < len(channels) - 1: curr_res //= 2

        self.mid_block1 = ResBlock(channels[-1], channels[-1], time_emb_dim)
        self.mid_attn = SpatialTransformer(channels[-1], context_dim)
        self.mid_block2 = ResBlock(channels[-1], channels[-1], time_emb_dim)

        self.up_blocks = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            in_ch = channels[-1] if i == 0 else channels[-i]
            skip_ch = out_ch
            attn = curr_res in attn_resolutions
            self.up_blocks.append(UpBlock(in_ch, out_ch, skip_ch, time_emb_dim, num_res_blocks, attn, context_dim, i < len(channels) - 1))
            if i < len(channels) - 1: curr_res *= 2

        self.out_norm = nn.GroupNorm(32, base_ch)
        self.conv_out = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x, t, context):
        t_emb = self.time_mlp(timestep_embedding(t, self.conv_in.out_channels))
        x = self.conv_in(x)
        skips = []
        for block in self.down_blocks:
            x, skip_outputs = block(x, t_emb, context)
            skips.extend(skip_outputs)
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t_emb)
        for block in self.up_blocks:
            x = block(x, skips, t_emb, context)
        return self.conv_out(F.silu(self.out_norm(x)))

class SyntheticDiffusionDataset(Dataset):
    def __init__(self, size, latent_size, latent_ch, context_dim, seq_len=77):
        self.size, self.latent_size, self.latent_ch = size, latent_size, latent_ch
        self.context_dim, self.seq_len = context_dim, seq_len
    def __len__(self): return self.size
    def __getitem__(self, i):
        latent = torch.randn(self.latent_ch, self.latent_size, self.latent_size)
        noise = torch.randn_like(latent)
        t = torch.randint(0, 1000, (1,)).float() / 1000
        context = torch.randn(self.seq_len, self.context_dim)
        return latent, noise, t, context

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = UNet(LATENT_CHANNELS, LATENT_CHANNELS, BASE_CHANNELS, CHANNEL_MULT,
                 NUM_RES_BLOCKS, ATTN_RESOLUTIONS, CONTEXT_DIM, TIME_EMB_DIM).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"diffusion_unet","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"sd_1_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticDiffusionDataset(NUM_SAMPLES, LATENT_SIZE, LATENT_CHANNELS, CONTEXT_DIM)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    mse = nn.MSELoss()

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for latent, noise, t, context in loader:
            latent, noise, t, context = latent.to(dev), noise.to(dev), t.to(dev).squeeze(), context.to(dev)
            noisy = latent + noise * t[:, None, None, None]
            optim.zero_grad()
            pred = model(noisy, t, context)
            loss = mse(pred, noise)
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
