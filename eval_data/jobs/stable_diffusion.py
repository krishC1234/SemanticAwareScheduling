#!/usr/bin/env python3
"""Stable Diffusion 2 UNet - denoising diffusion training, batch=1, ~865M params

Implements the SD2 UNet (noise prediction network) with DDPM training:
  1. Sample random timestep t
  2. Add Gaussian noise to latent at timestep t
  3. UNet predicts the noise conditioned on t and text embeddings
  4. MSE loss between predicted and actual noise

Architecture: encoder-decoder UNet with ResBlocks, cross-attention
(text conditioning), and sinusoidal time embeddings. Channel mults
[1, 2, 4, 4] with base=320, attention at 32x32, 16x16, 8x8.

Reference: Rombach et al., "High-Resolution Image Synthesis with
Latent Diffusion Models", CVPR 2022
"""
import time, json, math, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1

# === FIXED ===
EPOCHS = 10
NUM_SAMPLES = 10000
LATENT_SIZE = 64       # SD2 VAE encodes 512x512 → 64x64
LATENT_CH = 4          # VAE latent channels
CONTEXT_DIM = 1024     # SD2 uses OpenCLIP ViT-H (1024-d)
CONTEXT_LEN = 77       # CLIP max token length
BASE_CH = 320
CH_MULT = [1, 2, 4, 4]
NUM_RES_BLOCKS = 2
ATTN_RESOLUTIONS = [32, 16, 8]
NUM_HEADS = 8
TIMESTEPS = 1000

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = t[:, None].float() * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimeMLPBlock(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_dim),
        )

    def forward(self, t):
        return self.mlp(t)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = TimeMLPBlock(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        B, L, D = x.shape
        q = self.to_q(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, self.heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.to_out(out)


class SpatialTransformer(nn.Module):
    """Self-attention + cross-attention block operating on spatial features."""
    def __init__(self, channels, context_dim, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.self_attn = CrossAttention(channels, channels, heads)
        self.cross_attn = CrossAttention(channels, context_dim, heads)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.norm3 = nn.LayerNorm(channels)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x))
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        h = h + self.self_attn(self.norm1(h), self.norm1(h))
        h = h + self.cross_attn(self.norm2(h), context)
        h = h + self.ff(self.norm3(h))
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + self.proj_out(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = BASE_CH * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmb(BASE_CH),
            nn.Linear(BASE_CH, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_conv = nn.Conv2d(LATENT_CH, BASE_CH, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = BASE_CH
        channels = [ch]
        res = LATENT_SIZE
        for i, mult in enumerate(CH_MULT):
            out_ch = BASE_CH * mult
            block = nn.ModuleList()
            for _ in range(NUM_RES_BLOCKS):
                block.append(ResBlock(ch, out_ch, time_dim))
                if res in ATTN_RESOLUTIONS:
                    block.append(SpatialTransformer(out_ch, CONTEXT_DIM, NUM_HEADS))
                ch = out_ch
                channels.append(ch)
            self.down_blocks.append(block)
            if i < len(CH_MULT) - 1:
                self.down_samples.append(Downsample(ch))
                channels.append(ch)
                res //= 2
            else:
                self.down_samples.append(nn.Identity())

        # Middle
        self.mid_res1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SpatialTransformer(ch, CONTEXT_DIM, NUM_HEADS)
        self.mid_res2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in enumerate(reversed(CH_MULT)):
            out_ch = BASE_CH * mult
            block = nn.ModuleList()
            for j in range(NUM_RES_BLOCKS + 1):
                skip_ch = channels.pop()
                block.append(ResBlock(ch + skip_ch, out_ch, time_dim))
                if res in ATTN_RESOLUTIONS:
                    block.append(SpatialTransformer(out_ch, CONTEXT_DIM, NUM_HEADS))
                ch = out_ch
            self.up_blocks.append(block)
            if i < len(CH_MULT) - 1:
                self.up_samples.append(Upsample(ch))
                res *= 2
            else:
                self.up_samples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, LATENT_CH, 3, padding=1)

    def forward(self, x, t, context):
        t_emb = self.time_embed(t)
        h = self.input_conv(x)

        # Encoder (save skip connections)
        skips = [h]
        for block, down in zip(self.down_blocks, self.down_samples):
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                else:
                    h = layer(h, context)
                skips.append(h)
            if not isinstance(down, nn.Identity):
                h = down(h)
                skips.append(h)

        # Middle
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h, context)
        h = self.mid_res2(h, t_emb)

        # Decoder
        for block, up in zip(self.up_blocks, self.up_samples):
            for layer in block:
                if isinstance(layer, ResBlock):
                    h = layer(torch.cat([h, skips.pop()], dim=1), t_emb)
                else:
                    h = layer(h, context)
            if not isinstance(up, nn.Identity):
                h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Dataset & training
# ---------------------------------------------------------------------------
class SyntheticDiffusionDataset(Dataset):
    """Synthetic latents (mimics VAE-encoded images) + random text embeddings."""
    def __init__(self, size):
        self.size = size

    def __len__(self): return self.size

    def __getitem__(self, _):
        latent = torch.randn(LATENT_CH, LATENT_SIZE, LATENT_SIZE)
        context = torch.randn(CONTEXT_LEN, CONTEXT_DIM)
        return latent, context


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = UNet().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "diffusion", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"stable_diffusion | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticDiffusionDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Linear beta schedule for DDPM noise
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS, device=dev)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for latents, context in loader:
            latents, context = latents.to(dev), context.to(dev)
            # Sample random timestep per sample
            t = torch.randint(0, TIMESTEPS, (latents.shape[0],), device=dev)
            noise = torch.randn_like(latents)
            # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
            ab = alpha_bar[t][:, None, None, None]
            noisy = ab.sqrt() * latents + (1 - ab).sqrt() * noise
            # Predict noise
            pred = model(noisy, t, context)
            loss = F.mse_loss(pred, noise)
            optim.zero_grad()
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