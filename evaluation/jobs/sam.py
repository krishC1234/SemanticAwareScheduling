#!/usr/bin/env python3
"""SAM (Segment Anything Model) ViT-H - image segmentation, batch=1, ~641M params

SAM uses a Vision Transformer (ViT-H) as the image encoder, a prompt
encoder for points/boxes/masks, and a lightweight mask decoder. ViT-H:
32 layers, 1280 hidden, 16 heads, 14×14 window attention with 4 global
attention layers. Trained with a combined focal + dice loss.

Reference: Kirillov et al., "Segment Anything", Meta AI 2023
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
EPOCHS = 5
NUM_SAMPLES = 20000
IMG_SIZE = 1024          # SAM input resolution
PATCH_SIZE = 16          # ViT patch size → 64×64 patches
EMBED_DIM = 1280         # ViT-H
NUM_LAYERS = 32
NUM_HEADS = 16
FFN_RATIO = 4
WINDOW_SIZE = 14
MASK_DIM = 256           # mask decoder output dimension
NUM_MASK_TOKENS = 4

# ---------------------------------------------------------------------------
# ViT-H Image Encoder (simplified SAM encoder)
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, EMBED_DIM, PATCH_SIZE, stride=PATCH_SIZE)
        self.num_patches = (IMG_SIZE // PATCH_SIZE) ** 2

    def forward(self, x):
        # (B, 3, H, W) → (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, E, H/P, W/P)
        return x.flatten(2).transpose(1, 2)


class ViTAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, NUM_HEADS, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, EMBED_DIM)
        return self.proj(out)


class ViTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = ViTAttention()
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * FFN_RATIO),
            nn.GELU(),
            nn.Linear(EMBED_DIM * FFN_RATIO, EMBED_DIM),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    """ViT-H image encoder: ~632M parameters."""
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        grid = IMG_SIZE // PATCH_SIZE
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, EMBED_DIM))
        self.blocks = nn.ModuleList([ViTBlock() for _ in range(NUM_LAYERS)])
        self.neck = nn.Sequential(
            nn.Conv2d(EMBED_DIM, MASK_DIM, 1),
            nn.LayerNorm([MASK_DIM, grid, grid]),
            nn.Conv2d(MASK_DIM, MASK_DIM, 3, padding=1),
            nn.LayerNorm([MASK_DIM, grid, grid]),
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        grid = IMG_SIZE // PATCH_SIZE
        x = x.transpose(1, 2).view(B, EMBED_DIM, grid, grid)
        return self.neck(x)  # (B, MASK_DIM, grid, grid)


# ---------------------------------------------------------------------------
# Lightweight Mask Decoder
# ---------------------------------------------------------------------------
class MaskDecoder(nn.Module):
    """Simplified SAM mask decoder. ~9M parameters."""
    def __init__(self):
        super().__init__()
        self.mask_tokens = nn.Embedding(NUM_MASK_TOKENS, MASK_DIM)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(MASK_DIM, MASK_DIM // 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(MASK_DIM // 2, MASK_DIM // 4, 2, stride=2),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(MASK_DIM // 4, NUM_MASK_TOKENS, 1)

    def forward(self, image_embed):
        """image_embed: (B, MASK_DIM, H, W) → masks: (B, NUM_MASK_TOKENS, 4H, 4W)"""
        x = self.upscale(image_embed)
        return self.mask_head(x)


class SAMModel(nn.Module):
    """SAM: ViT-H encoder + mask decoder. ~641M trainable parameters."""
    def __init__(self):
        super().__init__()
        self.encoder = ImageEncoder()
        self.decoder = MaskDecoder()

    def forward(self, images):
        embed = self.encoder(images)
        masks = self.decoder(embed)
        return masks


class SyntheticSegDataset(Dataset):
    """Synthetic images + binary masks for segmentation training."""
    def __init__(self, size):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, _):
        img = torch.randn(3, IMG_SIZE, IMG_SIZE)
        mask_h = (IMG_SIZE // PATCH_SIZE) * 4  # after 2× upscale twice
        target = torch.randint(0, 2, (NUM_MASK_TOKENS, mask_h, mask_h)).float()
        return img, target


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = SAMModel().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "other", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"sam | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticSegDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for imgs, targets in loader:
            imgs, targets = imgs.to(dev), targets.to(dev)
            optim.zero_grad()
            masks = model(imgs)
            # Resize masks to match target
            masks = F.interpolate(masks, size=targets.shape[-2:], mode='bilinear', align_corners=False)
            # Combined focal + dice loss (simplified)
            bce = F.binary_cross_entropy_with_logits(masks, targets)
            pred_sig = masks.sigmoid()
            intersection = (pred_sig * targets).sum(dim=(-2, -1))
            union = pred_sig.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
            dice = 1 - (2 * intersection + 1) / (union + 1)
            loss = bce + dice.mean()
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