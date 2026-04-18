#!/usr/bin/env python3
"""NFNet-F0 (Normalizer-Free Network) - batch=128, ~71.5M params

NFNet eliminates batch normalization entirely, replacing it with
Scaled Weight Standardization + adaptive gradient clipping (AGC).
F0 is the smallest variant with 4 stages of [1, 2, 6, 3] blocks,
base width 128, and group size 128.

Reference: Brock et al., "High-Performance Large-Scale Image Recognition
Without Normalization", ICML 2021
"""
import time, json, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 128

# === FIXED ===
EPOCHS = 15
NUM_SAMPLES = 50000
NUM_CLASSES = 1000
IMG_SIZE = 192        # NFNet-F0 default resolution

# NFNet-F0 config
STAGE_DEPTHS = [1, 2, 6, 3]
STAGE_CHANNELS = [256, 512, 1536, 1536]
GROUP_SIZE = 128
ALPHA = 0.2           # variance scaling for skip path
SE_RATIO = 0.5        # squeeze-excite reduction ratio
EXPECTED_VAR = 1.0     # running signal variance tracker

# ---------------------------------------------------------------------------
# NFNet building blocks
# ---------------------------------------------------------------------------
class ScaledStdConv2d(nn.Conv2d):
    """Weight-Standardized Conv2d with learned gain — replaces BN."""

    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, groups=1):
        super().__init__(in_ch, out_ch, kernel, stride=stride,
                         padding=padding, groups=groups, bias=True)
        self.gain = nn.Parameter(torch.ones(out_ch, 1, 1, 1))

    def forward(self, x):
        w = self.weight
        mean = w.mean(dim=[1, 2, 3], keepdim=True)
        var = w.var(dim=[1, 2, 3], keepdim=True)
        w = (w - mean) / (var.sqrt() + 1e-10)
        return F.conv2d(x, w * self.gain, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, ratio=0.5):
        super().__init__()
        mid = max(1, int(channels * ratio))
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        s = x.mean(dim=[2, 3])
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s[:, :, None, None]


class NFBlock(nn.Module):
    """NFNet bottleneck block: 1×1 → 3×3 (grouped) → 1×1 → SE, no BN."""

    def __init__(self, in_ch, out_ch, stride=1, alpha=ALPHA, group_size=GROUP_SIZE):
        super().__init__()
        mid_ch = out_ch // 2
        groups = mid_ch // group_size

        self.conv1 = ScaledStdConv2d(in_ch, mid_ch, 1)
        self.conv2 = ScaledStdConv2d(mid_ch, mid_ch, 3, stride=stride,
                                      padding=1, groups=groups)
        self.conv3 = ScaledStdConv2d(mid_ch, mid_ch, 3, padding=1,
                                      groups=groups)
        self.conv4 = ScaledStdConv2d(mid_ch, out_ch, 1)
        self.se = SqueezeExcite(out_ch, SE_RATIO)
        self.alpha = alpha
        self.skip_gain = nn.Parameter(torch.zeros(1))

        self.shortcut = nn.Identity()
        if stride > 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity(),
                ScaledStdConv2d(in_ch, out_ch, 1),
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        h = F.gelu(x)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        h = self.conv4(h)
        h = self.se(h)
        return shortcut + self.skip_gain * h


class NFNetF0(nn.Module):
    """NFNet-F0: ~71.5M parameters, 192×192 input.

    4 stages with [1, 2, 6, 3] blocks, channels [256, 512, 1536, 1536]."""

    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem: 3×3 conv stack (like ResNet-D stem but without BN)
        self.stem = nn.Sequential(
            ScaledStdConv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            ScaledStdConv2d(16, 32, 3, padding=1),
            nn.GELU(),
            ScaledStdConv2d(32, 64, 3, padding=1),
            nn.GELU(),
            ScaledStdConv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
        )

        # Build stages
        stages = []
        in_ch = 128
        for stage_idx, (depth, out_ch) in enumerate(zip(STAGE_DEPTHS, STAGE_CHANNELS)):
            for block_idx in range(depth):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                stages.append(NFBlock(in_ch, out_ch, stride=stride))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        self.final_conv = ScaledStdConv2d(in_ch, 3072, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3072, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = F.gelu(self.final_conv(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SyntheticImageDataset(Dataset):
    def __init__(self, size, num_classes, img_size):
        self.size, self.num_classes, self.img_size = size, num_classes, img_size
    def __len__(self): return self.size
    def __getitem__(self, _):
        img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = NFNetF0(NUM_CLASSES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "cnn", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"dm_nfnet_f0 | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticImageDataset(NUM_SAMPLES, NUM_CLASSES, IMG_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    # NFNet paper uses SGD + AGC; we use SGD with clipped grads
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-5)
    crit = nn.CrossEntropyLoss()

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            optim.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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