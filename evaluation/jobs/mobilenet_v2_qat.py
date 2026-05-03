#!/usr/bin/env python3
"""MobileNetV2 Quantization-Aware Training - batch=96, ~3.4M params

MobileNetV2 uses inverted residual blocks with depthwise separable
convolutions. This script trains with full precision (QAT fakequant
observers are not included here since DDP + quantize_fx interaction
is non-trivial; the compute profile matches the real QAT workload).

Reference: Sandler et al., "MobileNetV2: Inverted Residuals and
Linear Bottlenecks", CVPR 2018
"""
import time, json, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 96

# === FIXED ===
EPOCHS = 30
NUM_SAMPLES = 6000
NUM_CLASSES = 1000
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# MobileNetV2 building blocks
# ---------------------------------------------------------------------------
class ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm → ReLU6 (standard MobileNet activation)."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Inverted residual block: pointwise expand → depthwise → pointwise project.
    Uses residual connection when stride=1 and in_ch == out_ch."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        hidden = int(in_ch * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_ch, hidden, kernel=1))
        layers.extend([
            ConvBNReLU(hidden, hidden, stride=stride, groups=hidden),  # depthwise
            nn.Conv2d(hidden, out_ch, 1, bias=False),                  # pointwise linear
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """Full MobileNetV2 architecture — ~3.4M trainable parameters.

    Block config: (expand_ratio, output_channels, num_blocks, stride)
    matching the original paper Table 2."""

    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()
        input_ch = int(32 * width_mult)
        last_ch = int(1280 * width_mult)

        # t, c, n, s
        block_config = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        features = [ConvBNReLU(3, input_ch, stride=2)]
        for t, c, n, s in block_config:
            out_ch = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_ch, out_ch, stride, t))
                input_ch = out_ch
        features.append(ConvBNReLU(input_ch, last_ch, kernel=1))
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_ch, num_classes),
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


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

    model = MobileNetV2(NUM_CLASSES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "cnn", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"mobilenet_v2_qat | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticImageDataset(NUM_SAMPLES, NUM_CLASSES, IMG_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
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