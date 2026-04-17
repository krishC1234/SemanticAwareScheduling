#!/usr/bin/env python3
"""ResNeSt-14d (Split-Attention Network) - batch=32, timm default ~10.6M params

ResNeSt applies split-attention blocks (cardinal groups with radix splits)
inside a ResNet backbone. resnest14d is the lightweight variant with
14 layers and a 'd' stem (3×3 convolutions replacing the 7×7 stem conv).

Reference: Zhang et al., "ResNeSt: Split-Attention Networks", 2022
"""
import time, json, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32

# === FIXED ===
EPOCHS = 20
NUM_SAMPLES = 50000
NUM_CLASSES = 1000
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# Split-Attention (SplAt) block — the core of ResNeSt
# ---------------------------------------------------------------------------
class SplitAttention(nn.Module):
    """Radix-major split attention: split channels into `radix` groups within
    each cardinal group, compute channel attention across the radix splits,
    and recombine.  radix=2, groups=1 matches the resnest14d default."""

    def __init__(self, channels, radix=2, groups=1, reduction=4):
        super().__init__()
        self.radix = radix
        self.groups = groups
        inter = max(channels * radix // reduction, 32)
        self.fc1 = nn.Conv2d(channels, inter, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(inter)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter, channels * radix, 1, groups=groups)
        self.rsoftmax = nn.Softmax(dim=1) if radix > 1 else nn.Sigmoid()

    def forward(self, x):
        B, RC, H, W = x.shape
        C = RC // self.radix
        # split along channel dim into `radix` chunks
        splits = x.reshape(B, self.radix, C, H, W)
        gap = splits.sum(dim=1).mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        attn = self.fc2(self.relu(self.bn1(self.fc1(gap))))     # (B, R*C, 1, 1)
        attn = attn.reshape(B, self.radix, C, 1, 1)
        if self.radix > 1:
            attn = attn.softmax(dim=1)
        else:
            attn = attn.sigmoid()
        out = (splits * attn).sum(dim=1)
        return out


class SplAtBottleneck(nn.Module):
    """ResNeSt bottleneck: 1×1 → 3×3‐SplitAttention → 1×1, with optional
    avg-pool anti-alias downsampling ('d' variant)."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=2, groups=1, base_width=64):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width * radix, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * radix)
        self.splat = SplitAttention(width, radix=radix, groups=groups)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.splat(out)
        out = self.pool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNeSt14d(nn.Module):
    """Minimal ResNeSt-14d: deep 3×3 stem + 4 stages of [1, 1, 1, 1]
    SplAt bottleneck blocks. Matches ~10.6M parameters."""

    def __init__(self, num_classes=1000, radix=2, groups=1, base_width=64):
        super().__init__()
        self.inplanes = 64
        # 'd' stem: three 3×3 convs instead of one 7×7
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64,  1, stride=1, radix=radix, groups=groups, base_width=base_width)
        self.layer2 = self._make_layer(128, 1, stride=2, radix=radix, groups=groups, base_width=base_width)
        self.layer3 = self._make_layer(256, 1, stride=2, radix=radix, groups=groups, base_width=base_width)
        self.layer4 = self._make_layer(512, 1, stride=2, radix=radix, groups=groups, base_width=base_width)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * SplAtBottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, radix, groups, base_width):
        downsample = None
        out_ch = planes * SplAtBottleneck.expansion
        if stride != 1 or self.inplanes != out_ch:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(self.inplanes, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [SplAtBottleneck(self.inplanes, planes, stride, downsample,
                                  radix=radix, groups=groups, base_width=base_width)]
        self.inplanes = out_ch
        for _ in range(1, blocks):
            layers.append(SplAtBottleneck(self.inplanes, planes,
                                          radix=radix, groups=groups, base_width=base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SyntheticImageDataset(Dataset):
    def __init__(self, size, num_classes, img_size):
        self.size, self.num_classes, self.img_size = size, num_classes, img_size
    def __len__(self): return self.size
    def __getitem__(self, i):
        img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return img, label


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = ResNeSt14d(NUM_CLASSES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "cnn", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"resnest14d_32 | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticImageDataset(NUM_SAMPLES, NUM_CLASSES, IMG_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
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