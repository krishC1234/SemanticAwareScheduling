#!/usr/bin/env python3
"""Phlippe ResNet (UvA DLC Tutorial 5) - batch=128, ~0.27M params, CIFAR-10

A simple ResNet designed for CIFAR-10 (32×32 images) from the UvA Deep
Learning Course. Three groups of [3, 3, 3] basic blocks with channels
[16, 32, 64]. Much smaller than ImageNet ResNets.

Reference: UvA DLC Notebooks — Tutorial 5: Inception, ResNet & DenseNet
"""
import time, json, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 128

# === FIXED ===
EPOCHS = 30
NUM_SAMPLES = 50000
NUM_CLASSES = 10
IMG_SIZE = 32          # CIFAR-10 resolution
C_HIDDEN = [16, 32, 64]
NUM_BLOCKS = [3, 3, 3]

# ---------------------------------------------------------------------------
# ResNet blocks (matching the phlippe tutorial implementation)
# ---------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out, subsample=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1,
                      stride=2 if subsample else 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.downsample = (
            nn.Conv2d(c_in, c_out, 1, stride=2) if subsample else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(z + x)


class PhilippeResNet(nn.Module):
    """Small CIFAR-10 ResNet: 3 groups × 3 blocks, channels [16, 32, 64].
    ~0.27M trainable parameters."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_net = nn.Sequential(
            nn.Conv2d(3, C_HIDDEN[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(C_HIDDEN[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for group_idx, block_count in enumerate(NUM_BLOCKS):
            for bc in range(block_count):
                subsample = bc == 0 and group_idx > 0
                c_in = C_HIDDEN[group_idx - 1] if subsample else C_HIDDEN[group_idx]
                blocks.append(ResNetBlock(c_in, C_HIDDEN[group_idx], subsample))
        self.blocks = nn.Sequential(*blocks)
        self.output_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C_HIDDEN[-1], num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_net(x)
        x = self.blocks(x)
        return self.output_net(x)


class SyntheticCIFARDataset(Dataset):
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

    model = PhilippeResNet(NUM_CLASSES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "cnn", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"phlippe_resnet | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticCIFARDataset(NUM_SAMPLES, NUM_CLASSES, IMG_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
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