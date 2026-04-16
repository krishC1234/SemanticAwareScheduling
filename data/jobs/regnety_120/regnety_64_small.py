#!/usr/bin/env python3
"""RegNetY-120 - batch=64, small params (~13M)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import math
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 64
WIDTH_MULT = 0.5

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
NUM_CLASSES = 1000
IMG_SIZE = 224

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, rd_ch):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, rd_ch, 1)
        self.fc2 = nn.Conv2d(rd_ch, in_ch, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.act(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, activation=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)

class YBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, group_width=24, bottleneck_ratio=1.0, se_ratio=0.25):
        super().__init__()
        
        mid_ch = int(out_ch * bottleneck_ratio)
        groups = max(1, mid_ch // group_width)
        mid_ch = groups * group_width
        
        self.conv1 = ConvBNAct(in_ch, mid_ch, kernel_size=1)
        self.conv2 = ConvBNAct(mid_ch, mid_ch, kernel_size=3, stride=stride, groups=groups)
        
        se_ch = max(1, int(round(in_ch * se_ratio)))
        self.se = SqueezeExcitation(mid_ch, se_ch)
        
        self.conv3 = ConvBNAct(mid_ch, out_ch, kernel_size=1, activation=False)
        
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=stride, activation=False)
        else:
            self.shortcut = None
        
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.shortcut is None else self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        
        return self.act(out + shortcut)

class RegNetY(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.0):
        super().__init__()
        
        stage_config = [
            [2, 336, 2, 24],
            [5, 672, 2, 24],
            [11, 1344, 2, 24],
            [1, 2688, 2, 24],
        ]
        
        stem_ch = _make_divisible(32 * width_mult, 8)
        self.stem = ConvBNAct(3, stem_ch, kernel_size=3, stride=2)
        
        stages = []
        in_ch = stem_ch
        for depth, width, stride, group_width in stage_config:
            out_ch = _make_divisible(width * width_mult, 8)
            gw = max(1, _make_divisible(group_width * width_mult, 8))
            blocks = []
            for i in range(depth):
                s = stride if i == 0 else 1
                blocks.append(YBlock(in_ch, out_ch, stride=s, group_width=gw))
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
        
        self.stages = nn.Sequential(*stages)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_ch, num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

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
    
    model = RegNetY(NUM_CLASSES, WIDTH_MULT).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"cnn","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"regnety_64_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticImageDataset(NUM_SAMPLES, NUM_CLASSES, IMG_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            optim.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
