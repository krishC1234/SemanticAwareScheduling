#!/usr/bin/env python3
"""EfficientNet-B0 - batch=128, small params (~1.5M)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import math
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 128
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

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, squeeze_ch):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, squeeze_ch, 1)
        self.fc2 = nn.Conv2d(squeeze_ch, in_ch, 1)
        self.activation = Swish()
        self.scale_activation = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.activation(self.fc1(scale))
        scale = self.scale_activation(self.fc2(scale))
        return x * scale

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, activation=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3),
        ]
        if activation:
            layers.append(Swish())
        super().__init__(*layers)

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_connect_rate=0.0):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch
        self.drop_connect_rate = drop_connect_rate
        
        mid_ch = _make_divisible(in_ch * expand_ratio, 8)
        squeeze_ch = max(1, int(in_ch * se_ratio))
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, mid_ch, kernel_size=1))
        
        layers.append(ConvBNAct(mid_ch, mid_ch, kernel_size=kernel_size, stride=stride, groups=mid_ch))
        layers.append(SqueezeExcitation(mid_ch, squeeze_ch))
        layers.append(ConvBNAct(mid_ch, out_ch, kernel_size=1, activation=False))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            if self.training and self.drop_connect_rate > 0:
                keep_prob = 1.0 - self.drop_connect_rate
                random_tensor = keep_prob + torch.rand((x.size(0), 1, 1, 1), dtype=x.dtype, device=x.device)
                binary_tensor = torch.floor(random_tensor)
                result = result / keep_prob * binary_tensor
            result = x + result
        return result

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.2, drop_connect_rate=0.2):
        super().__init__()
        
        config = [
            [1,  16, 1, 1, 3],
            [6,  24, 2, 2, 3],
            [6,  40, 2, 2, 5],
            [6,  80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        stem_ch = _make_divisible(32 * width_mult, 8)
        self.stem = ConvBNAct(3, stem_ch, kernel_size=3, stride=2)
        
        layers = []
        in_ch = stem_ch
        total_blocks = sum(c[2] for c in config)
        block_idx = 0
        
        for expand_ratio, out_ch, repeats, stride, kernel_size in config:
            out_ch = _make_divisible(out_ch * width_mult, 8)
            for i in range(repeats):
                s = stride if i == 0 else 1
                dc_rate = drop_connect_rate * block_idx / total_blocks
                layers.append(MBConv(in_ch, out_ch, kernel_size, s, expand_ratio, drop_connect_rate=dc_rate))
                in_ch = out_ch
                block_idx += 1
        
        self.blocks = nn.Sequential(*layers)
        
        head_ch = _make_divisible(1280 * width_mult, 8)
        self.head = ConvBNAct(in_ch, head_ch, kernel_size=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(head_ch, num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
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
    
    model = EfficientNetB0(NUM_CLASSES, WIDTH_MULT).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"cnn","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"efficientnet_128_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
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
