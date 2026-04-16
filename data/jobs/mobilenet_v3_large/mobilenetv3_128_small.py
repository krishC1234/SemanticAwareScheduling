#!/usr/bin/env python3
"""MobileNetV3 Large - batch=128, small params (~2M)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from functools import partial
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

class Hardswish(nn.Module):
    def forward(self, x):
        return x * nn.functional.relu6(x + 3, inplace=True) / 6

class Hardsigmoid(nn.Module):
    def forward(self, x):
        return nn.functional.relu6(x + 3, inplace=True) / 6

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = Hardsigmoid()

    def forward(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale

class ConvBNActivation(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, activation=nn.ReLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            activation() if activation else nn.Identity()
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, kernel_size, stride, use_se, activation):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch
        
        layers = []
        if exp_ch != in_ch:
            layers.append(ConvBNActivation(in_ch, exp_ch, kernel_size=1, activation=activation))
        
        layers.append(ConvBNActivation(exp_ch, exp_ch, kernel_size=kernel_size, stride=stride, groups=exp_ch, activation=activation))
        
        if use_se:
            squeeze_ch = _make_divisible(exp_ch // 4, 8)
            layers.append(SqueezeExcitation(exp_ch, squeeze_ch))
        
        layers.append(ConvBNActivation(exp_ch, out_ch, kernel_size=1, activation=None))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        result = self.block(x)
        if self.use_res_connect:
            result = x + result
        return result

class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.2):
        super().__init__()
        
        bneck_conf = [
            [3,  16,  16, False, 0, 1],
            [3,  64,  24, False, 0, 2],
            [3,  72,  24, False, 0, 1],
            [5,  72,  40, True,  0, 2],
            [5, 120,  40, True,  0, 1],
            [5, 120,  40, True,  0, 1],
            [3, 240,  80, False, 1, 2],
            [3, 200,  80, False, 1, 1],
            [3, 184,  80, False, 1, 1],
            [3, 184,  80, False, 1, 1],
            [3, 480, 112, True,  1, 1],
            [3, 672, 112, True,  1, 1],
            [5, 672, 160, True,  1, 2],
            [5, 960, 160, True,  1, 1],
            [5, 960, 160, True,  1, 1],
        ]
        
        activations = [nn.ReLU, Hardswish]
        
        firstconv_out = _make_divisible(16 * width_mult, 8)
        layers = [ConvBNActivation(3, firstconv_out, kernel_size=3, stride=2, activation=Hardswish)]
        
        in_ch = firstconv_out
        for k, exp, out, se, act_idx, s in bneck_conf:
            exp_ch = _make_divisible(exp * width_mult, 8)
            out_ch = _make_divisible(out * width_mult, 8)
            layers.append(InvertedResidual(in_ch, exp_ch, out_ch, k, s, se, activations[act_idx]))
            in_ch = out_ch
        
        lastconv_in = in_ch
        lastconv_out = _make_divisible(960 * width_mult, 8)
        last_ch = _make_divisible(1280 * width_mult, 8)
        
        layers.append(ConvBNActivation(lastconv_in, lastconv_out, kernel_size=1, activation=Hardswish))
        
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_out, last_ch),
            Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(last_ch, num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
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
    
    model = MobileNetV3Large(NUM_CLASSES, WIDTH_MULT).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"cnn","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"mobilenetv3_128_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
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
