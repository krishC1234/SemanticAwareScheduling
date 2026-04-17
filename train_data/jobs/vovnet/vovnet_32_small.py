#!/usr/bin/env python3
"""VoVNet (Variety of View Network) - batch=32, small params (~11M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 32
BASE_CHANNELS = 64
STAGE_CHANNELS = [128, 160, 192, 224]  # Reduced from VoVNet39a
LAYER_PER_BLOCK = 5
EPOCHS = 3
NUM_SAMPLES = 2000
IMG_SIZE = 224
IN_CHANNELS = 3
NUM_CLASSES = 1000
LR = 0.1

def conv_bn_relu(ic, oc, k=3, s=1, p=1, g=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True)
    )

class OSAModule(nn.Module):
    """One-Shot Aggregation Module - concatenates features from all layers"""
    def __init__(self, in_ch, mid_ch, out_ch, num_layers=5, reduction=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(conv_bn_relu(in_ch, mid_ch, 3, 1, 1))
        for _ in range(num_layers - 1):
            self.layers.append(conv_bn_relu(mid_ch, mid_ch, 3, 1, 1))
        # Aggregation: concat all layer outputs
        agg_ch = in_ch + mid_ch * num_layers
        self.agg = conv_bn_relu(agg_ch, out_ch, 1, 1, 0)
        self.reduction = reduction
        if reduction:
            self.pool = nn.MaxPool2d(3, 2, 1)
    
    def forward(self, x):
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.agg(x)
        if self.reduction:
            x = self.pool(x)
        return x

class VoVNet(nn.Module):
    """VoVNet: An Energy and GPU-Computation Efficient Backbone"""
    def __init__(self, base_ch=BASE_CHANNELS, stage_ch=STAGE_CHANNELS, layers_per_block=LAYER_PER_BLOCK):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            conv_bn_relu(IN_CHANNELS, base_ch // 2, 3, 2, 1),
            conv_bn_relu(base_ch // 2, base_ch // 2, 3, 1, 1),
            conv_bn_relu(base_ch // 2, base_ch, 3, 1, 1),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # OSA stages
        self.stages = nn.ModuleList()
        in_ch = base_ch
        for i, out_ch in enumerate(stage_ch):
            mid_ch = out_ch // 2
            reduction = (i < len(stage_ch) - 1)
            self.stages.append(OSAModule(in_ch, mid_ch, out_ch, layers_per_block, reduction))
            in_ch = out_ch
        
        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_ch[-1], NUM_CLASSES)
        
        self._init_weights()
    
    def _init_weights(self):
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
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class ImageDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.rand(IN_CHANNELS, IMG_SIZE, IMG_SIZE), torch.randint(0, NUM_CLASSES, (1,)).item()

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = VoVNet().to(dev); pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"cnn_vovnet","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"vovnet_32_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = ImageDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for img, lbl in loader:
            img, lbl = img.to(dev), lbl.to(dev)
            opt.zero_grad(); loss = crit(model(img), lbl); loss.backward(); opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
