#!/usr/bin/env python3
"""UNet for Image Segmentation - batch=8, large params (~31M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 8
BASE_CHANNELS = 64
EPOCHS = 3
NUM_SAMPLES = 500
IMAGE_SIZE = 256
IN_CHANNELS = 3
N_CLASSES = 2
BILINEAR = True
LR = 1e-5

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False), nn.BatchNorm2d(mid_ch), nn.ReLU(True), nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x): return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.mp(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], 1))

class UNet(nn.Module):
    def __init__(self, n_ch, n_cls, base=BASE_CHANNELS, bil=BILINEAR):
        super().__init__()
        self.n_classes = n_cls
        f = 2 if bil else 1
        self.inc = DoubleConv(n_ch, base)
        self.d1, self.d2, self.d3, self.d4 = Down(base, base*2), Down(base*2, base*4), Down(base*4, base*8), Down(base*8, base*16//f)
        self.u1, self.u2, self.u3, self.u4 = Up(base*16, base*8//f, bil), Up(base*8, base*4//f, bil), Up(base*4, base*2//f, bil), Up(base*2, base, bil)
        self.outc = nn.Conv2d(base, n_cls, 1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3); x5 = self.d4(x4)
        return self.outc(self.u4(self.u3(self.u2(self.u1(x5, x4), x3), x2), x1))

def dice_loss(pred, tgt):
    smooth = 1e-6
    dice = sum((2 * (pred[:,c].reshape(-1) * tgt[:,c].reshape(-1)).sum() + smooth) / (pred[:,c].reshape(-1).sum() + tgt[:,c].reshape(-1).sum() + smooth) for c in range(pred.size(1)))
    return 1 - dice / pred.size(1)

class SegDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.rand(IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.randint(0, N_CLASSES, (IMAGE_SIZE, IMAGE_SIZE))

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = UNet(IN_CHANNELS, N_CLASSES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"segmentation_unet","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"unet_8_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SegDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    crit, opt, scaler = nn.CrossEntropyLoss(), torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9), torch.cuda.amp.GradScaler()
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for img, msk in loader:
            img, msk = img.to(dev), msk.to(dev)
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss = crit(pred, msk) + dice_loss(F.softmax(pred, 1), F.one_hot(msk, N_CLASSES).permute(0,3,1,2).float())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
