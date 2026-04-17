#!/usr/bin/env python3
"""UNet for Image Segmentation - batch=4, small params (~7M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 4
BASE_CHANNELS = 32  # Base channel count (original UNet uses 64)

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 500
IMAGE_SIZE = 256    # Reduced from 640x959 for benchmarking
IN_CHANNELS = 3
N_CLASSES = 2
BILINEAR = True
LR = 1e-5

class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet: Convolutional Networks for Biomedical Image Segmentation"""
    def __init__(self, n_channels, n_classes, base_ch=BASE_CHANNELS, bilinear=BILINEAR):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16 // factor)
        
        # Decoder (expanding path)
        self.up1 = Up(base_ch * 16, base_ch * 8 // factor, bilinear)
        self.up2 = Up(base_ch * 8, base_ch * 4 // factor, bilinear)
        self.up3 = Up(base_ch * 4, base_ch * 2 // factor, bilinear)
        self.up4 = Up(base_ch * 2, base_ch, bilinear)
        self.outc = OutConv(base_ch, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def dice_loss(pred, target, multiclass=True):
    """Dice loss for segmentation"""
    smooth = 1e-6
    if multiclass:
        dice = 0
        for c in range(pred.size(1)):
            p = pred[:, c].reshape(-1)
            t = target[:, c].reshape(-1)
            inter = (p * t).sum()
            dice += (2 * inter + smooth) / (p.sum() + t.sum() + smooth)
        return 1 - dice / pred.size(1)
    else:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        inter = (pred * target).sum()
        return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)

class SegmentationDataset(Dataset):
    """Synthetic segmentation dataset"""
    def __init__(self, size, img_size, in_channels, n_classes):
        self.size = size
        self.img_size = img_size
        self.in_channels = in_channels
        self.n_classes = n_classes
    
    def __len__(self): return self.size
    
    def __getitem__(self, i):
        image = torch.rand(self.in_channels, self.img_size, self.img_size)
        # Random segmentation mask
        mask = torch.randint(0, self.n_classes, (self.img_size, self.img_size))
        return image, mask

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = UNet(IN_CHANNELS, N_CLASSES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"segmentation_unet","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"unet_4_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SegmentationDataset(NUM_SAMPLES, IMAGE_SIZE, IN_CHANNELS, N_CLASSES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-8, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for images, masks in loader:
            images, masks = images.to(dev), masks.to(dev)
            
            with torch.cuda.amp.autocast():
                pred = model(images)
                ce_loss = criterion(pred, masks)
                # Dice loss
                pred_softmax = F.softmax(pred, dim=1)
                masks_onehot = F.one_hot(masks, N_CLASSES).permute(0, 3, 1, 2).float()
                d_loss = dice_loss(pred_softmax, masks_onehot, multiclass=True)
                loss = ce_loss + d_loss
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
