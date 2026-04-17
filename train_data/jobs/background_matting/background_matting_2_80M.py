#!/usr/bin/env python3
"""Background Matting (GAN-based Video Matting) - batch=2, ~80M params"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 2
N_BLOCKS1 = 7
N_BLOCKS2 = 3
NGF = 64
NDF = 64
RESOLUTION = 256
EPOCHS = 3
NUM_SAMPLES = 80
LR_G = 1e-4
LR_D = 1e-5

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.conv(x)

class ResnetConditionHR(nn.Module):
    def __init__(self, n_blocks1, n_blocks2, ngf):
        super().__init__()
        input_nc = 11
        self.enc1 = nn.Sequential(nn.Conv2d(input_nc, ngf, 7, padding=3), nn.InstanceNorm2d(ngf), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True))
        self.res_blocks1 = nn.Sequential(*[ResBlock(ngf*4) for _ in range(n_blocks1)])
        self.res_blocks2 = nn.Sequential(*[ResBlock(ngf*4) for _ in range(n_blocks2)])
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(ngf), nn.ReLU(True))
        self.alpha_head = nn.Sequential(nn.Conv2d(ngf, ngf, 3, padding=1), nn.ReLU(True), nn.Conv2d(ngf, 1, 7, padding=3), nn.Tanh())
        self.fg_head = nn.Sequential(nn.Conv2d(ngf, ngf, 3, padding=1), nn.ReLU(True), nn.Conv2d(ngf, 3, 7, padding=3), nn.Tanh())
    def forward(self, image, bg, seg, multi_fr):
        x = torch.cat([image, bg, seg, multi_fr], dim=1)
        x = self.enc1(x); x = self.enc2(x); x = self.enc3(x)
        x = self.res_blocks1(x); x = self.res_blocks2(x)
        x = self.dec1(x); x = self.dec2(x)
        return self.alpha_head(x), self.fg_head(x)

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, ndf, num_D=2):
        super().__init__()
        self.num_D = num_D
        self.discriminators = nn.ModuleList([self._make_disc(ndf) for _ in range(num_D)])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    def _make_disc(self, ndf):
        return nn.Sequential(
            nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, padding=1)
        )
    def forward(self, x):
        results = []
        for disc in self.discriminators:
            results.append(disc(x)); x = self.downsample(x)
        return results

def alpha_loss(pred, target, mask): return F.l1_loss(pred * mask, target * mask)
def compose_loss(image, alpha, fg, bg, mask):
    alpha_norm = (alpha + 1) / 2
    return F.l1_loss((fg * alpha_norm + bg * (1 - alpha_norm)) * mask, image * mask)
def alpha_gradient_loss(pred, target, mask):
    return F.l1_loss(pred[:,:,:,1:] - pred[:,:,:,:-1], target[:,:,:,1:] - target[:,:,:,:-1]) + \
           F.l1_loss(pred[:,:,1:,:] - pred[:,:,:-1,:], target[:,:,1:,:] - target[:,:,:-1,:])
def gan_loss(pred, target_is_real):
    return F.mse_loss(pred, torch.ones_like(pred) if target_is_real else torch.zeros_like(pred))

class MattingDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i):
        return (torch.rand(3, RESOLUTION, RESOLUTION) * 2 - 1, torch.rand(3, RESOLUTION, RESOLUTION) * 2 - 1,
                torch.rand(1, RESOLUTION, RESOLUTION) * 2 - 1, torch.rand(4, RESOLUTION, RESOLUTION) * 2 - 1,
                torch.rand(1, RESOLUTION, RESOLUTION) * 2 - 1, torch.rand(3, RESOLUTION, RESOLUTION) * 2 - 1)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    netG = ResnetConditionHR(N_BLOCKS1, N_BLOCKS2, NGF).to(dev)
    netD = MultiscaleDiscriminator(NDF).to(dev)
    pc = count_params(netG) + count_params(netD)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"background_matting_gan","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"background_matting_2_80M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    netG = DDP(netG, device_ids=[rank]); netD = DDP(netD, device_ids=[rank])
    ds = MattingDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optG = torch.optim.Adam(netG.parameters(), lr=LR_G); optD = torch.optim.Adam(netD.parameters(), lr=LR_D)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        netG.train(); netD.train(); sampler.set_epoch(ep); es = time.time()
        for idx, (image, bg, seg, multi_fr, alpha_gt, fg_gt) in enumerate(loader):
            image, bg, seg, multi_fr = image.to(dev), bg.to(dev), seg.to(dev), multi_fr.to(dev)
            alpha_gt, fg_gt = alpha_gt.to(dev), fg_gt.to(dev)
            mask = torch.ones_like(alpha_gt)
            alpha_pred, fg_pred = netG(image, bg, seg, multi_fr)
            al_loss = alpha_loss(alpha_pred, alpha_gt, mask)
            fg_loss = F.l1_loss(fg_pred, fg_gt)
            comp_loss = compose_loss(image, alpha_pred, fg_pred, bg, mask)
            grad_loss = alpha_gradient_loss(alpha_pred, alpha_gt, mask)
            alpha_norm = (alpha_pred + 1) / 2
            composed = fg_pred * alpha_norm + bg * (1 - alpha_norm)
            fake_response = netD(composed)
            loss_ganG = sum(gan_loss(r, True) for r in fake_response) / len(fake_response)
            lossG = loss_ganG + 0.05 * (al_loss + fg_loss + comp_loss + grad_loss)
            optG.zero_grad(); lossG.backward(); optG.step()
            if idx % 5 == 0:
                with torch.no_grad():
                    alpha_pred, fg_pred = netG(image, bg, seg, multi_fr)
                    alpha_norm = (alpha_pred + 1) / 2
                    composed = fg_pred * alpha_norm + bg * (1 - alpha_norm)
                fake_response = netD(composed.detach()); real_response = netD(image)
                loss_D_fake = sum(gan_loss(r, False) for r in fake_response) / len(fake_response)
                loss_D_real = sum(gan_loss(r, True) for r in real_response) / len(real_response)
                lossD = (loss_D_fake + loss_D_real) * 0.5
                optD.zero_grad(); lossD.backward(); optD.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
