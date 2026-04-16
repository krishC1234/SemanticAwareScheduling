#!/usr/bin/env python3
"""DCGAN - batch=64, small params (~1M with ngf=ndf=32)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 64
NGF = 32
NDF = 32
EPOCHS = 3
NUM_SAMPLES = 2000
IMAGE_SIZE = 64
NC = 3
NZ = 100
LR = 0.0002
BETA1 = 0.5

def weights_init(m):
    cn = m.__class__.__name__
    if cn.find('Conv') != -1: nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif cn.find('BatchNorm') != -1: nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh(),
        )
    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False), nn.Sigmoid(),
        )
    def forward(self, x): return self.main(x).view(-1)

class SyntheticImageDataset(Dataset):
    def __init__(self, sz, nc, isz): self.sz, self.nc, self.isz = sz, nc, isz
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.randn(self.nc, self.isz, self.isz)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    
    netG = Generator(NZ, NGF, NC).to(dev); netD = Discriminator(NC, NDF).to(dev)
    netG.apply(weights_init); netD.apply(weights_init)
    pc = count_params(netG) + count_params(netD)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gan_dcgan","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"dcgan_64_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    netG, netD = DDP(netG, device_ids=[rank]), DDP(netD, device_ids=[rank])
    ds = SyntheticImageDataset(NUM_SAMPLES, NC, IMAGE_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        netG.train(); netD.train(); sampler.set_epoch(ep); es = time.time()
        for real in loader:
            real = real.to(dev); b = real.size(0)
            netD.zero_grad()
            label = torch.full((b,), 1.0, dtype=torch.float, device=dev)
            criterion(netD(real), label).backward()
            noise = torch.randn(b, NZ, 1, 1, device=dev)
            fake = netG(noise)
            label.fill_(0.0)
            criterion(netD(fake.detach()), label).backward()
            optimizerD.step()
            netG.zero_grad()
            label.fill_(1.0)
            criterion(netD(fake), label).backward()
            optimizerG.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
