#!/usr/bin/env python3
"""DCGAN - batch=32, small params (~1M with ngf=ndf=32)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32
NGF = 32  # Generator feature maps
NDF = 32  # Discriminator feature maps

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
IMAGE_SIZE = 64
NC = 3      # Number of channels (RGB)
NZ = 100    # Latent vector size
LR = 0.0002
BETA1 = 0.5

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: (nc) x 64 x 64
        )
    
    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # output: 1 x 1 x 1
        )
    
    def forward(self, x):
        return self.main(x).view(-1)

class SyntheticImageDataset(Dataset):
    def __init__(self, size, nc, image_size):
        self.size, self.nc, self.image_size = size, nc, image_size
    def __len__(self): return self.size
    def __getitem__(self, i):
        return torch.randn(self.nc, self.image_size, self.image_size)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    # Create Generator and Discriminator
    netG = Generator(NZ, NGF, NC).to(dev)
    netD = Discriminator(NC, NDF).to(dev)
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    pc = count_params(netG) + count_params(netD)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gan_dcgan","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"dcgan_32_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,} (G:{count_params(netG):,} D:{count_params(netD):,})")
    
    netG = DDP(netG, device_ids=[rank])
    netD = DDP(netD, device_ids=[rank])
    
    ds = SyntheticImageDataset(NUM_SAMPLES, NC, IMAGE_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    real_label, fake_label = 1.0, 0.0
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        netG.train(); netD.train()
        sampler.set_epoch(ep); es = time.time()
        
        for real_images in loader:
            real_images = real_images.to(dev)
            b_size = real_images.size(0)
            
            # === Train Discriminator ===
            # Real images
            netD.zero_grad()
            label = torch.full((b_size,), real_label, dtype=torch.float, device=dev)
            output = netD(real_images)
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # Fake images
            noise = torch.randn(b_size, NZ, 1, 1, device=dev)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()
            
            # === Train Generator ===
            netG.zero_grad()
            label.fill_(real_label)  # Fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
        
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
