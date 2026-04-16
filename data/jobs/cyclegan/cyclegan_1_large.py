#!/usr/bin/env python3
"""CycleGAN - batch=1, large params (~20M with ngf=ndf=64)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import itertools
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 1
NGF = 64
NDF = 64
EPOCHS = 3
NUM_SAMPLES = 500
IMAGE_SIZE = 256
NC = 3
N_RESIDUAL_BLOCKS = 9
LR = 0.0002
BETA1 = 0.5
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 0.5

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False), nn.InstanceNorm2d(ch), nn.ReLU(True),
                                   nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3, bias=False), nn.InstanceNorm2d(ch))
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, nc, ngf, n_res):
        super().__init__()
        m = [nn.ReflectionPad2d(3), nn.Conv2d(nc, ngf, 7, bias=False), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch * 2
            m += [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False), nn.InstanceNorm2d(out_ch), nn.ReLU(True)]
            in_ch = out_ch
        for _ in range(n_res): m += [ResidualBlock(in_ch)]
        for _ in range(2):
            out_ch = in_ch // 2
            m += [nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False), nn.InstanceNorm2d(out_ch), nn.ReLU(True)]
            in_ch = out_ch
        m += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*m)
    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1, bias=False), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1))
    def forward(self, x): return self.model(x)

class ImageBuffer:
    def __init__(self, max_size=50): self.max_size, self.data = max_size, []
    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size: self.data.append(img); result.append(img)
            elif torch.rand(1).item() > 0.5:
                idx = torch.randint(0, self.max_size, (1,)).item()
                result.append(self.data[idx].clone()); self.data[idx] = img
            else: result.append(img)
        return torch.cat(result, 0)

class SyntheticPairedDataset(Dataset):
    def __init__(self, sz, nc, isz): self.sz, self.nc, self.isz = sz, nc, isz
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.randn(self.nc, self.isz, self.isz), torch.randn(self.nc, self.isz, self.isz)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    
    G_AB, G_BA = Generator(NC, NGF, N_RESIDUAL_BLOCKS).to(dev), Generator(NC, NGF, N_RESIDUAL_BLOCKS).to(dev)
    D_A, D_B = Discriminator(NC, NDF).to(dev), Discriminator(NC, NDF).to(dev)
    pc = count_params(G_AB) + count_params(G_BA) + count_params(D_A) + count_params(D_B)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gan_cyclegan","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"cyclegan_1_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    G_AB, G_BA = DDP(G_AB, device_ids=[rank]), DDP(G_BA, device_ids=[rank])
    D_A, D_B = DDP(D_A, device_ids=[rank]), DDP(D_B, device_ids=[rank])
    
    ds = SyntheticPairedDataset(NUM_SAMPLES, NC, IMAGE_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    crit_GAN, crit_cycle, crit_id = nn.MSELoss(), nn.L1Loss(), nn.L1Loss()
    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(BETA1, 0.999))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(BETA1, 0.999))
    fake_A_buf, fake_B_buf = ImageBuffer(), ImageBuffer()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        sampler.set_epoch(ep); es = time.time()
        for real_A, real_B in loader:
            real_A, real_B = real_A.to(dev), real_B.to(dev)
            valid = torch.ones(real_A.size(0), 1, 30, 30, device=dev)
            fake_t = torch.zeros(real_A.size(0), 1, 30, 30, device=dev)
            # Train G
            opt_G.zero_grad()
            fake_B, fake_A = G_AB(real_A), G_BA(real_B)
            loss_G = (crit_GAN(D_B(fake_B), valid) + crit_GAN(D_A(fake_A), valid) +
                      crit_cycle(G_BA(fake_B), real_A) * LAMBDA_CYCLE + crit_cycle(G_AB(fake_A), real_B) * LAMBDA_CYCLE +
                      crit_id(G_BA(real_A), real_A) * LAMBDA_CYCLE * LAMBDA_IDENTITY + crit_id(G_AB(real_B), real_B) * LAMBDA_CYCLE * LAMBDA_IDENTITY)
            loss_G.backward(); opt_G.step()
            # Train D_A
            opt_D_A.zero_grad()
            ((crit_GAN(D_A(real_A), valid) + crit_GAN(D_A(fake_A_buf.push_and_pop(fake_A.detach())), fake_t)) * 0.5).backward(); opt_D_A.step()
            # Train D_B
            opt_D_B.zero_grad()
            ((crit_GAN(D_B(real_B), valid) + crit_GAN(D_B(fake_B_buf.push_and_pop(fake_B.detach())), fake_t)) * 0.5).backward(); opt_D_B.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
