#!/usr/bin/env python3
"""CycleGAN - batch=1, small params (~5M with ngf=ndf=32)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import itertools
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1
NGF = 32  # Generator filters
NDF = 32  # Discriminator filters

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 500  # Smaller due to heavy model (4 networks)
IMAGE_SIZE = 256
NC = 3
N_RESIDUAL_BLOCKS = 6  # Reduced for smaller variant
LR = 0.0002
BETA1 = 0.5
LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 0.5

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    """ResNet-based Generator for CycleGAN"""
    def __init__(self, nc, ngf, n_residual):
        super().__init__()
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nc, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]
        # Downsampling
        in_ch = ngf
        for _ in range(2):
            out_ch = in_ch * 2
            model += [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        # Residual blocks
        for _ in range(n_residual):
            model += [ResidualBlock(in_ch)]
        # Upsampling
        for _ in range(2):
            out_ch = in_ch // 2
            model += [
                nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        # Output
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """PatchGAN Discriminator (70x70 receptive field)"""
    def __init__(self, nc, ndf):
        super().__init__()
        model = [
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        model += [nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class ImageBuffer:
    """Buffer to store previously generated images for training stability"""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, images):
        result = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            else:
                if torch.rand(1).item() > 0.5:
                    idx = torch.randint(0, self.max_size, (1,)).item()
                    result.append(self.data[idx].clone())
                    self.data[idx] = img
                else:
                    result.append(img)
        return torch.cat(result, dim=0)

class SyntheticPairedDataset(Dataset):
    def __init__(self, size, nc, image_size):
        self.size, self.nc, self.image_size = size, nc, image_size
    def __len__(self): return self.size
    def __getitem__(self, i):
        # Return unpaired images from domains A and B
        img_A = torch.randn(self.nc, self.image_size, self.image_size)
        img_B = torch.randn(self.nc, self.image_size, self.image_size)
        return img_A, img_B

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    # Create generators and discriminators
    G_AB = Generator(NC, NGF, N_RESIDUAL_BLOCKS).to(dev)  # A -> B
    G_BA = Generator(NC, NGF, N_RESIDUAL_BLOCKS).to(dev)  # B -> A
    D_A = Discriminator(NC, NDF).to(dev)  # Discriminates real A from fake A
    D_B = Discriminator(NC, NDF).to(dev)  # Discriminates real B from fake B
    
    pc = count_params(G_AB) + count_params(G_BA) + count_params(D_A) + count_params(D_B)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gan_cyclegan","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"cyclegan_1_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    G_AB = DDP(G_AB, device_ids=[rank])
    G_BA = DDP(G_BA, device_ids=[rank])
    D_A = DDP(D_A, device_ids=[rank])
    D_B = DDP(D_B, device_ids=[rank])
    
    ds = SyntheticPairedDataset(NUM_SAMPLES, NC, IMAGE_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    # Losses
    criterion_GAN = nn.MSELoss()  # LSGAN loss
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=LR, betas=(BETA1, 0.999))
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=LR, betas=(BETA1, 0.999))
    
    # Image buffers
    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        sampler.set_epoch(ep); es = time.time()
        
        for real_A, real_B in loader:
            real_A, real_B = real_A.to(dev), real_B.to(dev)
            
            # Target tensors for GAN loss
            valid = torch.ones(real_A.size(0), 1, 30, 30, device=dev)
            fake = torch.zeros(real_A.size(0), 1, 30, 30, device=dev)
            
            # === Train Generators ===
            optimizer_G.zero_grad()
            
            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A) * LAMBDA_CYCLE * LAMBDA_IDENTITY
            loss_id_B = criterion_identity(G_AB(real_B), real_B) * LAMBDA_CYCLE * LAMBDA_IDENTITY
            
            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
            
            # Cycle consistency loss
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A) * LAMBDA_CYCLE
            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B) * LAMBDA_CYCLE
            
            # Total generator loss
            loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()
            
            # === Train Discriminator A ===
            optimizer_D_A.zero_grad()
            loss_real = criterion_GAN(D_A(real_A), valid)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A.detach())
            loss_fake = criterion_GAN(D_A(fake_A_), fake)
            loss_D_A = (loss_real + loss_fake) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # === Train Discriminator B ===
            optimizer_D_B.zero_grad()
            loss_real = criterion_GAN(D_B(real_B), valid)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B.detach())
            loss_fake = criterion_GAN(D_B(fake_B_), fake)
            loss_D_B = (loss_real + loss_fake) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
        
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
