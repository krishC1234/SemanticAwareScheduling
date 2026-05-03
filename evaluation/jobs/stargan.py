#!/usr/bin/env python3
"""StarGAN - multi-domain image-to-image translation, batch=16, ~53M params

StarGAN uses a single generator and discriminator to perform image
translation across multiple domains (e.g. hair color, gender, age).
The generator is an encoder-decoder with residual blocks; the
discriminator produces both a real/fake score and a domain classification.

Reference: Choi et al., "StarGAN: Unified Generative Adversarial Networks
for Multi-Domain Image-to-Image Translation", CVPR 2018
"""
import time, json, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 16

# === FIXED ===
EPOCHS = 30
NUM_SAMPLES = 100000
IMG_SIZE = 128
NUM_DOMAINS = 5          # CelebA attributes (e.g. black hair, blond, brown, male, young)
G_CONV_DIM = 64
D_CONV_DIM = 64
NUM_RES_BLOCKS = 6

# ---------------------------------------------------------------------------
# StarGAN Generator
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """StarGAN generator: encoder → residual blocks → decoder.
    Input: image (3) + domain label (NUM_DOMAINS) = 3+5 channels."""

    def __init__(self, c_dim=NUM_DOMAINS, conv_dim=G_CONV_DIM, num_res=NUM_RES_BLOCKS):
        super().__init__()
        # Encoder: down-sampling
        layers = [
            nn.Conv2d(3 + c_dim, conv_dim, 7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
        ]
        # Down-sample
        curr_dim = conv_dim
        for _ in range(2):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2
        # Bottleneck
        for _ in range(num_res):
            layers.append(ResidualBlock(curr_dim))
        # Decoder: up-sampling
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            curr_dim //= 2
        layers.append(nn.Conv2d(curr_dim, 3, 7, stride=1, padding=3))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate domain label spatially and concat with image
        c = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return self.net(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    """StarGAN discriminator: produces real/fake + domain classification."""

    def __init__(self, c_dim=NUM_DOMAINS, conv_dim=D_CONV_DIM):
        super().__init__()
        layers = [nn.Conv2d(3, conv_dim, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
        curr_dim = conv_dim
        for _ in range(5):
            layers += [
                nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1),
                nn.LeakyReLU(0.01),
            ]
            curr_dim *= 2
        self.main = nn.Sequential(*layers)
        # Output: real/fake (patch-based) + domain classification
        self.out_src = nn.Conv2d(curr_dim, 1, 3, stride=1, padding=1)
        self.out_cls = nn.Conv2d(curr_dim, c_dim, kernel_size=IMG_SIZE // 64)

    def forward(self, x):
        h = self.main(x)
        out_src = self.out_src(h)
        out_cls = self.out_cls(h).view(h.size(0), -1)
        return out_src, out_cls


class SyntheticStarGANDataset(Dataset):
    def __init__(self, size, num_domains, img_size):
        self.size, self.num_domains, self.img_size = size, num_domains, img_size
    def __len__(self): return self.size
    def __getitem__(self, _):
        img = torch.randn(3, self.img_size, self.img_size)
        label_org = torch.zeros(self.num_domains)
        label_org[torch.randint(0, self.num_domains, (1,))] = 1.0
        label_trg = torch.zeros(self.num_domains)
        label_trg[torch.randint(0, self.num_domains, (1,))] = 1.0
        return img, label_org, label_trg


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    gen = Generator().to(dev)
    disc = Discriminator().to(dev)
    pc_g = count_params(gen)
    pc_d = count_params(disc)
    pc = pc_g + pc_d
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "gan", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"stargan | GPUs:{ws} | Batch:{BATCH_SIZE} | "
              f"Params: G={pc_g:,} + D={pc_d:,} = {pc:,}")

    gen = DDP(gen, device_ids=[rank])
    disc = DDP(disc, device_ids=[rank])
    ds = SyntheticStarGANDataset(NUM_SAMPLES, NUM_DOMAINS, IMG_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    g_optim = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
    d_optim = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        gen.train()
        disc.train()
        sampler.set_epoch(ep)
        es = time.time()
        for imgs, labels_org, labels_trg in loader:
            imgs = imgs.to(dev)
            labels_org = labels_org.to(dev)
            labels_trg = labels_trg.to(dev)

            # --- Discriminator step ---
            out_src, out_cls = disc(imgs)
            d_loss_real = -out_src.mean()
            d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels_org)

            fake = gen(imgs, labels_trg).detach()
            out_src_fake, _ = disc(fake)
            d_loss_fake = out_src_fake.mean()
            d_loss = d_loss_real + d_loss_fake + d_loss_cls

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # --- Generator step ---
            fake = gen(imgs, labels_trg)
            out_src, out_cls = disc(fake)
            g_loss_fake = -out_src.mean()
            g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, labels_trg)
            # Reconstruction loss
            recon = gen(fake, labels_org)
            g_loss_rec = F.l1_loss(recon, imgs)
            g_loss = g_loss_fake + g_loss_cls + 10.0 * g_loss_rec

            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        tsp += len(ds)
        if rank == 0:
            print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | "
                  f"throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | "
              f"Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###")
        print(json.dumps({"batch_size": BATCH_SIZE, "param_count": pc,
                           "gpu_count": ws, "total_time_sec": round(tt, 2),
                           "avg_throughput": round(tsp / tt, 1)}))
        print("###END_RESULTS###")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
