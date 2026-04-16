#!/usr/bin/env python3
"""CLIP (Contrastive Language-Image Pre-training) - batch=32, ~40M params"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 32
EMBED_DIM = 256
VISION_WIDTH = 384
TEXT_WIDTH = 256
VISION_LAYERS = 6
TEXT_LAYERS = 6
VISION_HEADS = 6
TEXT_HEADS = 4
EPOCHS = 3
NUM_SAMPLES = 2000
IMAGE_SIZE = 224
PATCH_SIZE = 32
NUM_CHANNELS = 3
VOCAB_SIZE = 49408
MAX_SEQ_LEN = 77
TEMPERATURE = 0.07
LR = 5e-4

class PatchEmbed(nn.Module):
    def __init__(self, img_sz, p_sz, in_ch, dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, p_sz, p_sz)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class MHA(nn.Module):
    def __init__(self, dim, nh):
        super().__init__()
        self.nh, self.hd = nh, dim // nh
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.hd ** -0.5)
        if mask is not None: attn = attn.masked_fill(mask, float('-inf'))
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class Block(nn.Module):
    def __init__(self, dim, nh, mlp_r=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MHA(dim, nh)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_r)), nn.GELU(), nn.Linear(int(dim * mlp_r), dim))
    def forward(self, x, mask=None): return x + self.mlp(self.n2(x + self.attn(self.n1(x), mask)))

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        np = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch = PatchEmbed(IMAGE_SIZE, PATCH_SIZE, NUM_CHANNELS, VISION_WIDTH)
        self.cls = nn.Parameter(torch.zeros(1, 1, VISION_WIDTH))
        self.pos = nn.Parameter(torch.zeros(1, np + 1, VISION_WIDTH))
        self.blocks = nn.ModuleList([Block(VISION_WIDTH, VISION_HEADS) for _ in range(VISION_LAYERS)])
        self.norm = nn.LayerNorm(VISION_WIDTH)
        self.proj = nn.Linear(VISION_WIDTH, EMBED_DIM, bias=False)
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)
    def forward(self, x):
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), self.patch(x)], 1) + self.pos
        for b in self.blocks: x = b(x)
        return self.proj(self.norm(x[:, 0]))

class TextEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB_SIZE, TEXT_WIDTH)
        self.pos = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, TEXT_WIDTH))
        self.blocks = nn.ModuleList([Block(TEXT_WIDTH, TEXT_HEADS) for _ in range(TEXT_LAYERS)])
        self.norm = nn.LayerNorm(TEXT_WIDTH)
        self.proj = nn.Linear(TEXT_WIDTH, EMBED_DIM, bias=False)
        self.register_buffer('mask', torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), 1).bool())
        nn.init.normal_(self.pos, std=0.02)
    def forward(self, t):
        x = self.tok(t) + self.pos[:, :t.size(1)]
        m = self.mask[:t.size(1), :t.size(1)].unsqueeze(0).unsqueeze(0)
        for b in self.blocks: x = b(x, m)
        return self.proj(self.norm(x[:, -1]))

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis, self.txt = ViT(), TextEnc()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / TEMPERATURE))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, std=0.02); m.bias is not None and nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)
    def forward(self, img, tok):
        img_f, txt_f = F.normalize(self.vis(img), -1), F.normalize(self.txt(tok), -1)
        s = self.logit_scale.exp()
        lpi, lpt = s * img_f @ txt_f.t(), s * txt_f @ img_f.t()
        lbl = torch.arange(img.size(0), device=img.device)
        return (F.cross_entropy(lpi, lbl) + F.cross_entropy(lpt, lbl)) / 2, img_f, txt_f

class CLIPDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.rand(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), torch.randint(1, VOCAB_SIZE, (MAX_SEQ_LEN,))

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = CLIP().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"vision_language_clip","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"clip_32_40M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = CLIPDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4, eps=1e-6)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for img, tok in loader:
            img, tok = img.to(dev), tok.to(dev)
            opt.zero_grad()
            loss, _, _ = model(img, tok)
            loss.backward()
            opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
