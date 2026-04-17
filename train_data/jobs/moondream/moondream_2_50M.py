#!/usr/bin/env python3
"""Moondream (Lightweight VLM) - batch=2, ~50M params"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 2
VISION_WIDTH = 384
VISION_LAYERS = 6
VISION_HEADS = 6
IMAGE_SIZE = 384
PATCH_SIZE = 14
NUM_CHANNELS = 3
LLM_DIM = 512
LLM_LAYERS = 6
LLM_HEADS = 8
VOCAB_SIZE = 51200
MAX_SEQ_LEN = 256
EPOCHS = 3
NUM_SAMPLES = 500
LR = 2e-5

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(NUM_CHANNELS, VISION_WIDTH, PATCH_SIZE, PATCH_SIZE)
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

class SigLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.patch = PatchEmbed()
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, VISION_WIDTH))
        self.blocks = nn.ModuleList([Block(VISION_WIDTH, VISION_HEADS) for _ in range(VISION_LAYERS)])
        self.norm = nn.LayerNorm(VISION_WIDTH)
        nn.init.normal_(self.pos, std=0.02)
    def forward(self, x):
        x = self.patch(x) + self.pos
        for b in self.blocks: x = b(x)
        return self.norm(x)

class VisionProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(VISION_WIDTH, LLM_DIM), nn.GELU(), nn.Linear(LLM_DIM, LLM_DIM))
    def forward(self, x): return self.proj(x)

class PhiMLP(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = LLM_DIM * 4
        self.fc1 = nn.Linear(LLM_DIM, hidden)
        self.fc2 = nn.Linear(hidden, LLM_DIM)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x), approximate='tanh'))

class PhiBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(LLM_DIM)
        self.attn = MHA(LLM_DIM, LLM_HEADS)
        self.mlp = PhiMLP()
    def forward(self, x, mask=None):
        h = self.norm(x)
        x = x + self.attn(h, mask) + self.mlp(h)
        return x

class PhiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, LLM_DIM)
        self.blocks = nn.ModuleList([PhiBlock() for _ in range(LLM_LAYERS)])
        self.norm = nn.LayerNorm(LLM_DIM)
        self.head = nn.Linear(LLM_DIM, VOCAB_SIZE, bias=False)
        self.register_buffer('causal_mask', torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), 1).bool())
    def forward(self, x):
        seq_len = x.size(1)
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        for b in self.blocks: x = b(x, mask)
        return self.head(self.norm(x))

class Moondream(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = SigLIPEncoder()
        self.projector = VisionProjector()
        self.llm = PhiLM()
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    def forward(self, images, input_ids, labels=None):
        vis_features = self.vision(images)
        vis_embeds = self.projector(vis_features)
        txt_embeds = self.llm.tok_emb(input_ids)
        combined = torch.cat([vis_embeds, txt_embeds], dim=1)
        logits = self.llm(combined)
        if labels is not None:
            num_vis = vis_embeds.size(1)
            text_logits = logits[:, num_vis-1:-1]
            loss = F.cross_entropy(text_logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=-100)
            return loss, logits
        return logits

class MoondreamDataset(Dataset):
    def __init__(self, sz):
        self.sz = sz
        self.num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    def __len__(self): return self.sz
    def __getitem__(self, i):
        image = torch.rand(NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        seq_len = MAX_SEQ_LEN - self.num_patches
        input_ids = torch.randint(1, VOCAB_SIZE, (seq_len,))
        labels = torch.randint(1, VOCAB_SIZE, (seq_len,))
        return image, input_ids, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = Moondream().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"moondream","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"moondream_2_50M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = MoondreamDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for img, ids, lbl in loader:
            img, ids, lbl = img.to(dev), ids.to(dev), lbl.to(dev)
            opt.zero_grad()
            loss, _ = model(img, ids, lbl)
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
