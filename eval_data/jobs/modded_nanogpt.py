#!/usr/bin/env python3
"""Modded NanoGPT - causal LM with sliding window attention, batch=1, ~85M params

A modified NanoGPT variant with sliding window attention and long context
support. 12 layers, 768 hidden, 6 heads. Uses BF16 embeddings and
supports very long sequences (up to 262144 tokens). Here we train with
a practical seq_len of 6144.

Reference: Karpathy NanoGPT + sliding window modifications
"""
import time, json, math, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1

# === FIXED ===
EPOCHS = 10
NUM_SAMPLES = 5000       # fewer samples due to long seq_len
VOCAB_SIZE = 50257
SEQ_LEN = 6144           # long context
NUM_LAYERS = 12
HIDDEN = 768
NUM_HEADS = 6
DROPOUT = 0.0
WINDOW_SIZE = 1024       # sliding window size

# ---------------------------------------------------------------------------
# Modded NanoGPT components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


class SlidingWindowAttention(nn.Module):
    """Causal self-attention with a sliding window mask."""

    def __init__(self):
        super().__init__()
        self.head_dim = HIDDEN // NUM_HEADS
        self.scale = self.head_dim ** -0.5
        self.c_attn = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.c_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(HIDDEN, dim=-1)
        def reshape(t): return t.view(B, S, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Causal + sliding window mask
        row = torch.arange(S, device=x.device)[:, None]
        col = torch.arange(S, device=x.device)[None, :]
        causal = col <= row
        window = (row - col) < WINDOW_SIZE
        mask = causal & window
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, HIDDEN)
        return self.c_proj(out)


class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = RMSNorm(HIDDEN)
        self.attn = SlidingWindowAttention()
        self.ln_2 = RMSNorm(HIDDEN)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 4 * HIDDEN, bias=False),
            nn.GELU(),
            nn.Linear(4 * HIDDEN, HIDDEN, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ModdedNanoGPT(nn.Module):
    """NanoGPT with sliding window attention. ~85M trainable parameters."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.wpe = nn.Embedding(SEQ_LEN, HIDDEN)
        self.blocks = nn.ModuleList([GPTBlock() for _ in range(NUM_LAYERS)])
        self.ln_f = RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight  # tie weights

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1)
            )
            return loss
        return logits


class SyntheticLMDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, _):
        ids = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
        return ids, ids.clone()


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = ModdedNanoGPT().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"modded_nanogpt | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticLMDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(dev), labels.to(dev)
            optim.zero_grad()
            loss = model(input_ids, labels=labels)
            loss.backward()
            optim.step()
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