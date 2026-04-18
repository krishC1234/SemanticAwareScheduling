#!/usr/bin/env python3
"""LLaMA 7B - causal LM training, batch=1, ~6.7B params

LLaMA architecture: pre-norm transformer with RMSNorm, SwiGLU FFN,
rotary position embeddings (RoPE), and no bias terms. 32 layers,
4096 hidden, 32 heads.

Reference: Touvron et al., "LLaMA: Open and Efficient Foundation
Language Models", Meta AI 2023
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
EPOCHS = 5
NUM_SAMPLES = 2000       # small due to massive model
VOCAB_SIZE = 32000       # LLaMA tokenizer
SEQ_LEN = 512            # shortened for memory (full is 2048)
NUM_LAYERS = 32
HIDDEN = 4096
NUM_HEADS = 32
FFN_HIDDEN = 11008       # SwiGLU: 2/3 * 4 * 4096, rounded
DROPOUT = 0.0
ROPE_BASE = 10000.0

# ---------------------------------------------------------------------------
# LLaMA components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.weight * x / (rms + self.eps)


def precompute_rope(dim, seq_len, base=ROPE_BASE):
    """Precompute cos/sin for rotary position embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to x of shape (B, H, S, D)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class LLaMAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = HIDDEN // NUM_HEADS
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wk = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wv = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.wo = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        def reshape(t): return t.view(B, S, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(self.wq(x)), reshape(self.wk(x)), reshape(self.wv(x))
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, HIDDEN)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU FFN: gate(x) * up(x), then down projection."""
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(HIDDEN, FFN_HIDDEN, bias=False)   # gate
        self.w2 = nn.Linear(FFN_HIDDEN, HIDDEN, bias=False)   # down
        self.w3 = nn.Linear(HIDDEN, FFN_HIDDEN, bias=False)   # up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(HIDDEN)
        self.attn = LLaMAAttention()
        self.norm2 = RMSNorm(HIDDEN)
        self.ffn = SwiGLU()

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


class LLaMA7B(nn.Module):
    """LLaMA 7B: 32 layers, 4096 hidden, 32 heads, SwiGLU, RoPE.
    ~6.7B trainable parameters."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.layers = nn.ModuleList([LLaMABlock() for _ in range(NUM_LAYERS)])
        self.norm = RMSNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tie weights

        # Precompute RoPE (registered as buffer so they move with .to())
        head_dim = HIDDEN // NUM_HEADS
        cos, sin = precompute_rope(head_dim, SEQ_LEN)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        x = self.tok_emb(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.norm(x))

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

    model = LLaMA7B().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"llama_tp | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

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
