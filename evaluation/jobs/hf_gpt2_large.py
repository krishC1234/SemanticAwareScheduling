#!/usr/bin/env python3
"""GPT-2 Large - autoregressive language modeling, batch=4, ~774M params

GPT-2 Large: 36 layers, 1280 hidden, 20 attention heads. Trained with
causal (left-to-right) language modeling loss on synthetic token sequences.

Reference: Radford et al., "Language Models are Unsupervised Multitask
Learners", OpenAI 2019
"""
import time, json, math, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 4000
VOCAB_SIZE = 50257      # GPT-2 BPE vocab
SEQ_LEN = 1024          # GPT-2 default context length
NUM_LAYERS = 36
HIDDEN = 1280
NUM_HEADS = 20
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# GPT-2 components
# ---------------------------------------------------------------------------
class GPT2Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.wpe = nn.Embedding(SEQ_LEN, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, input_ids):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)
        return self.drop(self.wte(input_ids) + self.wpe(pos))


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = HIDDEN // NUM_HEADS
        self.c_attn = nn.Linear(HIDDEN, 3 * HIDDEN)  # fused QKV projection
        self.c_proj = nn.Linear(HIDDEN, HIDDEN)
        self.attn_drop = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(HIDDEN, dim=-1)
        def reshape(t): return t.view(B, S, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, HIDDEN)
        return self.resid_drop(self.c_proj(out))


class GPT2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(HIDDEN, 4 * HIDDEN)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * HIDDEN, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.mlp = GPT2MLP()

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Large(nn.Module):
    """GPT-2 Large for causal LM. ~774M trainable parameters."""

    def __init__(self):
        super().__init__()
        self.embeddings = GPT2Embeddings()
        self.blocks = nn.ModuleList([GPT2Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(HIDDEN, eps=1e-5)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        # Tie weights
        self.lm_head.weight = self.embeddings.wte.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        x = self.embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE), shift_labels.view(-1)
            )
            return loss
        return logits


class SyntheticLMDataset(Dataset):
    def __init__(self, size, seq_len, vocab_size):
        self.size, self.seq_len, self.vocab_size = size, seq_len, vocab_size
    def __len__(self): return self.size
    def __getitem__(self, _):
        ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids, ids.clone()  # labels = input for causal LM


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = GPT2Large().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"hf_gpt2_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticLMDataset(NUM_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.01)

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