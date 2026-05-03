#!/usr/bin/env python3
"""Speech Transformer - sequence-to-sequence, batch=1, ~120M params

A transformer model designed for speech/sequence tasks with a large
embedding dimension (1536). Uses a standard encoder-decoder architecture
with learned position embeddings. Smaller vocab (128) reflecting
token-level speech coding.

Reference: torchbenchmark speech_transformer model
"""
import time, json, math, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1

# === FIXED ===
EPOCHS = 10
NUM_SAMPLES = 100000
VOCAB_SIZE = 128         # small vocab for speech tokens
SEQ_LEN = 64
EMBED_DIM = 1536         # large embedding dimension
NUM_LAYERS = 6
NUM_HEADS = 16
FFN_DIM = 4096
DROPOUT = 0.1
MAX_SEQ = 512

# ---------------------------------------------------------------------------
# Speech Transformer components
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = EMBED_DIM // NUM_HEADS
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.k_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.v_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.out_proj = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, context=None, causal=False):
        B, S, _ = x.shape
        kv = context if context is not None else x
        _, T, _ = kv.shape
        def reshape(t, l): return t.view(B, l, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(self.q_proj(x), S), reshape(self.k_proj(kv), T), reshape(self.v_proj(kv), T)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if causal:
            mask = torch.triu(torch.ones(S, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float("-inf"))
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, EMBED_DIM)
        return self.out_proj(out)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(EMBED_DIM, FFN_DIM)
        self.fc2 = nn.Linear(FFN_DIM, EMBED_DIM)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.fc2(nn.functional.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.attn = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.ffn = FFN()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.self_attn = MultiHeadAttention()
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        self.cross_attn = MultiHeadAttention()
        self.norm3 = nn.LayerNorm(EMBED_DIM)
        self.ffn = FFN()
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, enc_out):
        x = x + self.drop(self.self_attn(self.norm1(x), causal=True))
        x = x + self.drop(self.cross_attn(self.norm2(x), context=enc_out))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class SpeechTransformer(nn.Module):
    """Encoder-decoder transformer with 1536-d embeddings. ~120M parameters."""

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ, EMBED_DIM)
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(NUM_LAYERS)])
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(NUM_LAYERS)])
        self.enc_norm = nn.LayerNorm(EMBED_DIM)
        self.dec_norm = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        self.drop = nn.Dropout(DROPOUT)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0)

        # Encoder
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for layer in self.encoder:
            x = layer(x)
        enc_out = self.enc_norm(x)

        # Decoder (teacher forcing: use same input shifted)
        y = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for layer in self.decoder:
            y = layer(y, enc_out)
        logits = self.lm_head(self.dec_norm(y))

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), labels.view(-1)
            )
            return loss
        return logits


class SyntheticSpeechDataset(Dataset):
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

    model = SpeechTransformer().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"speech_transformer | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticSpeechDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

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
