#!/usr/bin/env python3
"""T5-Small (generation workload) - seq2seq training, batch=4, ~60M params

T5-Small used for generation tasks: 6 encoder + 6 decoder layers, 512
hidden, 8 heads, 2048 FFN dim. Longer decoder sequences than T5-Base
to reflect the generation workload (enc=512, dec=256).

Reference: Raffel et al., "Exploring the Limits of Transfer Learning
with a Unified Text-to-Text Transformer", JMLR 2020
"""
import time, json, math, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 4

# === FIXED ===
EPOCHS = 5
NUM_SAMPLES = 4000
VOCAB_SIZE = 32128
ENC_SEQ_LEN = 512
DEC_SEQ_LEN = 256      # longer decoder for generation workload
NUM_LAYERS = 6
HIDDEN = 512
NUM_HEADS = 8
FFN_DIM = 2048
DROPOUT = 0.1
NUM_BUCKETS = 32
MAX_DISTANCE = 128

# ---------------------------------------------------------------------------
# T5 components (T5-Small config)
# ---------------------------------------------------------------------------
class T5RelativePositionBias(nn.Module):
    def __init__(self, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.bias = nn.Embedding(NUM_BUCKETS, NUM_HEADS)

    def _relative_position_bucket(self, rel_pos):
        num_buckets = NUM_BUCKETS
        ret = 0
        if self.bidirectional:
            num_buckets //= 2
            ret = (rel_pos > 0).long() * num_buckets
            rel_pos = rel_pos.abs()
        else:
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact
        val_if_large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(MAX_DISTANCE / max_exact)
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return ret + torch.where(is_small, rel_pos, val_if_large)

    def forward(self, qlen, klen):
        dev = self.bias.weight.device
        q_pos = torch.arange(qlen, device=dev)[:, None]
        k_pos = torch.arange(klen, device=dev)[None, :]
        buckets = self._relative_position_bucket(k_pos - q_pos)
        values = self.bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


class T5Attention(nn.Module):
    def __init__(self, bidirectional=True):
        super().__init__()
        self.head_dim = HIDDEN // NUM_HEADS
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.pos_bias = T5RelativePositionBias(bidirectional)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, context=None, causal=False):
        B, S, _ = x.shape
        kv = context if context is not None else x
        _, T, _ = kv.shape
        def reshape(t, l): return t.view(B, l, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(self.q(x), S), reshape(self.k(kv), T), reshape(self.v(kv), T)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if context is None:
            attn = attn + self.pos_bias(S, T)
        if causal:
            mask = torch.triu(torch.ones(S, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask, float("-inf"))
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, HIDDEN)
        return self.o(out)


class T5RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(HIDDEN))
    def forward(self, x):
        return self.weight * x / (x.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)


class T5FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.wi = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.wo = nn.Linear(FFN_DIM, HIDDEN, bias=False)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.drop(self.wo(nn.functional.relu(self.wi(x))))


class T5EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = T5RMSNorm()
        self.attn = T5Attention(bidirectional=True)
        self.norm2 = T5RMSNorm()
        self.ffn = T5FFN()
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class T5DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = T5RMSNorm()
        self.self_attn = T5Attention(bidirectional=False)
        self.norm2 = T5RMSNorm()
        self.cross_attn = T5Attention(bidirectional=True)
        self.norm3 = T5RMSNorm()
        self.ffn = T5FFN()
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x, enc_out):
        x = x + self.drop(self.self_attn(self.norm1(x), causal=True))
        x = x + self.drop(self.cross_attn(self.norm2(x), context=enc_out))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x


class T5Small(nn.Module):
    """T5-Small encoder-decoder for generation. ~60M trainable parameters."""

    def __init__(self):
        super().__init__()
        self.shared_emb = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.encoder = nn.ModuleList([T5EncoderBlock() for _ in range(NUM_LAYERS)])
        self.enc_norm = T5RMSNorm()
        self.decoder = nn.ModuleList([T5DecoderBlock() for _ in range(NUM_LAYERS)])
        self.dec_norm = T5RMSNorm()
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.shared_emb.weight
        self.drop = nn.Dropout(DROPOUT)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=HIDDEN ** -0.5)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=1.0)

    def forward(self, enc_ids, dec_ids, labels=None):
        x = self.drop(self.shared_emb(enc_ids))
        for block in self.encoder:
            x = block(x)
        enc_out = self.enc_norm(x)

        y = self.drop(self.shared_emb(dec_ids))
        for block in self.decoder:
            y = block(y, enc_out)
        logits = self.lm_head(self.dec_norm(y))

        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100
            )
            return loss
        return logits


class SyntheticSeq2SeqDataset(Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, _):
        enc_ids = torch.randint(100, VOCAB_SIZE, (ENC_SEQ_LEN,))
        dec_ids = torch.randint(100, VOCAB_SIZE, (DEC_SEQ_LEN,))
        labels = dec_ids.clone()
        return enc_ids, dec_ids, labels


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = T5Small().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"hf_t5_generate | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    ds = SyntheticSeq2SeqDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for enc_ids, dec_ids, labels in loader:
            enc_ids = enc_ids.to(dev)
            dec_ids = dec_ids.to(dev)
            labels = labels.to(dev)
            optim.zero_grad()
            loss = model(enc_ids, dec_ids, labels=labels)
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