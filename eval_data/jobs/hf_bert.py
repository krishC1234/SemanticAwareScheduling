#!/usr/bin/env python3
"""BERT-Large (Uncased) for language modeling - batch=4, ~336M params

Full BERT-Large configuration: 24 layers, 1024 hidden, 16 attention heads,
4096 intermediate. Trained with masked-LM loss on synthetic token sequences.

Reference: Devlin et al., "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding", 2019
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
NUM_SAMPLES = 300000
VOCAB_SIZE = 30522       # bert-base/large-uncased vocab
SEQ_LEN = 512
NUM_LAYERS = 24
HIDDEN = 1024
NUM_HEADS = 16
INTERMEDIATE = 4096
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# BERT components — faithful to the original architecture
# ---------------------------------------------------------------------------
class BertEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.pos = nn.Embedding(SEQ_LEN, HIDDEN)
        self.tok_type = nn.Embedding(2, HIDDEN)
        self.norm = nn.LayerNorm(HIDDEN, eps=1e-12)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, input_ids, token_type_ids=None):
        B, S = input_ids.shape
        pos_ids = torch.arange(S, device=input_ids.device).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.word(input_ids) + self.pos(pos_ids) + self.tok_type(token_type_ids)
        return self.drop(self.norm(x))


class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = HIDDEN // NUM_HEADS
        self.query = nn.Linear(HIDDEN, HIDDEN)
        self.key = nn.Linear(HIDDEN, HIDDEN)
        self.value = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, S, _ = x.shape
        def reshape(t): return t.view(B, S, NUM_HEADS, self.head_dim).transpose(1, 2)
        q, k, v = reshape(self.query(x)), reshape(self.key(x)), reshape(self.value(x))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, S, HIDDEN)
        return self.out(out)


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = BertSelfAttention()
        self.norm1 = nn.LayerNorm(HIDDEN, eps=1e-12)
        self.ff = nn.Sequential(
            nn.Linear(HIDDEN, INTERMEDIATE),
            nn.GELU(),
            nn.Linear(INTERMEDIATE, HIDDEN),
            nn.Dropout(DROPOUT),
        )
        self.norm2 = nn.LayerNorm(HIDDEN, eps=1e-12)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x)))
        x = self.norm2(x + self.ff(x))
        return x


class BertLargeForMLM(nn.Module):
    """BERT-Large with a masked-LM head. ~336M trainable parameters."""

    def __init__(self):
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.layers = nn.ModuleList([BertLayer() for _ in range(NUM_LAYERS)])
        # MLM head: hidden → hidden (dense + GELU + LN) → vocab logits
        self.mlm_dense = nn.Linear(HIDDEN, HIDDEN)
        self.mlm_act = nn.GELU()
        self.mlm_norm = nn.LayerNorm(HIDDEN, eps=1e-12)
        self.mlm_decoder = nn.Linear(HIDDEN, VOCAB_SIZE)
        # Tie decoder weights to word embeddings
        self.mlm_decoder.weight = self.embeddings.word.weight

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
        for layer in self.layers:
            x = layer(x)
        logits = self.mlm_decoder(self.mlm_norm(self.mlm_act(self.mlm_dense(x))))
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100
            )
            return loss
        return logits


class SyntheticMLMDataset(Dataset):
    """Generates random token sequences with 15% masked for MLM."""
    MASK_ID = 103  # [MASK] token in bert-uncased vocab

    def __init__(self, size, seq_len, vocab_size):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self): return self.size

    def __getitem__(self, _):
        ids = torch.randint(999, self.vocab_size, (self.seq_len,))  # skip special tokens
        labels = torch.full_like(ids, -100)
        mask = torch.rand(self.seq_len) < 0.15
        labels[mask] = ids[mask]
        ids[mask] = self.MASK_ID
        return ids, labels


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = BertLargeForMLM().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"hf_bert_large_4 | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticMLMDataset(NUM_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

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