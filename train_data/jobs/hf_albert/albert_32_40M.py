#!/usr/bin/env python3
"""ALBERT - batch=32, ~40M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32
HIDDEN = 512
N_LAYERS = 8

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 30000
SEQ_LEN = 128
ATTN_HEADS = 8
EMBEDDING_SIZE = 128

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, heads, dropout=0.1):
        super().__init__()
        self.h = heads
        self.d_k = hidden // heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output_linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q, k, v = [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.output_linear(x)

class FeedForward(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 4), GELU(), nn.Dropout(dropout), nn.Linear(hidden * 4, hidden), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, hidden, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, heads, dropout)
        self.ff = FeedForward(hidden, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = x + self.drop1(self.attn(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.drop2(self.ff(x))
        return self.norm2(x)

class ALBERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden, seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(seq_len, embed_size)
        self.segment_emb = nn.Embedding(2, embed_size)
        self.projection = nn.Linear(embed_size, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, segment_ids):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        emb = self.token_emb(x) + self.pos_emb(pos) + self.segment_emb(segment_ids)
        return self.dropout(self.norm(self.projection(emb)))

class ALBERT(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = ALBERTEmbedding(vocab_size, embed_size, hidden, seq_len)
        self.shared_block = TransformerBlock(hidden, heads)
    def forward(self, x, segment_ids):
        mask = (x > 0).unsqueeze(1).unsqueeze(2)
        x = self.embedding(x, segment_ids)
        for _ in range(self.n_layers):
            x = self.shared_block(x, mask)
        return x

class ALBERTForPretraining(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.albert = ALBERT(vocab_size, embed_size, hidden, n_layers, heads, seq_len)
        self.sop_classifier = nn.Linear(hidden, 2)
        self.mlm_head = nn.Sequential(nn.Linear(hidden, embed_size), GELU(), nn.LayerNorm(embed_size), nn.Linear(embed_size, vocab_size))
    def forward(self, x, segment_ids):
        x = self.albert(x, segment_ids)
        return self.sop_classifier(x[:, 0]), self.mlm_head(x)

class SyntheticALBERTDataset(Dataset):
    def __init__(self, size, vocab_size, seq_len):
        self.size, self.vocab_size, self.seq_len = size, vocab_size, seq_len
    def __len__(self): return self.size
    def __getitem__(self, i):
        ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        seg = torch.zeros(self.seq_len, dtype=torch.long); seg[self.seq_len//2:] = 1
        sop = torch.randint(0, 2, (1,)).item()
        mlm = torch.randint(0, self.vocab_size, (self.seq_len,))
        return ids, seg, sop, mlm

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = ALBERTForPretraining(VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"transformer","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"albert_32_40M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticALBERTDataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for ids, seg, sop, mlm in loader:
            ids, seg, sop, mlm = ids.to(dev), seg.to(dev), sop.to(dev), mlm.to(dev)
            optim.zero_grad()
            sop_out, mlm_out = model(ids, seg)
            loss = crit(sop_out, sop) + crit(mlm_out.transpose(1, 2), mlm)
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
