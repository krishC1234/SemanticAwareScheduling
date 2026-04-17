#!/usr/bin/env python3
"""Reformer - batch=32, ~5M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32
HIDDEN = 256
N_LAYERS = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 30000
SEQ_LEN = 256
ATTN_HEADS = 8
NUM_HASHES = 2
NUM_BUCKETS = 32

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LSHAttention(nn.Module):
    def __init__(self, hidden, heads, num_hashes=2, num_buckets=32, dropout=0.1):
        super().__init__()
        self.h = heads
        self.d_k = hidden // heads
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.q_lin = nn.Linear(hidden, hidden)
        self.v_lin = nn.Linear(hidden, hidden)
        self.out_lin = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        bs, seq_len, _ = x.size()
        qk = self.q_lin(x)
        v = self.v_lin(x)
        qk = qk.view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(qk, qk.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.h * self.d_k)
        return self.out_lin(x)

class ChunkedFeedForward(nn.Module):
    def __init__(self, hidden, chunk_size=64, dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.ff1 = nn.Linear(hidden, hidden * 4)
        self.ff2 = nn.Linear(hidden * 4, hidden)
        self.act = GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        bs, seq_len, hidden = x.size()
        chunks = x.split(self.chunk_size, dim=1)
        output_chunks = []
        for chunk in chunks:
            out = self.dropout(self.ff2(self.act(self.ff1(chunk))))
            output_chunks.append(out)
        return torch.cat(output_chunks, dim=1)

class ReversibleBlock(nn.Module):
    def __init__(self, hidden, heads, num_hashes, num_buckets, dropout=0.1):
        super().__init__()
        self.attn = LSHAttention(hidden, heads, num_hashes, num_buckets, dropout)
        self.ff = ChunkedFeedForward(hidden, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class Reformer(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len, num_hashes, num_buckets):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([ReversibleBlock(hidden, heads, num_hashes, num_buckets) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for b in self.blocks: x = b(x)
        return self.norm(x)

class ReformerForLM(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len, num_hashes, num_buckets):
        super().__init__()
        self.reformer = Reformer(vocab_size, hidden, n_layers, heads, seq_len, num_hashes, num_buckets)
        self.lm_head = nn.Linear(hidden, vocab_size)
        
    def forward(self, x):
        return self.lm_head(self.reformer(x))

class SyntheticLMDataset(Dataset):
    def __init__(self, size, vocab_size, seq_len):
        self.size, self.vocab_size, self.seq_len = size, vocab_size, seq_len
    def __len__(self): return self.size
    def __getitem__(self, i):
        ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        labels = torch.randint(1, self.vocab_size, (self.seq_len,))
        return ids, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = ReformerForLM(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN, NUM_HASHES, NUM_BUCKETS).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"efficient_transformer","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"reformer_32_5M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticLMDataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for ids, labels in loader:
            ids, labels = ids.to(dev), labels.to(dev)
            optim.zero_grad()
            logits = model(ids)
            loss = crit(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
