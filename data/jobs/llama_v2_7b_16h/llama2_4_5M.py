#!/usr/bin/env python3
"""LLaMA-2 7B 16h - batch=4, ~5M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 4
HIDDEN = 256
N_LAYERS = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 32000
SEQ_LEN = 256
ATTN_HEADS = 16
FF_MULT = 4

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Llama2Attention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.h = heads
        self.d_k = hidden // heads
        self.wq = nn.Linear(hidden, hidden, bias=False)
        self.wk = nn.Linear(hidden, hidden, bias=False)
        self.wv = nn.Linear(hidden, hidden, bias=False)
        self.wo = nn.Linear(hidden, hidden, bias=False)
        
    def forward(self, x, freqs_cis, mask=None):
        bs, seq_len, _ = x.size()
        q = self.wq(x).view(bs, seq_len, self.h, self.d_k)
        k = self.wk(x).view(bs, seq_len, self.h, self.d_k)
        v = self.wv(x).view(bs, seq_len, self.h, self.d_k)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.h * self.d_k)
        return self.wo(x)

class Llama2FeedForward(nn.Module):
    def __init__(self, hidden, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, ff_dim, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class Llama2Block(nn.Module):
    def __init__(self, hidden, heads, ff_dim):
        super().__init__()
        self.attention = Llama2Attention(hidden, heads)
        self.feed_forward = Llama2FeedForward(hidden, ff_dim)
        self.attention_norm = RMSNorm(hidden)
        self.ffn_norm = RMSNorm(hidden)
    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class Llama2(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, ff_dim, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([Llama2Block(hidden, heads, ff_dim) for _ in range(n_layers)])
        self.norm = RMSNorm(hidden)
        self.output = nn.Linear(hidden, vocab_size, bias=False)
        self.register_buffer("freqs_cis", precompute_freqs_cis(hidden // heads, seq_len))
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.tok_emb(x)
        freqs_cis = self.freqs_cis[:seq_len]
        for layer in self.layers: x = layer(x, freqs_cis)
        return self.output(self.norm(x))

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
    
    ff_dim = int(2 * HIDDEN * FF_MULT / 3)
    ff_dim = ((ff_dim + 255) // 256) * 256
    
    model = Llama2(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, ff_dim, SEQ_LEN).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"causal_lm","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"llama2_4_5M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticLMDataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
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
