#!/usr/bin/env python3
"""GPT-2 - batch=4, ~40M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 4
HIDDEN = 512
N_LAYERS = 8

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 30000
SEQ_LEN = 128
ATTN_HEADS = 8

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden, heads, seq_len, dropout=0.1):
        super().__init__()
        self.h = heads
        self.d_k = hidden // heads
        self.c_attn = nn.Linear(hidden, 3 * hidden)
        self.c_proj = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
        
    def forward(self, x):
        bs, seq_len, _ = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.h * self.d_k, dim=-1)
        q = q.view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        k = k.view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        v = v.view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.h * self.d_k)
        return self.c_proj(x)

class MLP(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(hidden, hidden * 4)
        self.c_proj = nn.Linear(hidden * 4, hidden)
        self.act = GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class GPT2Block(nn.Module):
    def __init__(self, hidden, heads, seq_len, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, heads, seq_len, dropout)
        self.ln_2 = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden)
        self.wpe = nn.Embedding(seq_len, hidden)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([GPT2Block(hidden, heads, seq_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.drop(self.wte(x) + self.wpe(pos))
        for block in self.blocks: x = block(x)
        return self.lm_head(self.ln_f(x))

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
    
    model = GPT2(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"causal_lm","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"gpt2_4_40M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
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
