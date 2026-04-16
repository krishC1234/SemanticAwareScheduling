#!/usr/bin/env python3
"""fastNLP BERT Q&A - batch=8, ~5M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 8
HIDDEN = 256
N_LAYERS = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 21128
SEQ_LEN = 128
ATTN_HEADS = 8

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

class BertForQA(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.blocks = nn.ModuleList([TransformerBlock(hidden, heads) for _ in range(n_layers)])
        self.qa_out = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(0.1)
    def forward(self, x, mask=None):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        if mask is None: mask = (x.sum(-1) != 0).unsqueeze(1).unsqueeze(2)
        for b in self.blocks: x = b(x, mask)
        logits = self.qa_out(x)
        return logits[:,:,0], logits[:,:,1]

class SyntheticQADataset(Dataset):
    def __init__(self, size, vocab_size, seq_len):
        self.size, self.vocab_size, self.seq_len = size, vocab_size, seq_len
    def __len__(self): return self.size
    def __getitem__(self, i):
        ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        start = torch.randint(0, self.seq_len//2, (1,)).item()
        end = torch.randint(start, min(start+20, self.seq_len), (1,)).item()
        return ids, start, end

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = BertForQA(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"transformer","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"fastnlp_8_5M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticQADataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    crit = nn.CrossEntropyLoss()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for ids, start, end in loader:
            ids, start, end = ids.to(dev), start.to(dev), end.to(dev)
            optim.zero_grad()
            s_log, e_log = model(ids)
            loss = crit(s_log, start) + crit(e_log, end)
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
