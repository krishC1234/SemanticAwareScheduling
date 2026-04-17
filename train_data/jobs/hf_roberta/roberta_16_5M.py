#!/usr/bin/env python3
"""RoBERTa - batch=16, ~5M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 16
HIDDEN = 256
N_LAYERS = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 30000
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
        self.q_lin = nn.Linear(hidden, hidden)
        self.k_lin = nn.Linear(hidden, hidden)
        self.v_lin = nn.Linear(hidden, hidden)
        self.out_lin = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        bs, seq_len, _ = x.size()
        q = self.q_lin(x).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        k = self.k_lin(x).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        v = self.v_lin(x).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.h * self.d_k)
        return self.out_lin(x)

class FeedForward(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 4), GELU(), nn.Dropout(dropout), nn.Linear(hidden * 4, hidden), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class RoBERTaBlock(nn.Module):
    def __init__(self, hidden, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, heads, dropout)
        self.ff = FeedForward(hidden, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop1(self.attn(x, mask)))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x

class RoBERTa(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.pos_emb = nn.Embedding(seq_len + 2, hidden)
        self.blocks = nn.ModuleList([RoBERTaBlock(hidden, heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.1)
        
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).unsqueeze(2)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0) + 2
        x = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for b in self.blocks: x = b(x, mask)
        return self.norm(x)

class RoBERTaForMLM(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.roberta = RoBERTa(vocab_size, hidden, n_layers, heads, seq_len)
        self.mlm_dense = nn.Linear(hidden, hidden)
        self.mlm_norm = nn.LayerNorm(hidden)
        self.mlm_head = nn.Linear(hidden, vocab_size)
        self.act = GELU()
        
    def forward(self, x):
        h = self.roberta(x)
        h = self.act(self.mlm_dense(h))
        h = self.mlm_norm(h)
        return self.mlm_head(h)

class SyntheticMLMDataset(Dataset):
    def __init__(self, size, vocab_size, seq_len, mask_prob=0.15):
        self.size, self.vocab_size, self.seq_len = size, vocab_size, seq_len
        self.mask_prob = mask_prob
        self.mask_token_id = 1
        
    def __len__(self): return self.size
    
    def __getitem__(self, i):
        ids = torch.randint(2, self.vocab_size, (self.seq_len,))
        labels = ids.clone()
        mask_positions = torch.rand(self.seq_len) < self.mask_prob
        rand = torch.rand(self.seq_len)
        mask_positions_mask = mask_positions & (rand < 0.8)
        mask_positions_rand = mask_positions & (rand >= 0.8) & (rand < 0.9)
        ids[mask_positions_mask] = self.mask_token_id
        ids[mask_positions_rand] = torch.randint(2, self.vocab_size, (mask_positions_rand.sum(),))
        labels[~mask_positions] = -100
        return ids, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = RoBERTaForMLM(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"transformer","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"roberta_16_5M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticMLMDataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    
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
