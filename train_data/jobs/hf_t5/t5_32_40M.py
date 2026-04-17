#!/usr/bin/env python3
"""T5 - batch=32, ~40M params"""
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
FF_DIM_MULT = 4

class T5LayerNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class T5Attention(nn.Module):
    def __init__(self, hidden, heads, is_decoder=False, has_cross_attn=False, dropout=0.1):
        super().__init__()
        self.h = heads
        self.d_k = hidden // heads
        self.is_decoder = is_decoder
        self.has_cross_attn = has_cross_attn
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_bias = nn.Embedding(32, heads)
        
    def forward(self, x, enc_out=None, mask=None):
        bs, seq_len, _ = x.size()
        q = self.q(x).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        if self.has_cross_attn and enc_out is not None:
            k = self.k(enc_out).view(bs, enc_out.size(1), self.h, self.d_k).transpose(1, 2)
            v = self.v(enc_out).view(bs, enc_out.size(1), self.h, self.d_k).transpose(1, 2)
        else:
            k = self.k(x).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
            v = self.v(x).view(bs, seq_len, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if enc_out is None:
            positions = torch.arange(seq_len, device=x.device)
            rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
            rel_pos = rel_pos.clamp(-16, 15) + 16
            bias = self.rel_pos_bias(rel_pos).permute(2, 0, 1).unsqueeze(0)
            scores = scores + bias
        if self.is_decoder and enc_out is None:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        attn = torch.nan_to_num(attn, 0.0)
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, seq_len, self.h * self.d_k)
        return self.o(x)

class T5FeedForward(nn.Module):
    def __init__(self, hidden, ff_dim, dropout=0.1):
        super().__init__()
        self.wi_0 = nn.Linear(hidden, ff_dim, bias=False)
        self.wi_1 = nn.Linear(hidden, ff_dim, bias=False)
        self.wo = nn.Linear(ff_dim, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        gate = torch.relu(self.wi_0(x))
        hidden = self.wi_1(x)
        return self.dropout(self.wo(gate * hidden))

class T5EncoderBlock(nn.Module):
    def __init__(self, hidden, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = T5Attention(hidden, heads, is_decoder=False, dropout=dropout)
        self.ff = T5FeedForward(hidden, ff_dim, dropout)
        self.norm1 = T5LayerNorm(hidden)
        self.norm2 = T5LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = x + self.drop1(self.self_attn(self.norm1(x), mask=mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x

class T5DecoderBlock(nn.Module):
    def __init__(self, hidden, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = T5Attention(hidden, heads, is_decoder=True, dropout=dropout)
        self.cross_attn = T5Attention(hidden, heads, is_decoder=True, has_cross_attn=True, dropout=dropout)
        self.ff = T5FeedForward(hidden, ff_dim, dropout)
        self.norm1 = T5LayerNorm(hidden)
        self.norm2 = T5LayerNorm(hidden)
        self.norm3 = T5LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
    def forward(self, x, enc_out, enc_mask=None):
        x = x + self.drop1(self.self_attn(self.norm1(x)))
        x = x + self.drop2(self.cross_attn(self.norm2(x), enc_out=enc_out, mask=enc_mask))
        x = x + self.drop3(self.ff(self.norm3(x)))
        return x

class T5(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, ff_dim):
        super().__init__()
        self.shared_emb = nn.Embedding(vocab_size, hidden)
        self.encoder = nn.ModuleList([T5EncoderBlock(hidden, heads, ff_dim) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([T5DecoderBlock(hidden, heads, ff_dim) for _ in range(n_layers)])
        self.enc_norm = T5LayerNorm(hidden)
        self.dec_norm = T5LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.shared_emb.weight
        self.drop = nn.Dropout(0.1)
    def forward(self, enc_ids, dec_ids):
        enc_mask = (enc_ids > 0).unsqueeze(1).unsqueeze(2)
        x = self.drop(self.shared_emb(enc_ids))
        for block in self.encoder: x = block(x, enc_mask)
        enc_out = self.enc_norm(x)
        x = self.drop(self.shared_emb(dec_ids))
        for block in self.decoder: x = block(x, enc_out, enc_mask)
        return self.lm_head(self.dec_norm(x))

class SyntheticT5Dataset(Dataset):
    def __init__(self, size, vocab_size, seq_len):
        self.size, self.vocab_size, self.seq_len = size, vocab_size, seq_len
        self.sentinel_start = vocab_size - 100
    def __len__(self): return self.size
    def __getitem__(self, i):
        enc_ids = torch.randint(1, self.sentinel_start, (self.seq_len,))
        dec_ids = torch.randint(1, self.sentinel_start, (self.seq_len,))
        labels = torch.randint(1, self.sentinel_start, (self.seq_len,))
        return enc_ids, dec_ids, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    ff_dim = HIDDEN * FF_DIM_MULT
    model = T5(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, ff_dim).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"encoder_decoder","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"t5_32_40M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticT5Dataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for enc_ids, dec_ids, labels in loader:
            enc_ids, dec_ids, labels = enc_ids.to(dev), dec_ids.to(dev), labels.to(dev)
            optim.zero_grad()
            logits = model(enc_ids, dec_ids)
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
