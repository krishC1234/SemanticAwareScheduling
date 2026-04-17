#!/usr/bin/env python3
"""BART - batch=4, ~10M params (Encoder-Decoder)"""
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
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_lin(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_lin(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_lin(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.out_lin(x)

class FeedForward(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 4), GELU(), nn.Dropout(dropout), nn.Linear(hidden * 4, hidden), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class EncoderBlock(nn.Module):
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

class DecoderBlock(nn.Module):
    def __init__(self, hidden, heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden, heads, dropout)
        self.cross_attn = MultiHeadAttention(hidden, heads, dropout)
        self.ff = FeedForward(hidden, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = x + self.drop1(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x)
        x = x + self.drop2(self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm2(x)
        x = x + self.drop3(self.ff(x))
        return self.norm3(x)

class BART(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, heads, seq_len):
        super().__init__()
        self.enc_emb = nn.Embedding(vocab_size, hidden)
        self.dec_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.enc_blocks = nn.ModuleList([EncoderBlock(hidden, heads) for _ in range(n_layers)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(hidden, heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.drop = nn.Dropout(0.1)
    def make_causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool().logical_not()
    def forward(self, src, tgt):
        pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        src_mask = (src > 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = self.make_causal_mask(tgt.size(1), tgt.device).unsqueeze(0).unsqueeze(1)
        enc = self.drop(self.enc_emb(src) + self.pos_emb(pos))
        for b in self.enc_blocks: enc = b(enc, src_mask)
        pos_t = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        dec = self.drop(self.dec_emb(tgt) + self.pos_emb(pos_t))
        for b in self.dec_blocks: dec = b(dec, enc, src_mask, tgt_mask)
        return self.lm_head(dec)

class SyntheticSeq2SeqDataset(Dataset):
    def __init__(self, size, vocab_size, seq_len):
        self.size, self.vocab_size, self.seq_len = size, vocab_size, seq_len
    def __len__(self): return self.size
    def __getitem__(self, i):
        src = torch.randint(1, self.vocab_size, (self.seq_len,))
        tgt = torch.randint(1, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return src, tgt, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = BART(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"encoder_decoder","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"bart_4_10M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticSeq2SeqDataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for src, tgt, labels in loader:
            src, tgt, labels = src.to(dev), tgt.to(dev), labels.to(dev)
            optim.zero_grad()
            logits = model(src, tgt)
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
