#!/usr/bin/env python3
"""BERT - batch=16, ~5M params"""
import time,json,math,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 16
HIDDEN = 256
N_LAYERS = 4
ATTN_HEADS = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
VOCAB_SIZE = 30000
SEQ_LEN = 128

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
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.output_linear(x)

class FeedForward(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 4), GELU(), nn.Dropout(dropout), nn.Linear(hidden * 4, hidden), nn.Dropout(dropout))
    def forward(self, x):
        return self.net(x)

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

class BERT(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, attn_heads, seq_len, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.token_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.segment_emb = nn.Embedding(3, hidden)
        self.blocks = nn.ModuleList([TransformerBlock(hidden, attn_heads, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, segment_info):
        mask = (x > 0).unsqueeze(1).unsqueeze(2)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.dropout(self.token_emb(x) + self.pos_emb(pos) + self.segment_emb(segment_info))
        for block in self.blocks:
            x = block(x, mask)
        return x

class BERTForPretraining(nn.Module):
    def __init__(self, vocab_size, hidden, n_layers, attn_heads, seq_len):
        super().__init__()
        self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, seq_len)
        self.next_sentence = nn.Linear(hidden, 2)
        self.mask_lm = nn.Linear(hidden, vocab_size)
        
    def forward(self, x, segment_info):
        x = self.bert(x, segment_info)
        return self.next_sentence(x[:, 0]), self.mask_lm(x)

class SyntheticBERTDataset(Dataset):
    def __init__(self, size, vocab_size, seq_len):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len
    def __len__(self): return self.size
    def __getitem__(self, idx):
        bert_input = torch.randint(1, self.vocab_size, (self.seq_len,))
        segment_label = torch.zeros(self.seq_len, dtype=torch.long)
        segment_label[self.seq_len//2:] = 1
        is_next = torch.randint(0, 2, (1,)).item()
        bert_label = torch.randint(0, self.vocab_size, (self.seq_len,))
        return bert_input, segment_label, is_next, bert_label

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = BERTForPretraining(VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, SEQ_LEN).to(dev)
    param_count = count_params(model)
    
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "transformer", "batch_size": BATCH_SIZE, "param_count": param_count}))
        print("###END_FEATURES###")
        print("=" * 60)
        print(f"bert_16_5M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{param_count:,}")
        print("=" * 60)
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticBERTDataset(NUM_SAMPLES, VOCAB_SIZE, SEQ_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for bert_input, segment_label, is_next, bert_label in loader:
            bert_input, segment_label = bert_input.to(dev), segment_label.to(dev)
            is_next, bert_label = is_next.to(dev), bert_label.to(dev)
            optim.zero_grad()
            next_sent_out, mask_lm_out = model(bert_input, segment_label)
            next_loss = crit(next_sent_out, is_next)
            mask_loss = crit(mask_lm_out.transpose(1, 2), bert_label)
            loss = next_loss + mask_loss
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    total_time = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary")
        print(f"    GPUs:{ws} | Total time:{total_time:.2f}s | Avg throughput:{tsp/total_time:.1f} samples/sec")
        print("###RESULTS###")
        print(json.dumps({"batch_size": BATCH_SIZE, "param_count": param_count, "gpu_count": ws, "total_time_sec": round(total_time, 2), "avg_throughput": round(tsp/total_time, 1)}))
        print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
