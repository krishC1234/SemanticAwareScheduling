#!/usr/bin/env python3
"""Speech Transformer - batch=32, ~5M params"""
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
VOCAB_SIZE = 4233
ATTN_HEADS = 8
INPUT_DIM = 80
MAX_SEQ_LEN = 512
MAX_TARGET_LEN = 128
DROPOUT = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, heads, dropout=0.1):
        super().__init__()
        self.h = heads
        self.d_k = hidden // heads
        self.w_q = nn.Linear(hidden, hidden)
        self.w_k = nn.Linear(hidden, hidden)
        self.w_v = nn.Linear(hidden, hidden)
        self.w_o = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.w_q(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None: scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        attn = torch.nan_to_num(attn, 0.0)
        x = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.w_o(x)

class FeedForward(nn.Module):
    def __init__(self, hidden, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, hidden, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden, heads, dropout)
        self.ff = FeedForward(hidden, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        x = x + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)
        x = x + self.dropout2(self.ff(x))
        x = self.norm2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hidden, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden, heads, dropout)
        self.cross_attn = MultiHeadAttention(hidden, heads, dropout)
        self.ff = FeedForward(hidden, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        x = x + self.dropout1(self.self_attn(x, x, x, tgt_mask))
        x = self.norm1(x)
        x = x + self.dropout2(self.cross_attn(x, enc_out, enc_out, memory_mask))
        x = self.norm2(x)
        x = x + self.dropout3(self.ff(x))
        x = self.norm3(x)
        return x

class SpeechTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden, n_layers, heads, ff_dim, max_seq_len, max_target_len, dropout=0.1):
        super().__init__()
        self.encoder_proj = nn.Linear(input_dim, hidden)
        self.encoder_pe = PositionalEncoding(hidden, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden, heads, ff_dim, dropout) for _ in range(n_layers)])
        self.decoder_emb = nn.Embedding(vocab_size, hidden)
        self.decoder_pe = PositionalEncoding(hidden, max_target_len, dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden, heads, ff_dim, dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def make_pad_mask(self, lengths, max_len):
        bs = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(bs, -1) < lengths.unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(2)
    
    def make_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, padded_input, input_lengths, padded_target):
        bs, enc_seq_len, _ = padded_input.size()
        _, dec_seq_len = padded_target.size()
        enc_mask = self.make_pad_mask(input_lengths, enc_seq_len)
        x = self.encoder_proj(padded_input)
        x = self.encoder_pe(x)
        for layer in self.encoder_layers: x = layer(x, enc_mask)
        enc_out = x
        tgt_mask = self.make_causal_mask(dec_seq_len, padded_target.device)
        memory_mask = self.make_pad_mask(input_lengths, enc_seq_len)
        y = self.decoder_emb(padded_target)
        y = self.decoder_pe(y)
        for layer in self.decoder_layers: y = layer(y, enc_out, tgt_mask, memory_mask)
        return self.output_proj(y)

class SyntheticSpeechDataset(Dataset):
    def __init__(self, size, input_dim, max_seq_len, max_target_len, vocab_size):
        self.size, self.input_dim, self.max_seq_len = size, input_dim, max_seq_len
        self.max_target_len, self.vocab_size = max_target_len, vocab_size
    def __len__(self): return self.size
    def __getitem__(self, i):
        seq_len = torch.randint(self.max_seq_len // 2, self.max_seq_len, (1,)).item()
        padded_input = torch.zeros(self.max_seq_len, self.input_dim)
        padded_input[:seq_len] = torch.randn(seq_len, self.input_dim)
        input_length = torch.tensor(seq_len)
        target_len = torch.randint(self.max_target_len // 2, self.max_target_len, (1,)).item()
        padded_target = torch.zeros(self.max_target_len, dtype=torch.long)
        padded_target[:target_len] = torch.randint(1, self.vocab_size, (target_len,))
        labels = torch.zeros(self.max_target_len, dtype=torch.long)
        labels[:target_len] = torch.randint(1, self.vocab_size, (target_len,))
        return padded_input, input_length, padded_target, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    ff_dim = HIDDEN * 4
    model = SpeechTransformer(INPUT_DIM, VOCAB_SIZE, HIDDEN, N_LAYERS, ATTN_HEADS, ff_dim, MAX_SEQ_LEN, MAX_TARGET_LEN, DROPOUT).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_transformer","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"speech_transformer_32_5M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticSpeechDataset(NUM_SAMPLES, INPUT_DIM, MAX_SEQ_LEN, MAX_TARGET_LEN, VOCAB_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for padded_input, input_lengths, padded_target, labels in loader:
            padded_input, input_lengths = padded_input.to(dev), input_lengths.to(dev)
            padded_target, labels = padded_target.to(dev), labels.to(dev)
            optim.zero_grad()
            logits = model(padded_input, input_lengths, padded_target)
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
