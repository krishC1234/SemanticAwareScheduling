#!/usr/bin/env python3
"""Whisper - batch=4, large params (~80M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 4
D_MODEL = 512
N_HEADS = 8
N_ENCODER_LAYERS = 6
N_DECODER_LAYERS = 6
EPOCHS = 3
NUM_SAMPLES = 500
N_MELS = 80
AUDIO_SEQ_LEN = 3000
MAX_TEXT_LEN = 448
VOCAB_SIZE = 51865
DROPOUT = 0.1
LR = 1e-4

class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class AudioEncoder(nn.Module):
    def __init__(self, n_mels, d, nh, nl, dff, drop):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.pe = SinusoidalPE(d, AUDIO_SEQ_LEN)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, nh, dff, drop, 'gelu', batch_first=True), nl)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = self.pe(x.transpose(1, 2))
        return self.ln(self.tf(x))

class TextDecoder(nn.Module):
    def __init__(self, vs, d, nh, nl, dff, drop, ml):
        super().__init__()
        self.emb = nn.Embedding(vs, d)
        self.pe = SinusoidalPE(d, ml)
        self.tf = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, nh, dff, drop, 'gelu', batch_first=True), nl)
        self.ln = nn.LayerNorm(d)
        self.proj = nn.Linear(d, vs, bias=False)
        self.proj.weight = self.emb.weight
    def forward(self, tok, enc_out, mask=None):
        x = self.pe(self.emb(tok))
        return self.proj(self.ln(self.tf(x, enc_out, tgt_mask=mask)))

class Whisper(nn.Module):
    def __init__(self):
        super().__init__()
        dff = 4 * D_MODEL
        self.enc = AudioEncoder(N_MELS, D_MODEL, N_HEADS, N_ENCODER_LAYERS, dff, DROPOUT)
        self.dec = TextDecoder(VOCAB_SIZE, D_MODEL, N_HEADS, N_DECODER_LAYERS, dff, DROPOUT, MAX_TEXT_LEN)
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
    def forward(self, audio, tok):
        enc = self.enc(audio)
        mask = torch.triu(torch.ones(tok.size(1), tok.size(1), device=tok.device), 1).bool()
        return self.dec(tok, enc, mask)

class SyntheticASRDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i):
        tl = torch.randint(10, MAX_TEXT_LEN // 2, (1,)).item()
        return torch.randn(N_MELS, AUDIO_SEQ_LEN), torch.randint(1, VOCAB_SIZE, (tl,))

def collate_asr(batch):
    a, t = zip(*batch)
    ml = max(x.size(0) for x in t)
    tp = torch.zeros(len(t), ml, dtype=torch.long)
    for i, x in enumerate(t): tp[i, :x.size(0)] = x
    return torch.stack(a), tp

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = Whisper().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_whisper","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"whisper_4_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticASRDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_asr)
    crit, opt = nn.CrossEntropyLoss(ignore_index=0), torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for audio, tok in loader:
            audio, tok = audio.to(dev), tok.to(dev)
            opt.zero_grad()
            logits = model(audio, tok[:, :-1])
            crit(logits.reshape(-1, VOCAB_SIZE), tok[:, 1:].reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
