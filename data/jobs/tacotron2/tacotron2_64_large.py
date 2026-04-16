#!/usr/bin/env python3
"""Tacotron2 - batch=64, large params (~30M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 64
ENCODER_DIM = 512
DECODER_RNN_DIM = 1024
ATTENTION_RNN_DIM = 1024
PRENET_DIM = 256
ATTENTION_DIM = 128
POSTNET_DIM = 512
EPOCHS = 3
NUM_SAMPLES = 1000
N_SYMBOLS = 148
SYMBOLS_EMBEDDING_DIM = 512
ENCODER_KERNEL_SIZE = 5
ENCODER_N_CONVS = 3
N_MEL_CHANNELS = 80
ATTENTION_LOCATION_N_FILTERS = 32
ATTENTION_LOCATION_KERNEL_SIZE = 31
POSTNET_KERNEL_SIZE = 5
POSTNET_N_CONVS = 5
P_DROPOUT = 0.1
LR = 1e-3

class LinearNorm(nn.Module):
    def __init__(self, in_d, out_d, bias=True, w_init='linear'):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(w_init))
    def forward(self, x): return self.linear(x)

class ConvNorm(nn.Module):
    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=None, dil=1, bias=True, w_init='linear'):
        super().__init__()
        pad = pad if pad is not None else (ks - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, ks, stride, pad, dil, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init))
    def forward(self, x): return self.conv(x)

class Prenet(nn.Module):
    def __init__(self, in_d, sizes):
        super().__init__()
        self.layers = nn.ModuleList([LinearNorm(ind, outd, bias=False) for ind, outd in zip([in_d]+sizes[:-1], sizes)])
    def forward(self, x):
        for l in self.layers: x = F.dropout(F.relu(l(x)), 0.5, training=True)
        return x

class LocationLayer(nn.Module):
    def __init__(self, nf, ks, ad):
        super().__init__()
        self.loc_conv = ConvNorm(2, nf, ks, pad=(ks-1)//2, bias=False)
        self.loc_dense = LinearNorm(nf, ad, bias=False, w_init='tanh')
    def forward(self, x): return self.loc_dense(self.loc_conv(x).transpose(1,2))

class Attention(nn.Module):
    def __init__(self, ard, ed, ad, nf, ks):
        super().__init__()
        self.query = LinearNorm(ard, ad, bias=False, w_init='tanh')
        self.memory = LinearNorm(ed, ad, bias=False, w_init='tanh')
        self.v = LinearNorm(ad, 1, bias=False)
        self.loc = LocationLayer(nf, ks, ad)
    def forward(self, ah, mem, pm, awc, mask):
        align = self.v(torch.tanh(self.query(ah.unsqueeze(1)) + self.loc(awc) + pm)).squeeze(-1)
        if mask is not None: align.masked_fill_(mask, -float('inf'))
        aw = F.softmax(align, dim=1)
        return torch.bmm(aw.unsqueeze(1), mem).squeeze(1), aw

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N_SYMBOLS, SYMBOLS_EMBEDDING_DIM)
        self.convs = nn.ModuleList([nn.Sequential(ConvNorm(SYMBOLS_EMBEDDING_DIM, SYMBOLS_EMBEDDING_DIM, ENCODER_KERNEL_SIZE, pad=(ENCODER_KERNEL_SIZE-1)//2, w_init='relu'), nn.BatchNorm1d(SYMBOLS_EMBEDDING_DIM), nn.ReLU(), nn.Dropout(P_DROPOUT)) for _ in range(ENCODER_N_CONVS)])
        self.lstm = nn.LSTM(SYMBOLS_EMBEDDING_DIM, ENCODER_DIM//2, 1, batch_first=True, bidirectional=True)
    def forward(self, x, lens):
        x = self.emb(x).transpose(1,2)
        for c in self.convs: x = c(x)
        x = pack_padded_sequence(x.transpose(1,2), lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.lstm(x)
        o, _ = pad_packed_sequence(o, batch_first=True)
        return o

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.prenet = Prenet(N_MEL_CHANNELS, [PRENET_DIM, PRENET_DIM])
        self.attn_rnn = nn.LSTMCell(PRENET_DIM + ENCODER_DIM, ATTENTION_RNN_DIM)
        self.attn = Attention(ATTENTION_RNN_DIM, ENCODER_DIM, ATTENTION_DIM, ATTENTION_LOCATION_N_FILTERS, ATTENTION_LOCATION_KERNEL_SIZE)
        self.dec_rnn = nn.LSTMCell(ATTENTION_RNN_DIM + ENCODER_DIM, DECODER_RNN_DIM)
        self.proj = LinearNorm(DECODER_RNN_DIM + ENCODER_DIM, N_MEL_CHANNELS)
        self.gate = LinearNorm(DECODER_RNN_DIM + ENCODER_DIM, 1, w_init='sigmoid')

    def forward(self, mem, dec_in, mem_lens):
        B, T, _ = mem.size()
        go = mem.new_zeros(B, N_MEL_CHANNELS)
        dec_in = torch.cat((go.unsqueeze(0), dec_in.permute(2,0,1)), 0)
        dec_in = self.prenet(dec_in)
        ah, ac = mem.new_zeros(B, ATTENTION_RNN_DIM), mem.new_zeros(B, ATTENTION_RNN_DIM)
        dh, dc = mem.new_zeros(B, DECODER_RNN_DIM), mem.new_zeros(B, DECODER_RNN_DIM)
        aw, awc, ctx = mem.new_zeros(B, T), mem.new_zeros(B, T), mem.new_zeros(B, ENCODER_DIM)
        pm = self.attn.memory(mem)
        mask = ~(torch.arange(T, device=mem.device) < mem_lens.unsqueeze(1))
        mels, gates = [], []
        for i in range(dec_in.size(0)-1):
            ah, ac = self.attn_rnn(torch.cat((dec_in[i], ctx), -1), (ah, ac))
            ah = F.dropout(ah, P_DROPOUT, self.training)
            ctx, aw = self.attn(ah, mem, pm, torch.stack((aw, awc), 1), mask)
            awc = awc + aw
            dh, dc = self.dec_rnn(torch.cat((ah, ctx), -1), (dh, dc))
            dh = F.dropout(dh, P_DROPOUT, self.training)
            cat = torch.cat((dh, ctx), 1)
            mels.append(self.proj(cat))
            gates.append(self.gate(cat).squeeze(1))
        return torch.stack(mels, 2), torch.stack(gates, 1)

class Postnet(nn.Module):
    def __init__(self):
        super().__init__()
        cs = [nn.Sequential(ConvNorm(N_MEL_CHANNELS if i==0 else POSTNET_DIM, POSTNET_DIM if i<POSTNET_N_CONVS-1 else N_MEL_CHANNELS, POSTNET_KERNEL_SIZE, pad=(POSTNET_KERNEL_SIZE-1)//2, w_init='tanh' if i<POSTNET_N_CONVS-1 else 'linear'), nn.BatchNorm1d(POSTNET_DIM if i<POSTNET_N_CONVS-1 else N_MEL_CHANNELS), nn.Tanh() if i<POSTNET_N_CONVS-1 else nn.Identity(), nn.Dropout(P_DROPOUT)) for i in range(POSTNET_N_CONVS)]
        self.convs = nn.ModuleList(cs)
    def forward(self, x):
        for c in self.convs: x = c(x)
        return x

class Tacotron2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
        self.post = Postnet()
    def forward(self, txt, tl, mel, ml):
        enc = self.enc(txt, tl)
        mo, go = self.dec(enc, mel, tl)
        return mo, mo + self.post(mo), go

class Tacotron2Loss(nn.Module):
    def forward(self, mo, mop, go, mt, gt):
        return nn.MSELoss()(mo, mt) + nn.MSELoss()(mop, mt) + nn.BCEWithLogitsLoss()(go, gt)

class SyntheticTTSDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i):
        tl, ml = torch.randint(20, 100, (1,)).item(), torch.randint(50, 200, (1,)).item()
        g = torch.zeros(ml); g[-1] = 1.0
        return torch.randint(0, N_SYMBOLS, (tl,)), torch.randn(N_MEL_CHANNELS, ml), g

def collate_tts(batch):
    t, m, g = zip(*batch)
    tl, ml = torch.tensor([x.size(0) for x in t]), torch.tensor([x.size(1) for x in m])
    tp = pad_sequence(t, batch_first=True)
    mm = max(x.size(1) for x in m)
    mp, gp = torch.zeros(len(m), N_MEL_CHANNELS, mm), torch.zeros(len(m), mm)
    for i, (x, y) in enumerate(zip(m, g)): mp[i,:,:x.size(1)] = x; gp[i,:y.size(0)] = y
    return tp, tl, mp, ml, gp

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = Tacotron2().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_tacotron2","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"tacotron2_64_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticTTSDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_tts)
    crit, opt = Tacotron2Loss(), torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for txt, tl, mel, ml, gate in loader:
            txt, tl, mel, gate = txt.to(dev), tl.to(dev), mel.to(dev), gate.to(dev)
            opt.zero_grad()
            mo, mop, go = model(txt, tl, mel, tl)
            crit(mo, mop, go, mel, gate).backward()
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
