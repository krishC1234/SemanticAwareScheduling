#!/usr/bin/env python3
"""Mozilla TTS Speaker Encoder - batch=64, small params (~2M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
EMBEDDING_DIM = 128
EPOCHS = 3
NUM_SAMPLES = 2000
N_MELS = 80
UTTERANCE_LEN = 160
NUM_SPEAKERS = 100
LR = 1e-4

class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels=N_MELS, hd=HIDDEN_DIM, nl=NUM_LAYERS, ed=EMBEDDING_DIM):
        super().__init__()
        self.lstm = nn.LSTM(n_mels, hd, nl, batch_first=True, dropout=0.1 if nl > 1 else 0.0)
        self.proj = nn.Linear(hd, ed)
        for n, p in self.lstm.named_parameters():
            if 'weight_ih' in n: nn.init.xavier_uniform_(p)
            elif 'weight_hh' in n: nn.init.orthogonal_(p)
            elif 'bias' in n: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)
    def forward(self, mels):
        x = mels.transpose(1, 2)
        self.lstm.flatten_parameters()
        _, (h, _) = self.lstm(x)
        return F.normalize(self.proj(h[-1]), p=2, dim=1)

class SimplifiedGE2ELoss(nn.Module):
    def __init__(self, w=10.0, b=-5.0):
        super().__init__()
        self.w, self.b = nn.Parameter(torch.tensor(w)), nn.Parameter(torch.tensor(b))
    def forward(self, emb, ns, ups):
        emb = emb.view(ns, ups, -1)
        cent = emb.mean(dim=1)
        sim = self.w * torch.mm(emb.view(-1, emb.size(-1)), cent.t()) + self.b
        tgt = torch.arange(ns, device=emb.device).unsqueeze(1).expand(-1, ups).reshape(-1)
        return F.cross_entropy(sim, tgt)

class SpeakerDataset(Dataset):
    def __init__(self, sz, ns, ups, nm, ul):
        self.sz, self.ns, self.nm, self.ul = sz, ns, nm, ul
        self.means = torch.randn(ns, nm) * 0.5
        self.stds = torch.rand(ns, nm) * 0.3 + 0.2
    def __len__(self): return self.sz
    def __getitem__(self, i):
        sid = i % self.ns
        mel = torch.randn(self.nm, self.ul) * self.stds[sid].unsqueeze(1) + self.means[sid].unsqueeze(1)
        return mel, sid

def collate_speaker(batch):
    m, s = zip(*batch)
    return torch.stack(m), torch.tensor(s)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = SpeakerEncoder().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_tts_speaker_encoder","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"tts_64_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SpeakerDataset(NUM_SAMPLES, NUM_SPEAKERS, 10, N_MELS, UTTERANCE_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_speaker)
    crit = SimplifiedGE2ELoss().to(dev)
    opt = torch.optim.Adam(list(model.parameters()) + list(crit.parameters()), lr=LR)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for mels, _ in loader:
            mels = mels.to(dev)
            opt.zero_grad()
            emb = model(mels)
            ns, ups = min(BATCH_SIZE // 2, 10), BATCH_SIZE // min(BATCH_SIZE // 2, 10)
            crit(emb[:ns * ups], ns, ups).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
