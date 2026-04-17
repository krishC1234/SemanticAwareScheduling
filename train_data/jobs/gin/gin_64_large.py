#!/usr/bin/env python3
"""GIN (Graph Isomorphism Network) - batch=64, large params (~2M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 64
HIDDEN = 256
EPOCHS = 3
NUM_SAMPLES = 2000
NUM_NODES = 50
IN_FEATURES = 7
NUM_CLASSES = 2
NUM_LAYERS = 5
EDGE_PROB = 0.15

class MLP(nn.Module):
    def __init__(self, in_dim, hid, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(True), nn.Linear(hid, out_dim))
    def forward(self, x): return self.net(x)

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim)
        self.eps = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj):
        return self.mlp((1 + self.eps) * x + torch.mm(adj, x))

class GIN(nn.Module):
    def __init__(self, in_f, hid, n_cls, n_layers, drop=0.5):
        super().__init__()
        self.n_layers, self.drop = n_layers, drop
        self.convs = nn.ModuleList([GINConv(in_f if i == 0 else hid, hid) for i in range(n_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid) for _ in range(n_layers)])
        self.fc1, self.fc2 = nn.Linear(hid * n_layers, hid), nn.Linear(hid, n_cls)
    def forward(self, x, adj, batch_idx, ng):
        hs = []
        h = x
        for i in range(self.n_layers):
            h = F.dropout(F.relu(self.bns[i](self.convs[i](h, adj))), p=self.drop, training=self.training)
            hs.append(h)
        grs = []
        for h in hs:
            gr = torch.zeros(ng, h.size(1), device=h.device)
            gr.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, h.size(1)), h)
            grs.append(gr)
        return self.fc2(F.dropout(F.relu(self.fc1(torch.cat(grs, 1))), p=self.drop, training=self.training))

class SyntheticGraphDataset(Dataset):
    def __init__(self, sz, nn, inf, nc, ep): self.sz, self.nn, self.inf, self.nc, self.ep = sz, nn, inf, nc, ep
    def __len__(self): return self.sz
    def __getitem__(self, i):
        n = torch.randint(self.nn//2, self.nn+1, (1,)).item()
        em = torch.rand(n, n) < self.ep; em = em | em.t(); em.fill_diagonal_(False)
        return torch.randn(n, self.inf), em.float(), torch.randint(0, self.nc, (1,)).item()

def collate_graphs(batch):
    xs, adjs, labels = zip(*batch)
    nn = [x.size(0) for x in xs]; tn = sum(nn)
    x = torch.cat(xs, 0)
    adj = torch.zeros(tn, tn); off = 0
    for a in adjs: n = a.size(0); adj[off:off+n, off:off+n] = a; off += n
    bi = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(nn)])
    return x, adj, bi, len(batch), torch.tensor(labels, dtype=torch.long)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = GIN(IN_FEATURES, HIDDEN, NUM_CLASSES, NUM_LAYERS).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gnn_gin","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"gin_64_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticGraphDataset(NUM_SAMPLES, NUM_NODES, IN_FEATURES, NUM_CLASSES, EDGE_PROB)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_graphs)
    optim, crit = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.CrossEntropyLoss()
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for x, adj, bi, ng, lbl in loader:
            x, adj, bi, lbl = x.to(dev), adj.to(dev), bi.to(dev), lbl.to(dev)
            optim.zero_grad(); crit(model(x, adj, bi, ng), lbl).backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
