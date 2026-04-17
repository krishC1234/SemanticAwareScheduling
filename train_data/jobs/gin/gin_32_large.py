#!/usr/bin/env python3
"""GIN (Graph Isomorphism Network) - batch=32, large params (~2M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 32
HIDDEN = 256
EPOCHS = 3
NUM_SAMPLES = 2000
NUM_NODES = 50
IN_FEATURES = 7
NUM_CLASSES = 2
NUM_LAYERS = 5
EDGE_PROB = 0.15

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0, train_eps=True):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim, num_layers=2)
        self.eps = nn.Parameter(torch.tensor([eps])) if train_eps else eps
    def forward(self, x, adj):
        neighbor_sum = torch.sparse.mm(adj, x) if adj.is_sparse else torch.mm(adj, x)
        return self.mlp((1 + self.eps) * x + neighbor_sum)

class GIN(nn.Module):
    def __init__(self, in_features, hidden, num_classes, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers, self.dropout = num_layers, dropout
        self.convs = nn.ModuleList([GINConv(in_features if i == 0 else hidden, hidden) for i in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.fc1 = nn.Linear(hidden * num_layers, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x, adj, batch_idx, num_graphs):
        hidden_states = []
        h = x
        for i in range(self.num_layers):
            h = F.dropout(F.relu(self.bns[i](self.convs[i](h, adj))), p=self.dropout, training=self.training)
            hidden_states.append(h)
        graph_reprs = []
        for h in hidden_states:
            gr = torch.zeros(num_graphs, h.size(1), device=h.device)
            gr.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, h.size(1)), h)
            graph_reprs.append(gr)
        return self.fc2(F.dropout(F.relu(self.fc1(torch.cat(graph_reprs, dim=1))), p=self.dropout, training=self.training))

class SyntheticGraphDataset(Dataset):
    def __init__(self, size, num_nodes, in_features, num_classes, edge_prob):
        self.size, self.num_nodes, self.in_features, self.num_classes, self.edge_prob = size, num_nodes, in_features, num_classes, edge_prob
    def __len__(self): return self.size
    def __getitem__(self, i):
        n = torch.randint(self.num_nodes // 2, self.num_nodes + 1, (1,)).item()
        x = torch.randn(n, self.in_features)
        edge_mask = torch.rand(n, n) < self.edge_prob
        edge_mask = edge_mask | edge_mask.t()
        edge_mask.fill_diagonal_(False)
        return x, edge_mask.float(), torch.randint(0, self.num_classes, (1,)).item()

def collate_graphs(batch):
    xs, adjs, labels = zip(*batch)
    num_nodes = [x.size(0) for x in xs]
    total_nodes = sum(num_nodes)
    x = torch.cat(xs, dim=0)
    adj = torch.zeros(total_nodes, total_nodes)
    offset = 0
    for a in adjs:
        n = a.size(0)
        adj[offset:offset+n, offset:offset+n] = a
        offset += n
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes)])
    return x, adj, batch_idx, len(batch), torch.tensor(labels, dtype=torch.long)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = GIN(IN_FEATURES, HIDDEN, NUM_CLASSES, NUM_LAYERS).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gnn_gin","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"gin_32_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticGraphDataset(NUM_SAMPLES, NUM_NODES, IN_FEATURES, NUM_CLASSES, EDGE_PROB)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_graphs)
    optim, crit = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.CrossEntropyLoss()
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for x, adj, batch_idx, num_graphs, labels in loader:
            x, adj, batch_idx, labels = x.to(dev), adj.to(dev), batch_idx.to(dev), labels.to(dev)
            optim.zero_grad(); crit(model(x, adj, batch_idx, num_graphs), labels).backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
