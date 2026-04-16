#!/usr/bin/env python3
"""GIN (Graph Isomorphism Network) - batch=32, small params (~0.3M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32
HIDDEN = 64

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
NUM_NODES = 50       # Avg nodes per graph (molecular graphs)
IN_FEATURES = 7      # Node feature dim (atom features)
NUM_CLASSES = 2      # Binary classification (e.g., mutagenicity)
NUM_LAYERS = 5       # GIN layers
EDGE_PROB = 0.15     # Edge probability for random graphs

class MLP(nn.Module):
    """Multi-layer perceptron for GIN"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class GINConv(nn.Module):
    """Graph Isomorphism Network convolution layer"""
    def __init__(self, in_dim, out_dim, eps=0.0, train_eps=True):
        super().__init__()
        self.mlp = MLP(in_dim, out_dim, out_dim, num_layers=2)
        self.eps = nn.Parameter(torch.tensor([eps])) if train_eps else eps
        self.train_eps = train_eps
    
    def forward(self, x, adj):
        # x: (N, F), adj: (N, N) sparse or dense
        # GIN update: h' = MLP((1 + eps) * h + sum of neighbor h)
        neighbor_sum = torch.sparse.mm(adj, x) if adj.is_sparse else torch.mm(adj, x)
        out = (1 + self.eps) * x + neighbor_sum
        return self.mlp(out)

class GIN(nn.Module):
    """Graph Isomorphism Network for graph classification"""
    def __init__(self, in_features, hidden, num_classes, num_layers, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GIN convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer: in_features -> hidden
        self.convs.append(GINConv(in_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))
        
        # Hidden layers: hidden -> hidden
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        
        # Graph-level readout + classifier
        # Concatenate representations from all layers (JK connection)
        self.fc1 = nn.Linear(hidden * num_layers, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x, adj, batch_idx, num_graphs):
        """
        x: (total_nodes, in_features) - all node features
        adj: (total_nodes, total_nodes) - block diagonal adjacency
        batch_idx: (total_nodes,) - graph membership for each node
        num_graphs: number of graphs in batch
        """
        hidden_states = []
        
        # GIN layers with residual-style JK connections
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, adj)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hidden_states.append(h)
        
        # Graph-level readout: sum pooling per graph for each layer
        graph_reprs = []
        for h in hidden_states:
            # Sum pooling: aggregate node features per graph
            graph_repr = torch.zeros(num_graphs, h.size(1), device=h.device)
            graph_repr.scatter_add_(0, batch_idx.unsqueeze(1).expand(-1, h.size(1)), h)
            graph_reprs.append(graph_repr)
        
        # Concatenate all layer representations (Jumping Knowledge)
        graph_repr = torch.cat(graph_reprs, dim=1)
        
        # Final classifier
        out = F.relu(self.fc1(graph_repr))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.fc2(out)

class SyntheticGraphDataset(Dataset):
    """Generate random graphs for graph classification"""
    def __init__(self, size, num_nodes, in_features, num_classes, edge_prob):
        self.size = size
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.num_classes = num_classes
        self.edge_prob = edge_prob
    
    def __len__(self): return self.size
    
    def __getitem__(self, i):
        # Random number of nodes per graph (varying size)
        n = torch.randint(self.num_nodes // 2, self.num_nodes + 1, (1,)).item()
        
        # Random node features
        x = torch.randn(n, self.in_features)
        
        # Random edges (Erdos-Renyi graph)
        edge_mask = torch.rand(n, n) < self.edge_prob
        edge_mask = edge_mask | edge_mask.t()  # Symmetric
        edge_mask.fill_diagonal_(False)  # No self-loops
        adj = edge_mask.float()
        
        # Random label
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        return x, adj, label

def collate_graphs(batch):
    """Collate graphs into a batched graph (block diagonal adjacency)"""
    xs, adjs, labels = zip(*batch)
    
    # Count total nodes and create batch index
    num_nodes = [x.size(0) for x in xs]
    total_nodes = sum(num_nodes)
    num_graphs = len(batch)
    
    # Concatenate node features
    x = torch.cat(xs, dim=0)
    
    # Create block diagonal adjacency matrix
    adj = torch.zeros(total_nodes, total_nodes)
    offset = 0
    for i, a in enumerate(adjs):
        n = a.size(0)
        adj[offset:offset+n, offset:offset+n] = a
        offset += n
    
    # Create batch index (which graph each node belongs to)
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_nodes)])
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return x, adj, batch_idx, num_graphs, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = GIN(IN_FEATURES, HIDDEN, NUM_CLASSES, NUM_LAYERS).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gnn_gin","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"gin_32_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticGraphDataset(NUM_SAMPLES, NUM_NODES, IN_FEATURES, NUM_CLASSES, EDGE_PROB)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, 
                        pin_memory=True, drop_last=True, collate_fn=collate_graphs)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for x, adj, batch_idx, num_graphs, labels in loader:
            x, adj, batch_idx, labels = x.to(dev), adj.to(dev), batch_idx.to(dev), labels.to(dev)
            optim.zero_grad()
            out = model(x, adj, batch_idx, num_graphs)
            loss = crit(out, labels)
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
