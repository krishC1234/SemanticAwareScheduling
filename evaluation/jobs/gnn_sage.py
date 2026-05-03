#!/usr/bin/env python3
"""GraphSAGE - node classification, batch=512, ~0.3M params

GraphSAGE learns node embeddings by sampling and aggregating features
from local neighborhoods. Uses mean aggregation with 2 message-passing
layers. Trained on a synthetic graph with random node features.

Reference: Hamilton et al., "Inductive Representation Learning on
Large Graphs", NeurIPS 2017
"""
import time, json, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 512

# === FIXED ===
EPOCHS = 50
NUM_NODES = 100000
NUM_EDGES = 500000
IN_FEATURES = 128
HIDDEN = 256
OUT_CLASSES = 40
NUM_LAYERS = 2
NUM_SAMPLES = 100000     # one pass per epoch = all nodes
NEIGHBOR_SAMPLES = 25     # neighbors sampled per node per layer

# ---------------------------------------------------------------------------
# GraphSAGE components
# ---------------------------------------------------------------------------
class SAGEConv(nn.Module):
    """Single GraphSAGE layer with mean aggregation."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim, bias=True)

    def forward(self, x, adj_samples):
        """x: (B, in_dim), adj_samples: (B, K, in_dim) — sampled neighbor features."""
        neigh_mean = adj_samples.mean(dim=1)  # (B, in_dim)
        h = torch.cat([x, neigh_mean], dim=-1)  # (B, 2*in_dim)
        h = self.linear(h)
        return F.normalize(h, p=2, dim=-1)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE for node classification. ~0.3M parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(IN_FEATURES, HIDDEN)
        self.conv2 = SAGEConv(HIDDEN, HIDDEN)
        self.classifier = nn.Linear(HIDDEN, OUT_CLASSES)

    def forward(self, x, neigh1, neigh2):
        """
        x: (B, in_features) — target node features
        neigh1: (B, K, in_features) — 1-hop neighbor features
        neigh2: (B, K, HIDDEN) — 2-hop neighbor features (pre-embedded)
        """
        h = F.relu(self.conv1(x, neigh1))
        h = F.relu(self.conv2(h, neigh2))
        return self.classifier(h)


class SyntheticGraphDataset(Dataset):
    """Generates synthetic graph mini-batches with sampled neighborhoods.
    Each sample = one target node + its sampled 1-hop and 2-hop neighbors."""

    def __init__(self, num_nodes, in_features, num_classes, k):
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.num_classes = num_classes
        self.k = k
        # Persistent node features and labels (shared across epochs)
        self.node_feats = torch.randn(num_nodes, in_features)
        self.labels = torch.randint(0, num_classes, (num_nodes,))

    def __len__(self): return self.num_nodes

    def __getitem__(self, idx):
        x = self.node_feats[idx]
        label = self.labels[idx]
        # Sample random neighbors (approximation of graph sampling)
        neigh1_idx = torch.randint(0, self.num_nodes, (self.k,))
        neigh1 = self.node_feats[neigh1_idx]
        # 2-hop: sample neighbors of neighbors (random for synthetic)
        neigh2 = torch.randn(self.k, IN_FEATURES)  # placeholder for 2-hop aggregated
        return x, neigh1, neigh2, label


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = GraphSAGE().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "gnn", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"gnn_sage | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticGraphDataset(NUM_NODES, IN_FEATURES, OUT_CLASSES, NEIGHBOR_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for x, neigh1, neigh2, labels in loader:
            x, neigh1, neigh2, labels = (
                x.to(dev), neigh1.to(dev), neigh2.to(dev), labels.to(dev)
            )
            optim.zero_grad()
            out = model(x, neigh1, neigh2)
            loss = crit(out, labels)
            loss.backward()
            optim.step()
        tsp += len(ds)
        if rank == 0:
            print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | "
                  f"throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | "
              f"Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###")
        print(json.dumps({"batch_size": BATCH_SIZE, "param_count": pc,
                           "gpu_count": ws, "total_time_sec": round(tt, 2),
                           "avg_throughput": round(tsp / tt, 1)}))
        print("###END_RESULTS###")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
