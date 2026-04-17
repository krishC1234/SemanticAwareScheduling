#!/usr/bin/env python3
"""EdgeCNN (Dynamic Graph CNN) - batch=16, small params (~0.5M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 16
HIDDEN = 64

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
NUM_POINTS = 1024  # Points per point cloud
IN_CHANNELS = 3    # x, y, z coordinates
NUM_CLASSES = 40   # ModelNet40 classes
K = 20             # k-nearest neighbors

def knn(x, k):
    """Compute k-nearest neighbors in feature space"""
    # x: (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """Construct edge features for EdgeConv"""
    batch_size, num_dims, num_points = x.size()
    device = x.device
    
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Edge feature: concatenate [x_i, x_j - x_i]
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (B, 2*C, N, k)

class EdgeConv(nn.Module):
    """Edge Convolution layer"""
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        # x: (B, C, N)
        x = get_graph_feature(x, k=self.k)  # (B, 2*C, N, k)
        x = self.conv(x)  # (B, out_channels, N, k)
        x = x.max(dim=-1)[0]  # (B, out_channels, N) - aggregate over neighbors
        return x

class EdgeCNN(nn.Module):
    """Dynamic Graph CNN for point cloud classification"""
    def __init__(self, in_channels, hidden, num_classes, k=20):
        super().__init__()
        self.k = k
        
        # EdgeConv layers with increasing receptive field
        self.conv1 = EdgeConv(in_channels, hidden, k)
        self.conv2 = EdgeConv(hidden, hidden, k)
        self.conv3 = EdgeConv(hidden, hidden * 2, k)
        self.conv4 = EdgeConv(hidden * 2, hidden * 4, k)
        
        # Global feature aggregation + MLP classifier
        self.bn5 = nn.BatchNorm1d(hidden * 8)
        self.conv5 = nn.Conv1d(hidden * 8, hidden * 8, 1, bias=False)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 4),
            nn.BatchNorm1d(hidden * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden * 4, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden * 2, num_classes),
        )
    
    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1).contiguous()
        
        # EdgeConv blocks with skip connections
        x1 = self.conv1(x)   # (B, hidden, N)
        x2 = self.conv2(x1)  # (B, hidden, N)
        x3 = self.conv3(x2)  # (B, hidden*2, N)
        x4 = self.conv4(x3)  # (B, hidden*4, N)
        
        # Concatenate multi-scale features
        x = torch.cat((x1, x2, x3, x4), dim=1)  # (B, hidden*8, N)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        # Global max pooling
        x = x.max(dim=-1)[0]  # (B, hidden*8)
        
        return self.classifier(x)

class SyntheticPointCloudDataset(Dataset):
    def __init__(self, size, num_points, in_channels, num_classes):
        self.size = size
        self.num_points = num_points
        self.in_channels = in_channels
        self.num_classes = num_classes
    
    def __len__(self): return self.size
    
    def __getitem__(self, i):
        points = torch.randn(self.num_points, self.in_channels)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return points, label

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = EdgeCNN(IN_CHANNELS, HIDDEN, NUM_CLASSES, K).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gnn_edgecnn","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"edgecnn_16_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticPointCloudDataset(NUM_SAMPLES, NUM_POINTS, IN_CHANNELS, NUM_CLASSES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for points, labels in loader:
            points, labels = points.to(dev), labels.to(dev)
            optim.zero_grad()
            loss = crit(model(points), labels)
            loss.backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
