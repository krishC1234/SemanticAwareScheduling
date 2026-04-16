#!/usr/bin/env python3
"""EdgeCNN (Dynamic Graph CNN) - batch=16, large params (~4M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
HIDDEN = 128
EPOCHS = 3
NUM_SAMPLES = 2000
NUM_POINTS = 1024
IN_CHANNELS = 3
NUM_CLASSES = 40
K = 20

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]

def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    device = x.device
    if idx is None: idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        return self.conv(x).max(dim=-1)[0]

class EdgeCNN(nn.Module):
    def __init__(self, in_channels, hidden, num_classes, k=20):
        super().__init__()
        self.conv1 = EdgeConv(in_channels, hidden, k)
        self.conv2 = EdgeConv(hidden, hidden, k)
        self.conv3 = EdgeConv(hidden, hidden * 2, k)
        self.conv4 = EdgeConv(hidden * 2, hidden * 4, k)
        self.bn5 = nn.BatchNorm1d(hidden * 8)
        self.conv5 = nn.Conv1d(hidden * 8, hidden * 8, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 4), nn.BatchNorm1d(hidden * 4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden * 4, hidden * 2), nn.BatchNorm1d(hidden * 2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5),
            nn.Linear(hidden * 2, num_classes),
        )
    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        x1, x2, x3, x4 = self.conv1(x), self.conv2(self.conv1(x)), self.conv3(self.conv2(self.conv1(x))), None
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        return self.classifier(x.max(dim=-1)[0])

class SyntheticPointCloudDataset(Dataset):
    def __init__(self, size, num_points, in_channels, num_classes):
        self.size, self.num_points, self.in_channels, self.num_classes = size, num_points, in_channels, num_classes
    def __len__(self): return self.size
    def __getitem__(self, i):
        return torch.randn(self.num_points, self.in_channels), torch.randint(0, self.num_classes, (1,)).item()

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
        print(f"edgecnn_16_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
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
            optim.zero_grad(); crit(model(points), labels).backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
