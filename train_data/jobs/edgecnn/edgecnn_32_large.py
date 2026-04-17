#!/usr/bin/env python3
"""EdgeCNN (Dynamic Graph CNN) - batch=32, large params (~4M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 32
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
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]

def get_graph_feature(x, k=20):
    batch_size, num_dims, num_points = x.size()
    device = x.device
    idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

class EdgeConv(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv2d(in_ch * 2, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x): return self.conv(get_graph_feature(x, k=self.k)).max(dim=-1)[0]

class EdgeCNN(nn.Module):
    def __init__(self, in_ch, hid, n_cls, k):
        super().__init__()
        self.conv1, self.conv2, self.conv3, self.conv4 = EdgeConv(in_ch, hid, k), EdgeConv(hid, hid, k), EdgeConv(hid, hid*2, k), EdgeConv(hid*2, hid*4, k)
        self.bn5, self.conv5 = nn.BatchNorm1d(hid*8), nn.Conv1d(hid*8, hid*8, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hid*8, hid*4), nn.BatchNorm1d(hid*4), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5),
            nn.Linear(hid*4, hid*2), nn.BatchNorm1d(hid*2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.5), nn.Linear(hid*2, n_cls))
    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        x1 = self.conv1(x); x2 = self.conv2(x1); x3 = self.conv3(x2); x4 = self.conv4(x3)
        x = F.leaky_relu(self.bn5(self.conv5(torch.cat((x1,x2,x3,x4), dim=1))), 0.2)
        return self.classifier(x.max(dim=-1)[0])

class SyntheticPointCloudDataset(Dataset):
    def __init__(self, sz, np, ic, nc): self.sz, self.np, self.ic, self.nc = sz, np, ic, nc
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.randn(self.np, self.ic), torch.randint(0, self.nc, (1,)).item()

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = EdgeCNN(IN_CHANNELS, HIDDEN, NUM_CLASSES, K).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"gnn_edgecnn","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"edgecnn_32_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticPointCloudDataset(NUM_SAMPLES, NUM_POINTS, IN_CHANNELS, NUM_CLASSES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optim, crit = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4), nn.CrossEntropyLoss()
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for pts, lbl in loader:
            pts, lbl = pts.to(dev), lbl.to(dev)
            optim.zero_grad(); crit(model(pts), lbl).backward(); optim.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
