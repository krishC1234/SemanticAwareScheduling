#!/usr/bin/env python3
"""DeepRecommender (Autoencoder for Collaborative Filtering) - batch=512, large params (~20M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 512
HIDDEN_LAYERS = [512, 256, 128, 256, 512]
EPOCHS = 3
NUM_SAMPLES = 10000
NUM_ITEMS = 20000
DROPOUT = 0.5
LR = 0.001
WEIGHT_DECAY = 0.0

class DeepAutoencoder(nn.Module):
    def __init__(self, num_items, hidden_layers, dropout=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers = []
        sizes = [num_items] + hidden_layers
        for i in range(len(sizes) - 1):
            layers.extend([nn.Linear(sizes[i], sizes[i+1]), nn.SELU()])
        layers.append(nn.Linear(sizes[-1], num_items))
        self.layers = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.layers(self.dropout(x))

def masked_mse_loss(pred, target):
    mask = (target != 0).float()
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)

class RecommendationDataset(Dataset):
    def __init__(self, size, num_items, sparsity=0.99):
        self.size, self.num_items, self.sparsity = size, num_items, sparsity
    def __len__(self): return self.size
    def __getitem__(self, i):
        x = torch.zeros(self.num_items)
        ni = int(self.num_items * (1 - self.sparsity) * (0.5 + torch.rand(1).item()))
        idx = torch.randperm(self.num_items)[:max(1, ni)]
        x[idx] = torch.rand(len(idx))
        return x

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = DeepAutoencoder(NUM_ITEMS, HIDDEN_LAYERS, DROPOUT).to(dev); pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"recommendation_autoencoder","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"deeprec_512_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = RecommendationDataset(NUM_SAMPLES, NUM_ITEMS)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for x in loader:
            x = x.to(dev); opt.zero_grad()
            loss = masked_mse_loss(model(x), x)
            loss.backward(); opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
