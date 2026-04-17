#!/usr/bin/env python3
"""DLRM (Deep Learning Recommendation Model) - batch=2048, large params (~20M)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 2048
SPARSE_FEATURE_SIZE = 64
MLP_BOT = [512, 256, 64]
MLP_TOP = [512, 256, 1]
EPOCHS = 3
NUM_SAMPLES = 20000
NUM_DENSE_FEATURES = 13
NUM_SPARSE_FEATURES = 8
EMBEDDING_TABLE_SIZES = [50000, 50000, 50000, 50000, 50000, 50000, 50000, 50000]
NUM_INDICES_PER_LOOKUP = 20
LR = 0.01

class DLRM(nn.Module):
    def __init__(self, emb_sizes, sparse_dim, mlp_bot, mlp_top, num_dense):
        super().__init__()
        self.num_sparse = len(emb_sizes)
        self.embeddings = nn.ModuleList([nn.EmbeddingBag(sz, sparse_dim, mode='sum', sparse=False) for sz in emb_sizes])
        bot = []
        in_d = num_dense
        for out_d in mlp_bot:
            bot += [nn.Linear(in_d, out_d), nn.ReLU(True)]; in_d = out_d
        self.bot_mlp = nn.Sequential(*bot)
        num_fea = self.num_sparse + 1
        interact_dim = (num_fea * (num_fea - 1)) // 2 + mlp_bot[-1]
        top = []
        in_d = interact_dim
        for i, out_d in enumerate(mlp_top):
            top.append(nn.Linear(in_d, out_d))
            if i < len(mlp_top) - 1: top.append(nn.ReLU(True))
            in_d = out_d
        self.top_mlp = nn.Sequential(*top)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, nn.EmbeddingBag): nn.init.uniform_(m.weight, -0.01, 0.01)

    def forward(self, dense_x, sparse_off, sparse_idx):
        x = self.bot_mlp(dense_x)
        ly = [emb(sparse_idx[i], sparse_off[i]) for i, emb in enumerate(self.embeddings)]
        T = torch.cat([x.unsqueeze(1)] + [y.unsqueeze(1) for y in ly], dim=1)
        Z = torch.bmm(T, T.transpose(1, 2))
        li, lj = torch.triu_indices(T.size(1), T.size(1), offset=1, device=Z.device)
        z = torch.cat([x, Z[:, li, lj]], dim=1)
        return torch.sigmoid(self.top_mlp(z).squeeze(1))

class SyntheticDLRMDataset(Dataset):
    def __init__(self, sz, nd, ns, esz, npl): self.sz, self.nd, self.ns, self.esz, self.npl = sz, nd, ns, esz, npl
    def __len__(self): return self.sz
    def __getitem__(self, i):
        return torch.randn(self.nd), [torch.randint(0, self.esz[j], (self.npl,)) for j in range(self.ns)], float(torch.randint(0, 2, (1,)).item())

def collate_dlrm(batch):
    dense, sparse, labels = zip(*batch)
    bs, ns = len(batch), len(sparse[0])
    sparse_off = [torch.tensor([0] + [sparse[i][j].size(0) for i in range(bs-1)]).cumsum(0) for j in range(ns)]
    sparse_idx = [torch.cat([sparse[i][j] for i in range(bs)]) for j in range(ns)]
    return torch.stack(dense), sparse_off, sparse_idx, torch.tensor(labels)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = DLRM(EMBEDDING_TABLE_SIZES, SPARSE_FEATURE_SIZE, MLP_BOT, MLP_TOP, NUM_DENSE_FEATURES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"recommendation_dlrm","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"dlrm_2048_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    ds = SyntheticDLRMDataset(NUM_SAMPLES, NUM_DENSE_FEATURES, NUM_SPARSE_FEATURES, EMBEDDING_TABLE_SIZES, NUM_INDICES_PER_LOOKUP)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_dlrm)
    crit, opt = nn.BCELoss(), torch.optim.SGD(model.parameters(), lr=LR)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for dense, soff, sidx, labels in loader:
            dense, labels = dense.to(dev), labels.to(dev)
            soff, sidx = [o.to(dev) for o in soff], [i.to(dev) for i in sidx]
            opt.zero_grad(); crit(model(dense, soff, sidx), labels).backward(); opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
