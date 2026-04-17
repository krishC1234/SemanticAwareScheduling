#!/usr/bin/env python3
"""MAML Omniglot (Meta-Learning) - task_num=8, ~25K params"""
import time,json,functools,torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from torch.func import functional_call
import warnings
warnings.filterwarnings("ignore")

TASK_NUM = 8
N_WAY = 5
K_SHOT = 1
Q_QUERY = 15
N_INNER_ITER = 5
HIDDEN_CH = 32
IMG_SIZE = 28
EPOCHS = 3
NUM_META_BATCHES = 200
LR_META = 1e-3
LR_INNER = 0.1

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch, affine=True, track_running_stats=False)
    def forward(self, x):
        return F.max_pool2d(F.relu(self.bn(self.conv(x))), 2)

class MAMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, HIDDEN_CH)
        self.conv2 = ConvBlock(HIDDEN_CH, HIDDEN_CH)
        self.conv3 = ConvBlock(HIDDEN_CH, HIDDEN_CH)
        self.conv4 = ConvBlock(HIDDEN_CH, HIDDEN_CH)
        self.fc = nn.Linear(HIDDEN_CH, N_WAY)
    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        return self.fc(x.view(x.size(0), -1))

def compute_loss(params, buffers, model, x, y):
    logits = functional_call(model, (params, buffers), (x,))
    return F.cross_entropy(logits, y)

def inner_loop(model, params, buffers, x_spt, y_spt, n_inner_iter, lr_inner):
    new_params = {k: v.clone() for k, v in params.items()}
    for _ in range(n_inner_iter):
        loss = compute_loss(new_params, buffers, model, x_spt, y_spt)
        grads = torch.autograd.grad(loss, new_params.values(), create_graph=True)
        new_params = {k: p - lr_inner * g for (k, p), g in zip(new_params.items(), grads)}
    return new_params

def meta_loss_for_task(model, params, buffers, x_spt, y_spt, x_qry, y_qry):
    adapted_params = inner_loop(model, params, buffers, x_spt, y_spt, N_INNER_ITER, LR_INNER)
    qry_logits = functional_call(model, (adapted_params, buffers), (x_qry,))
    qry_loss = F.cross_entropy(qry_logits, y_qry)
    qry_acc = (qry_logits.argmax(dim=1) == y_qry).float().mean()
    return qry_loss, qry_acc

class OmniglotTaskDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i):
        spt_size = N_WAY * K_SHOT
        x_spt = torch.rand(spt_size, 1, IMG_SIZE, IMG_SIZE)
        y_spt = torch.arange(N_WAY).repeat_interleave(K_SHOT)
        qry_size = N_WAY * Q_QUERY
        x_qry = torch.rand(qry_size, 1, IMG_SIZE, IMG_SIZE)
        y_qry = torch.arange(N_WAY).repeat_interleave(Q_QUERY)
        perm_spt, perm_qry = torch.randperm(spt_size), torch.randperm(qry_size)
        return x_spt[perm_spt], y_spt[perm_spt], x_qry[perm_qry], y_qry[perm_qry]

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = MAMLNet().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"maml_omniglot","batch_size":TASK_NUM,"param_count":pc})); print("###END_FEATURES###")
        print(f"maml_omniglot_8_25K | GPUs:{ws} | Tasks:{TASK_NUM} | Params:{pc:,}")
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    meta_opt = optim.Adam(model.parameters(), lr=LR_META)
    ds = OmniglotTaskDataset(NUM_META_BATCHES * TASK_NUM)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=TASK_NUM, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for x_spt, y_spt, x_qry, y_qry in loader:
            x_spt, y_spt = x_spt.to(dev), y_spt.to(dev)
            x_qry, y_qry = x_qry.to(dev), y_qry.to(dev)
            meta_opt.zero_grad()
            total_loss = 0
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            for t in range(TASK_NUM):
                loss, acc = meta_loss_for_task(model, params, buffers, x_spt[t], y_spt[t], x_qry[t], y_qry[t])
                total_loss = total_loss + loss
            total_loss = total_loss / TASK_NUM
            total_loss.backward()
            meta_opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} tasks/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} tasks/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":TASK_NUM,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
