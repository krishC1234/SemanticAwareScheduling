#!/usr/bin/env python3
"""MoCo (Momentum Contrast) - batch=16, ~50M params (ResNet-34)"""
import time,json,copy,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
MOCO_DIM = 128
MOCO_K = 4096
MOCO_M = 0.999
MOCO_T = 0.07
EPOCHS = 3
NUM_SAMPLES = 800
LR = 0.03
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: identity = self.downsample(x)
        return F.relu(out + identity)

class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_ch:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))
        layers = [BasicBlock(self.in_channels, out_ch, stride, downsample)]
        self.in_channels = out_ch
        for _ in range(1, blocks): layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.avgpool(x).flatten(1))

class MoCo(nn.Module):
    def __init__(self, dim=MOCO_DIM, K=MOCO_K, m=MOCO_M, T=MOCO_T):
        super().__init__()
        self.K, self.m, self.T = K, m, T
        self.encoder_q = ResNet34(num_classes=dim)
        self.encoder_k = ResNet34(num_classes=dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data); param_k.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]; ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K: self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            overflow = (ptr + batch_size) - self.K
            self.queue[:, ptr:] = keys[:self.K - ptr].T; self.queue[:, :overflow] = keys[self.K - ptr:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.K
    def forward(self, im_q, im_k):
        q = F.normalize(self.encoder_q(im_q), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(im_k), dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

class MoCoDataset(Dataset):
    def __init__(self, sz): self.sz = sz
    def __len__(self): return self.sz
    def __getitem__(self, i): return torch.rand(3, 224, 224), torch.rand(3, 224, 224)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    model = MoCo().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"moco_contrastive","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"moco_16_50M | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    ds = MoCoDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for im_q, im_k in loader:
            im_q, im_k = im_q.to(dev), im_k.to(dev)
            output, target = model(im_q, im_k)
            loss = criterion(output, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
