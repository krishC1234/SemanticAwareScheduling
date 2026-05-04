#!/usr/bin/env python3
"""MAML (Model-Agnostic Meta-Learning) - few-shot classification, batch=1, ~0.11M params

MAML learns an initialization for a small CNN that can be quickly
adapted to new tasks with a few gradient steps. The outer loop
optimizes across tasks; the inner loop fine-tunes on each task's
support set.

Architecture: 4 conv blocks (64 filters each, 3×3, stride 2) → linear.
Trained on 5-way 1-shot tasks with 28×28 grayscale images (Omniglot-style).

Reference: Finn et al., "Model-Agnostic Meta-Learning for Fast
Adaptation of Deep Networks", ICML 2017
"""
import time, json, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1           # meta-batch = 1 (but each sample = 32 tasks)

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 400        # number of meta-batches
N_WAY = 5
K_SHOT = 1
K_QUERY = 15
TASK_NUM = 32            # tasks per meta-batch
IMG_SIZE = 28
IMG_C = 1                # grayscale
CONV_DIM = 64
INNER_LR = 0.4
INNER_STEPS = 5
META_LR = 1e-3

# ---------------------------------------------------------------------------
# MAML components
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class MAMLModel(nn.Module):
    """4-layer CNN for few-shot classification. ~0.11M parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(IMG_C, CONV_DIM)
        self.conv2 = ConvBlock(CONV_DIM, CONV_DIM)
        self.conv3 = ConvBlock(CONV_DIM, CONV_DIM)
        self.conv4 = ConvBlock(CONV_DIM, CONV_DIM)
        self.classifier = nn.Linear(CONV_DIM * 2 * 2, N_WAY)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def functional_forward(self, x, params):
        """Forward pass using external params (for inner-loop adaptation)."""
        # Conv1
        x = F.conv2d(x, params['conv1.conv.weight'], params['conv1.conv.bias'], stride=2, padding=1)
        x = F.batch_norm(x, None, None, params['conv1.bn.weight'], params['conv1.bn.bias'], training=True)
        x = F.relu(x)
        # Conv2
        x = F.conv2d(x, params['conv2.conv.weight'], params['conv2.conv.bias'], stride=2, padding=1)
        x = F.batch_norm(x, None, None, params['conv2.bn.weight'], params['conv2.bn.bias'], training=True)
        x = F.relu(x)
        # Conv3
        x = F.conv2d(x, params['conv3.conv.weight'], params['conv3.conv.bias'], stride=2, padding=1)
        x = F.batch_norm(x, None, None, params['conv3.bn.weight'], params['conv3.bn.bias'], training=True)
        x = F.relu(x)
        # Conv4
        x = F.conv2d(x, params['conv4.conv.weight'], params['conv4.conv.bias'], stride=2, padding=1)
        x = F.batch_norm(x, None, None, params['conv4.bn.weight'], params['conv4.bn.bias'], training=True)
        x = F.relu(x)
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['classifier.weight'], params['classifier.bias'])
        return x


class SyntheticMetaDataset(Dataset):
    """Generates synthetic few-shot tasks (Omniglot-style).
    Each item = (support_x, support_y, query_x, query_y) for TASK_NUM tasks."""

    def __init__(self, size):
        self.size = size

    def __len__(self): return self.size

    def __getitem__(self, _):
        spt_x = torch.randn(TASK_NUM, N_WAY * K_SHOT, IMG_C, IMG_SIZE, IMG_SIZE)
        spt_y = torch.arange(N_WAY).repeat(K_SHOT).unsqueeze(0).expand(TASK_NUM, -1)
        qry_x = torch.randn(TASK_NUM, N_WAY * K_QUERY, IMG_C, IMG_SIZE, IMG_SIZE)
        qry_y = torch.arange(N_WAY).repeat(K_QUERY).unsqueeze(0).expand(TASK_NUM, -1)
        return spt_x, spt_y, qry_x, qry_y


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = MAMLModel().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "other", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"maml_omniglot | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticMetaDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=0, pin_memory=True, drop_last=True)
    meta_optim = torch.optim.Adam(model.parameters(), lr=META_LR)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for spt_x, spt_y, qry_x, qry_y in loader:
            spt_x = spt_x.squeeze(0).to(dev)  # (TASK_NUM, N*K, C, H, W)
            spt_y = spt_y.squeeze(0).to(dev)
            qry_x = qry_x.squeeze(0).to(dev)
            qry_y = qry_y.squeeze(0).to(dev)

            meta_loss = torch.tensor(0.0, device=dev)
            for t in range(TASK_NUM):
                # Inner loop: adapt on support set
                params = OrderedDict(
                    (n, p.clone()) for n, p in model.module.named_parameters()
                )
                for _ in range(INNER_STEPS):
                    logits = model.module.functional_forward(spt_x[t], params)
                    loss = F.cross_entropy(logits, spt_y[t])
                    grads = torch.autograd.grad(loss, params.values(), create_graph=True)
                    params = OrderedDict(
                        (n, p - INNER_LR * g) for (n, p), g in zip(params.items(), grads)
                    )
                # Outer loop: evaluate on query set
                qry_logits = model.module.functional_forward(qry_x[t], params)
                meta_loss = meta_loss + F.cross_entropy(qry_logits, qry_y[t])

            meta_loss = meta_loss / TASK_NUM
            meta_optim.zero_grad()
            meta_loss.backward()
            meta_optim.step()

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