#!/usr/bin/env python3
"""MAML (Higher variant) - few-shot classification, batch=1, ~0.03M params

MAML variant using a 3-layer CNN with MaxPool (vs 4-layer stride-2 in
maml_omniglot). Smaller architecture, 5-way 5-shot with 5 inner SGD
steps. Uses manual second-order differentiation through the inner loop.

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
BATCH_SIZE = 1

# === FIXED ===
EPOCHS = 15
NUM_SAMPLES = 5000
N_WAY = 5
K_SHOT = 5              # 5-shot (vs 1-shot in maml_omniglot)
K_QUERY = 15
TASK_NUM = 5             # meta-batch size (smaller than maml_omniglot's 32)
IMG_SIZE = 28
IMG_C = 1
CONV_DIM = 64
INNER_LR = 0.1           # SGD inner lr
INNER_STEPS = 5
META_LR = 1e-3

# ---------------------------------------------------------------------------
# 3-layer CNN with MaxPool (matches the higher/functorch variant)
# ---------------------------------------------------------------------------
class MAMLNet(nn.Module):
    """3-layer CNN + MaxPool for few-shot classification. ~0.03M parameters.

    Architecture matches the higher library example:
    Conv(1→64, 3) → BN → ReLU → MaxPool(2)  ×3  → Flatten → Linear(64→5)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(IMG_C, CONV_DIM, 3),
            nn.BatchNorm2d(CONV_DIM, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(CONV_DIM, CONV_DIM, 3),
            nn.BatchNorm2d(CONV_DIM, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(CONV_DIM, CONV_DIM, 3),
            nn.BatchNorm2d(CONV_DIM, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(CONV_DIM, N_WAY)

    def forward(self, x):
        return self.classifier(self.features(x))

    def functional_forward(self, x, params):
        """Forward pass with external params for inner-loop differentiation."""
        keys = list(params.keys())
        idx = 0
        for i in range(3):  # 3 conv blocks
            # Conv2d
            x = F.conv2d(x, params[keys[idx]], params[keys[idx + 1]])
            idx += 2
            # BatchNorm
            x = F.batch_norm(x, None, None, params[keys[idx]], params[keys[idx + 1]], training=True)
            idx += 2
            x = F.relu(x, inplace=False)
            x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        # Linear
        x = F.linear(x, params[keys[idx]], params[keys[idx + 1]])
        return x


class SyntheticMetaDataset(Dataset):
    """Synthetic few-shot tasks: 5-way 5-shot with 15 query per task."""
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

    model = MAMLNet().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "other", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"maml | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

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
            spt_x = spt_x.squeeze(0).to(dev)
            spt_y = spt_y.squeeze(0).to(dev)
            qry_x = qry_x.squeeze(0).to(dev)
            qry_y = qry_y.squeeze(0).to(dev)

            meta_optim.zero_grad()
            meta_loss = torch.tensor(0.0, device=dev)

            for t in range(TASK_NUM):
                # Inner loop: SGD adaptation on support set
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

                # Outer loop: query loss with adapted params
                qry_logits = model.module.functional_forward(qry_x[t], params)
                meta_loss = meta_loss + F.cross_entropy(qry_logits, qry_y[t])

            meta_loss = meta_loss / TASK_NUM
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