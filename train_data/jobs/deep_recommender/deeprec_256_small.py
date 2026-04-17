#!/usr/bin/env python3
"""DeepRecommender (Autoencoder for Collaborative Filtering) - batch=256, small params (~5M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 256
HIDDEN_LAYERS = [256, 128, 256]  # Encoder-Decoder symmetric
EPOCHS = 3
NUM_SAMPLES = 10000
NUM_ITEMS = 10000  # Vocabulary size (items in catalog)
DROPOUT = 0.5
LR = 0.001
WEIGHT_DECAY = 0.0

class DeepAutoencoder(nn.Module):
    """Denoising Autoencoder for Collaborative Filtering
    
    Architecture: Input -> Encode -> Decode -> Reconstruct
    Uses dropout as noise injection for denoising autoencoder
    """
    def __init__(self, num_items, hidden_layers, dropout=0.5, activation='selu'):
        super().__init__()
        self.num_items = num_items
        self.dropout = nn.Dropout(dropout)
        
        # Build encoder-decoder
        layers = []
        layer_sizes = [num_items] + hidden_layers
        
        # Encoder layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if activation == 'selu':
                layers.append(nn.SELU())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(layer_sizes[-1], num_items))
        
        self.layers = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (B, num_items) - sparse user-item interaction vector
        Returns: (B, num_items) - reconstructed vector
        """
        x = self.dropout(x)
        return self.layers(x)
    
    def encode(self, x):
        """Get latent representation"""
        x = self.dropout(x)
        for layer in list(self.layers)[:-1]:
            x = layer(x)
        return x

def masked_mse_loss(pred, target, mask=None):
    """Masked MSE Loss - only compute loss on observed (non-zero) entries"""
    if mask is None:
        mask = (target != 0).float()
    
    diff = (pred - target) ** 2
    masked_diff = diff * mask
    
    # Average over observed entries
    num_observed = mask.sum() + 1e-8
    return masked_diff.sum() / num_observed

class RecommendationDataset(Dataset):
    """Synthetic sparse user-item interaction data"""
    def __init__(self, size, num_items, sparsity=0.99):
        self.size = size
        self.num_items = num_items
        self.sparsity = sparsity
    
    def __len__(self): return self.size
    
    def __getitem__(self, i):
        # Generate sparse interaction vector
        x = torch.zeros(self.num_items)
        # Random number of interactions (1-5% of items)
        num_interactions = int(self.num_items * (1 - self.sparsity) * (0.5 + torch.rand(1).item()))
        indices = torch.randperm(self.num_items)[:max(1, num_interactions)]
        # Random ratings (1-5 scale, normalized to 0-1)
        x[indices] = torch.rand(len(indices))
        return x

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    
    model = DeepAutoencoder(NUM_ITEMS, HIDDEN_LAYERS, DROPOUT).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"recommendation_autoencoder","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"deeprec_256_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = RecommendationDataset(NUM_SAMPLES, NUM_ITEMS)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for x in loader:
            x = x.to(dev)
            opt.zero_grad()
            pred = model(x)
            loss = masked_mse_loss(pred, x)
            loss.backward(); opt.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
