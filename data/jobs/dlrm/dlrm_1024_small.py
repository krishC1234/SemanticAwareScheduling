#!/usr/bin/env python3
"""DLRM (Deep Learning Recommendation Model) - batch=1024, small params (~5M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 1024
SPARSE_FEATURE_SIZE = 32  # Embedding dimension
MLP_BOT = [256, 128, 32]  # Bottom MLP layers
MLP_TOP = [256, 128, 1]   # Top MLP layers (adjusted by interaction output)

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 20000
NUM_DENSE_FEATURES = 13       # Number of dense (continuous) features
NUM_SPARSE_FEATURES = 8       # Number of sparse (categorical) features  
EMBEDDING_TABLE_SIZES = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]  # Reduced from 1M
NUM_INDICES_PER_LOOKUP = 20   # Pooling factor for each sparse feature
LR = 0.01

class DLRM(nn.Module):
    """Deep Learning Recommendation Model"""
    def __init__(self, embedding_sizes, sparse_feature_size, mlp_bot, mlp_top, 
                 num_dense_features, interaction_op="dot"):
        super().__init__()
        self.interaction_op = interaction_op
        self.num_sparse_features = len(embedding_sizes)
        
        # Embedding tables for sparse features
        self.embeddings = nn.ModuleList([
            nn.EmbeddingBag(size, sparse_feature_size, mode='sum', sparse=False)
            for size in embedding_sizes
        ])
        
        # Bottom MLP for dense features
        bot_layers = []
        in_dim = num_dense_features
        for out_dim in mlp_bot:
            bot_layers.append(nn.Linear(in_dim, out_dim))
            bot_layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.bot_mlp = nn.Sequential(*bot_layers)
        
        # Calculate interaction output dimension
        num_features = self.num_sparse_features + 1  # sparse features + dense
        if interaction_op == "dot":
            # Upper triangular dot product (excluding diagonal)
            interact_dim = (num_features * (num_features - 1)) // 2 + mlp_bot[-1]
        else:  # cat
            interact_dim = num_features * mlp_bot[-1]
        
        # Top MLP for final prediction
        top_layers = []
        in_dim = interact_dim
        for i, out_dim in enumerate(mlp_top):
            top_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(mlp_top) - 1:
                top_layers.append(nn.ReLU(inplace=True))
        self.top_mlp = nn.Sequential(*top_layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.EmbeddingBag):
                nn.init.uniform_(m.weight, -0.01, 0.01)
    
    def interact_features(self, x, ly):
        """Feature interaction layer"""
        # x: dense features after bottom MLP (B, D)
        # ly: list of sparse embeddings, each (B, D)
        
        if self.interaction_op == "dot":
            # Concatenate all features
            batch_size = x.size(0)
            feature_dim = x.size(1)
            
            # Stack: (B, num_features, D)
            T = torch.cat([x.unsqueeze(1)] + [y.unsqueeze(1) for y in ly], dim=1)
            
            # Dot product: (B, num_features, num_features)
            Z = torch.bmm(T, T.transpose(1, 2))
            
            # Extract upper triangular (excluding diagonal)
            _, ni, nj = Z.shape
            li, lj = torch.triu_indices(ni, nj, offset=1, device=Z.device)
            Zflat = Z[:, li, lj]
            
            # Concatenate with dense features
            R = torch.cat([x, Zflat], dim=1)
        else:  # cat
            R = torch.cat([x] + ly, dim=1)
        
        return R
    
    def forward(self, dense_x, sparse_offsets, sparse_indices):
        """
        dense_x: (B, num_dense_features) - continuous features
        sparse_offsets: list of (B,) tensors - offsets for each embedding bag
        sparse_indices: list of (B * pooling_factor,) tensors - indices for each embedding
        """
        # Bottom MLP on dense features
        x = self.bot_mlp(dense_x)
        
        # Embedding lookups for sparse features
        ly = []
        for i, emb in enumerate(self.embeddings):
            # EmbeddingBag: indices and offsets
            V = emb(sparse_indices[i], sparse_offsets[i])
            ly.append(V)
        
        # Feature interaction
        z = self.interact_features(x, ly)
        
        # Top MLP
        p = self.top_mlp(z)
        
        return torch.sigmoid(p.squeeze(1))

class SyntheticDLRMDataset(Dataset):
    """Generate synthetic recommendation data"""
    def __init__(self, size, num_dense, num_sparse, embedding_sizes, num_indices_per_lookup):
        self.size = size
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.embedding_sizes = embedding_sizes
        self.num_indices_per_lookup = num_indices_per_lookup
    
    def __len__(self): return self.size
    
    def __getitem__(self, i):
        # Dense features (continuous)
        dense = torch.randn(self.num_dense)
        
        # Sparse features (categorical indices for embedding lookup)
        sparse_indices = []
        for j in range(self.num_sparse):
            indices = torch.randint(0, self.embedding_sizes[j], (self.num_indices_per_lookup,))
            sparse_indices.append(indices)
        
        # Binary label (click/no-click)
        label = torch.randint(0, 2, (1,)).float().item()
        
        return dense, sparse_indices, label

def collate_dlrm(batch):
    """Custom collate function for DLRM batch"""
    dense_list, sparse_list, labels = zip(*batch)
    batch_size = len(batch)
    num_sparse = len(sparse_list[0])
    
    # Stack dense features
    dense = torch.stack(dense_list, dim=0)
    
    # Process sparse features for EmbeddingBag
    sparse_offsets = []
    sparse_indices = []
    
    for j in range(num_sparse):
        # Concatenate all indices for this sparse feature
        indices = torch.cat([sparse_list[i][j] for i in range(batch_size)])
        sparse_indices.append(indices)
        
        # Create offsets (cumulative sum of lengths)
        lengths = [sparse_list[i][j].size(0) for i in range(batch_size)]
        offsets = torch.tensor([0] + lengths[:-1]).cumsum(0)
        sparse_offsets.append(offsets)
    
    labels = torch.tensor(labels)
    
    return dense, sparse_offsets, sparse_indices, labels

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = DLRM(EMBEDDING_TABLE_SIZES, SPARSE_FEATURE_SIZE, MLP_BOT, MLP_TOP, 
                 NUM_DENSE_FEATURES).to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"recommendation_dlrm","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"dlrm_1024_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticDLRMDataset(NUM_SAMPLES, NUM_DENSE_FEATURES, NUM_SPARSE_FEATURES, 
                               EMBEDDING_TABLE_SIZES, NUM_INDICES_PER_LOOKUP)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, 
                        pin_memory=True, drop_last=True, collate_fn=collate_dlrm)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for dense, sparse_offsets, sparse_indices, labels in loader:
            dense = dense.to(dev)
            sparse_offsets = [o.to(dev) for o in sparse_offsets]
            sparse_indices = [i.to(dev) for i in sparse_indices]
            labels = labels.to(dev)
            
            optimizer.zero_grad()
            output = model(dense, sparse_offsets, sparse_indices)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
