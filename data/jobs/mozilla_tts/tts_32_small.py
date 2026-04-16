#!/usr/bin/env python3
"""Mozilla TTS Speaker Encoder - batch=32, small params (~2M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32
HIDDEN_DIM = 256
NUM_LAYERS = 2
EMBEDDING_DIM = 128

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
N_MELS = 80               # Mel spectrogram features
UTTERANCE_LEN = 160       # Frames per utterance (~1.6 sec at 100 Hz)
NUM_SPEAKERS = 100        # Number of speakers in training
UTTERANCES_PER_SPEAKER = 10  # Utterances per speaker in batch
LR = 1e-4

class SpeakerEncoder(nn.Module):
    """
    Speaker Encoder from Mozilla TTS / GE2E
    Maps variable-length mel spectrograms to fixed-size speaker embeddings
    
    Architecture:
    - Stack of LSTM layers
    - Linear projection to embedding space
    - L2 normalization
    """
    def __init__(self, n_mels=N_MELS, hidden_dim=HIDDEN_DIM, 
                 num_layers=NUM_LAYERS, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1
        )
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, mels):
        """
        mels: (B, n_mels, T) mel spectrogram
        returns: (B, embedding_dim) L2-normalized speaker embedding
        """
        # (B, n_mels, T) -> (B, T, n_mels)
        x = mels.transpose(1, 2)
        
        # LSTM
        self.lstm.flatten_parameters()
        outputs, (hidden, _) = self.lstm(x)
        
        # Take last hidden state of last layer
        embedding = self.projection(hidden[-1])
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End Loss for Speaker Verification
    
    Computes angular similarity between speaker embeddings and learns
    to maximize intra-speaker similarity while minimizing inter-speaker similarity.
    
    Reference: "Generalized End-to-End Loss for Speaker Verification" (Google, 2018)
    """
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings, num_speakers, utterances_per_speaker):
        """
        embeddings: (N, embedding_dim) where N = num_speakers * utterances_per_speaker
        """
        # Reshape to (num_speakers, utterances_per_speaker, embedding_dim)
        embeddings = embeddings.view(num_speakers, utterances_per_speaker, -1)
        
        # Compute centroids for each speaker (excluding the utterance being compared)
        centroids_incl = embeddings.mean(dim=1)  # (num_speakers, embedding_dim)
        
        # Compute similarity matrix
        # For each utterance, compute similarity to all speaker centroids
        total_loss = 0.0
        
        for spk_idx in range(num_speakers):
            for utt_idx in range(utterances_per_speaker):
                # Compute centroid excluding current utterance
                mask = torch.ones(utterances_per_speaker, device=embeddings.device)
                mask[utt_idx] = 0
                centroid_excl = (embeddings[spk_idx] * mask.unsqueeze(1)).sum(dim=0) / (utterances_per_speaker - 1)
                
                # Current utterance embedding
                utt_embedding = embeddings[spk_idx, utt_idx]
                
                # Compute similarities to all centroids
                # Use centroid_excl for same speaker, centroids_incl for others
                similarities = []
                for other_spk in range(num_speakers):
                    if other_spk == spk_idx:
                        centroid = centroid_excl
                    else:
                        centroid = centroids_incl[other_spk]
                    
                    # Cosine similarity scaled by learned parameters
                    sim = self.w * F.cosine_similarity(utt_embedding.unsqueeze(0), centroid.unsqueeze(0)) + self.b
                    similarities.append(sim)
                
                similarities = torch.cat(similarities)
                
                # Softmax loss: maximize similarity to own speaker, minimize to others
                target = torch.tensor(spk_idx, device=embeddings.device)
                loss = F.cross_entropy(similarities.unsqueeze(0), target.unsqueeze(0))
                total_loss += loss
        
        return total_loss / (num_speakers * utterances_per_speaker)

class SimplifiedGE2ELoss(nn.Module):
    """Simplified GE2E loss for faster training"""
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embeddings, num_speakers, utterances_per_speaker):
        # Reshape: (num_speakers, utterances_per_speaker, embedding_dim)
        embeddings = embeddings.view(num_speakers, utterances_per_speaker, -1)
        
        # Compute centroids
        centroids = embeddings.mean(dim=1)  # (num_speakers, embedding_dim)
        
        # Flatten embeddings for similarity computation
        flat_emb = embeddings.view(-1, embeddings.size(-1))  # (N, D)
        
        # Compute all pairwise similarities: (N, num_speakers)
        sim_matrix = self.w * torch.mm(flat_emb, centroids.t()) + self.b
        
        # Create target labels
        targets = torch.arange(num_speakers, device=embeddings.device)
        targets = targets.unsqueeze(1).expand(-1, utterances_per_speaker).reshape(-1)
        
        return F.cross_entropy(sim_matrix, targets)

class SpeakerDataset(Dataset):
    """Generate synthetic speaker verification data"""
    def __init__(self, size, num_speakers, utterances_per_speaker, n_mels, utterance_len):
        self.size = size
        self.num_speakers = num_speakers
        self.utterances_per_speaker = utterances_per_speaker
        self.n_mels = n_mels
        self.utterance_len = utterance_len
        
        # Pre-generate speaker "characteristics" for consistency
        self.speaker_means = torch.randn(num_speakers, n_mels) * 0.5
        self.speaker_stds = torch.rand(num_speakers, n_mels) * 0.3 + 0.2

    def __len__(self): return self.size

    def __getitem__(self, i):
        # Generate a batch of utterances for multiple speakers
        # This simulates the GE2E training setup
        speaker_id = i % self.num_speakers
        
        # Generate mel spectrogram with speaker-specific characteristics
        mel = torch.randn(self.n_mels, self.utterance_len)
        mel = mel * self.speaker_stds[speaker_id].unsqueeze(1) + self.speaker_means[speaker_id].unsqueeze(1)
        
        return mel, speaker_id

def collate_speaker(batch):
    mels, speaker_ids = zip(*batch)
    return torch.stack(mels), torch.tensor(speaker_ids)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = SpeakerEncoder().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_tts_speaker_encoder","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"tts_32_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SpeakerDataset(NUM_SAMPLES, NUM_SPEAKERS, UTTERANCES_PER_SPEAKER, N_MELS, UTTERANCE_LEN)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_speaker)

    criterion = SimplifiedGE2ELoss().to(dev)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=LR)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for mels, speaker_ids in loader:
            mels = mels.to(dev)
            
            optimizer.zero_grad()
            embeddings = model(mels)
            
            # For simplified loss, we need to reorganize by speaker
            # Sort by speaker_id and compute loss
            num_spk = min(BATCH_SIZE // 2, 10)  # Speakers per batch
            utt_per_spk = BATCH_SIZE // num_spk
            
            loss = criterion(embeddings[:num_spk * utt_per_spk], num_spk, utt_per_spk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
