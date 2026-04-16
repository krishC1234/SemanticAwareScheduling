#!/usr/bin/env python3
"""Whisper - batch=4, small params (~20M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 4
D_MODEL = 256
N_HEADS = 4
N_ENCODER_LAYERS = 4
N_DECODER_LAYERS = 4

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 500
N_MELS = 80           # Mel spectrogram features
AUDIO_SEQ_LEN = 3000  # ~30 seconds at 100 frames/sec
MAX_TEXT_LEN = 448    # Max decoder sequence length
VOCAB_SIZE = 51865    # Whisper vocabulary size
D_FF = None           # Will be 4 * D_MODEL
DROPOUT = 0.1
LR = 1e-4

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AudioEncoder(nn.Module):
    """Whisper-style audio encoder with conv layers + transformer"""
    def __init__(self, n_mels, d_model, n_heads, n_layers, d_ff, dropout):
        super().__init__()
        # Two conv layers to downsample audio (like Whisper)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=AUDIO_SEQ_LEN)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, n_mels, T)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (B, T//2, d_model)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x

class TextDecoder(nn.Module):
    """Whisper-style text decoder with cross-attention"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.output_proj.weight = self.token_embedding.weight

    def forward(self, tokens, encoder_output, tgt_mask=None):
        # tokens: (B, T_text)
        x = self.token_embedding(tokens)
        x = self.pos_encoding(x)
        x = self.transformer(x, encoder_output, tgt_mask=tgt_mask)
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        return logits

class Whisper(nn.Module):
    """Whisper: Speech Recognition Transformer"""
    def __init__(self, n_mels=N_MELS, vocab_size=VOCAB_SIZE, d_model=D_MODEL, 
                 n_heads=N_HEADS, n_enc_layers=N_ENCODER_LAYERS, 
                 n_dec_layers=N_DECODER_LAYERS, dropout=DROPOUT, max_text_len=MAX_TEXT_LEN):
        super().__init__()
        d_ff = 4 * d_model
        self.encoder = AudioEncoder(n_mels, d_model, n_heads, n_enc_layers, d_ff, dropout)
        self.decoder = TextDecoder(vocab_size, d_model, n_heads, n_dec_layers, d_ff, dropout, max_text_len)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, audio_features, tokens):
        # audio_features: (B, n_mels, T_audio)
        # tokens: (B, T_text)
        encoder_output = self.encoder(audio_features)
        tgt_mask = self.generate_square_subsequent_mask(tokens.size(1), tokens.device)
        logits = self.decoder(tokens, encoder_output, tgt_mask=tgt_mask)
        return logits

class SyntheticASRDataset(Dataset):
    """Generate synthetic audio-text pairs for ASR training"""
    def __init__(self, size, n_mels, audio_len, max_text_len, vocab_size):
        self.size = size
        self.n_mels = n_mels
        self.audio_len = audio_len
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size

    def __len__(self): return self.size

    def __getitem__(self, i):
        # Random mel spectrogram
        audio = torch.randn(self.n_mels, self.audio_len)
        # Random text tokens (variable length)
        text_len = torch.randint(10, self.max_text_len // 2, (1,)).item()
        tokens = torch.randint(1, self.vocab_size, (text_len,))  # 0 reserved for padding
        return audio, tokens

def collate_asr(batch):
    audios, tokens = zip(*batch)
    audios = torch.stack(audios, dim=0)
    # Pad tokens
    max_len = max(t.size(0) for t in tokens)
    tokens_padded = torch.zeros(len(tokens), max_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        tokens_padded[i, :t.size(0)] = t
    return audios, tokens_padded

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = Whisper().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_whisper","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"whisper_4_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticASRDataset(NUM_SAMPLES, N_MELS, AUDIO_SEQ_LEN, MAX_TEXT_LEN, VOCAB_SIZE)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_asr)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for audio, tokens in loader:
            audio, tokens = audio.to(dev), tokens.to(dev)
            # Teacher forcing: input is tokens[:-1], target is tokens[1:]
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]
            
            optimizer.zero_grad()
            logits = model(audio, input_tokens)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_tokens.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
