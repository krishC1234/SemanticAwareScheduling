#!/usr/bin/env python3
"""Demucs - music source separation, batch=64, ~64M params

Demucs is a waveform-domain model for music source separation (vocals,
drums, bass, other). It uses a U-Net-like encoder-decoder with 1D
convolutions, LSTM bottleneck, and GLU activations. Operates directly
on raw audio waveforms.

Reference: Défossez et al., "Music Source Separation in the Waveform
Domain", 2019
"""
import time, json, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 64

# === FIXED ===
EPOCHS = 20
NUM_SAMPLES = 2000
AUDIO_CHANNELS = 2       # stereo
SOURCES = 4              # vocals, drums, bass, other
SAMPLE_LENGTH = 80000    # ~5 seconds at 16kHz
CHANNELS = 64            # base channel width
DEPTH = 5                # encoder/decoder depth
LSTM_LAYERS = 2
KERNEL_SIZE = 8
STRIDE = 4

# ---------------------------------------------------------------------------
# Demucs components
# ---------------------------------------------------------------------------
class GLU(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch * 2, KERNEL_SIZE, stride=STRIDE, padding=KERNEL_SIZE // 2)
        self.glu = GLU(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.glu(self.conv(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_ch, out_ch * 2, KERNEL_SIZE, stride=STRIDE, padding=KERNEL_SIZE // 2)
        self.glu = GLU(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.glu(self.convtr(x)))


class DemucsModel(nn.Module):
    """Demucs U-Net for source separation. ~64M trainable parameters.

    Encoder: 5 strided conv blocks (stride=4), doubles channels each level.
    Bottleneck: 2-layer bidirectional LSTM.
    Decoder: 5 transposed conv blocks with skip connections.
    Output: SOURCES * AUDIO_CHANNELS channels, reshaped to (B, sources, channels, time).
    """

    def __init__(self):
        super().__init__()
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = AUDIO_CHANNELS
        ch = CHANNELS
        for i in range(DEPTH):
            self.encoders.append(EncoderBlock(in_ch, ch))
            in_ch = ch
            ch = min(ch * 2, 512)

        # LSTM bottleneck
        self.lstm = nn.LSTM(in_ch, in_ch, num_layers=LSTM_LAYERS,
                            bidirectional=True, batch_first=True)
        self.lstm_linear = nn.Linear(in_ch * 2, in_ch)

        # Decoder (reverse order, with skip connections)
        self.decoders = nn.ModuleList()
        for i in range(DEPTH):
            out_ch = CHANNELS * min(2 ** (DEPTH - 2 - i), 8) if i < DEPTH - 1 else SOURCES * AUDIO_CHANNELS
            # skip connection doubles input channels
            self.decoders.append(DecoderBlock(in_ch * 2 if i > 0 else in_ch, out_ch))
            in_ch = out_ch

    def forward(self, mix):
        """mix: (B, audio_channels, time) → (B, sources, audio_channels, time)"""
        B = mix.shape[0]
        x = mix
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # LSTM
        x = x.transpose(1, 2)  # (B, T, C)
        x, _ = self.lstm(x)
        x = self.lstm_linear(x)
        x = x.transpose(1, 2)  # (B, C, T)

        for i, dec in enumerate(self.decoders):
            if i > 0:
                skip = skips[-(i + 1)]
                # Trim to match lengths
                min_len = min(x.shape[-1], skip.shape[-1])
                x = torch.cat([x[..., :min_len], skip[..., :min_len]], dim=1)
            x = dec(x)

        # Trim output to match input length
        min_len = min(x.shape[-1], mix.shape[-1])
        x = x[..., :min_len]
        return x.view(B, SOURCES, AUDIO_CHANNELS, -1)


class SyntheticAudioDataset(Dataset):
    """Synthetic multi-source audio: (B, sources+1, channels, time).
    Channel 0 = mix, channels 1..4 = individual sources."""
    def __init__(self, size):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, _):
        # Generate sources, mix is sum
        sources = torch.randn(SOURCES, AUDIO_CHANNELS, SAMPLE_LENGTH)
        mix = sources.sum(dim=0)
        return mix, sources


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = DemucsModel().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "audio", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"demucs | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticAudioDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.L1Loss()

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for mix, sources in loader:
            mix, sources = mix.to(dev), sources.to(dev)
            optim.zero_grad()
            estimates = model(mix)
            # Trim sources to match estimate length
            min_len = min(estimates.shape[-1], sources.shape[-1])
            loss = crit(estimates[..., :min_len], sources[..., :min_len])
            loss.backward()
            optim.step()
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