#!/usr/bin/env python3
"""Tacotron2 - batch=32, small params (~10M)"""
import time,json,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 32
ENCODER_DIM = 256
DECODER_RNN_DIM = 512
ATTENTION_RNN_DIM = 512
PRENET_DIM = 128
ATTENTION_DIM = 64
POSTNET_DIM = 256

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 1000
N_SYMBOLS = 148  # Text vocabulary size
SYMBOLS_EMBEDDING_DIM = 256
ENCODER_KERNEL_SIZE = 5
ENCODER_N_CONVS = 3
N_MEL_CHANNELS = 80
N_FRAMES_PER_STEP = 1
MAX_DECODER_STEPS = 200
ATTENTION_LOCATION_N_FILTERS = 32
ATTENTION_LOCATION_KERNEL_SIZE = 31
POSTNET_KERNEL_SIZE = 5
POSTNET_N_CONVS = 5
P_DROPOUT = 0.1
LR = 1e-3

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(w_init))
    def forward(self, x):
        return self.linear(x)

class ConvNorm(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init='linear'):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=bias)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init))
    def forward(self, x):
        return self.conv(x)

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([LinearNorm(in_s, out_s, bias=False) for in_s, out_s in zip(in_sizes, sizes)])
    def forward(self, x):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x)), p=0.5, training=True)  # Always dropout in prenet
        return x

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        self.location_conv = ConvNorm(2, attention_n_filters, attention_kernel_size, padding=(attention_kernel_size - 1) // 2, bias=False)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init='tanh')
    def forward(self, attention_weights_cat):
        processed = self.location_conv(attention_weights_cat)
        return self.location_dense(processed.transpose(1, 2))

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim, attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention + processed_memory))
        return energies.squeeze(-1)

    def forward(self, attention_hidden, memory, processed_memory, attention_weights_cat, mask):
        alignment = self.get_alignment_energies(attention_hidden, processed_memory, attention_weights_cat)
        if mask is not None:
            alignment.masked_fill_(mask, self.score_mask_value)
        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return attention_context, attention_weights

class Encoder(nn.Module):
    def __init__(self, n_symbols, symbols_embedding_dim, encoder_n_convs, encoder_dim, encoder_kernel_size):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        convs = []
        for _ in range(encoder_n_convs):
            convs.append(nn.Sequential(
                ConvNorm(symbols_embedding_dim, symbols_embedding_dim, encoder_kernel_size, padding=(encoder_kernel_size - 1) // 2, w_init='relu'),
                nn.BatchNorm1d(symbols_embedding_dim),
                nn.ReLU(),
                nn.Dropout(P_DROPOUT)
            ))
        self.convs = nn.ModuleList(convs)
        self.lstm = nn.LSTM(symbols_embedding_dim, encoder_dim // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x).transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)
        x = pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs

class Decoder(nn.Module):
    def __init__(self, n_mel, encoder_dim, attention_rnn_dim, decoder_rnn_dim, prenet_dim, attention_dim, 
                 attention_location_n_filters, attention_location_kernel_size, p_dropout):
        super().__init__()
        self.n_mel = n_mel
        self.encoder_dim = encoder_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.p_dropout = p_dropout

        self.prenet = Prenet(n_mel, [prenet_dim, prenet_dim])
        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_dim, attention_rnn_dim)
        self.attention = Attention(attention_rnn_dim, encoder_dim, attention_dim, attention_location_n_filters, attention_location_kernel_size)
        self.decoder_rnn = nn.LSTMCell(attention_rnn_dim + encoder_dim, decoder_rnn_dim)
        self.linear_projection = LinearNorm(decoder_rnn_dim + encoder_dim, n_mel)
        self.gate_layer = LinearNorm(decoder_rnn_dim + encoder_dim, 1, bias=True, w_init='sigmoid')

    def get_go_frame(self, memory):
        return memory.new_zeros(memory.size(0), self.n_mel)

    def initialize_decoder_states(self, memory):
        B, T, _ = memory.size()
        self.attention_hidden = memory.new_zeros(B, self.attention_rnn_dim)
        self.attention_cell = memory.new_zeros(B, self.attention_rnn_dim)
        self.decoder_hidden = memory.new_zeros(B, self.decoder_rnn_dim)
        self.decoder_cell = memory.new_zeros(B, self.decoder_rnn_dim)
        self.attention_weights = memory.new_zeros(B, T)
        self.attention_weights_cum = memory.new_zeros(B, T)
        self.attention_context = memory.new_zeros(B, self.encoder_dim)
        self.memory = memory
        self.processed_memory = self.attention.memory_layer(memory)

    def decode(self, decoder_input, mask):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_dropout, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention(
            self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, mask)
        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        mel_output = self.linear_projection(decoder_hidden_attention_context)
        gate_output = self.gate_layer(decoder_hidden_attention_context)
        return mel_output, gate_output, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = decoder_inputs.permute(2, 0, 1)  # (T, B, n_mel)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory)
        mask = ~self.get_mask_from_lengths(memory_lengths, memory.size(1), memory.device)

        mel_outputs, gate_outputs, alignments = [], [], []
        for i in range(decoder_inputs.size(0) - 1):
            mel_output, gate_output, attention_weights = self.decode(decoder_inputs[i], mask)
            mel_outputs.append(mel_output)
            gate_outputs.append(gate_output.squeeze(1))
            alignments.append(attention_weights)

        mel_outputs = torch.stack(mel_outputs, dim=2)
        gate_outputs = torch.stack(gate_outputs, dim=1)
        alignments = torch.stack(alignments, dim=1)
        return mel_outputs, gate_outputs, alignments

    def get_mask_from_lengths(self, lengths, max_len, device):
        ids = torch.arange(max_len, device=device)
        return ids < lengths.unsqueeze(1)

class Postnet(nn.Module):
    def __init__(self, n_mel, postnet_dim, postnet_kernel_size, postnet_n_convs):
        super().__init__()
        convs = [nn.Sequential(
            ConvNorm(n_mel, postnet_dim, postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2, w_init='tanh'),
            nn.BatchNorm1d(postnet_dim),
            nn.Tanh(),
            nn.Dropout(P_DROPOUT)
        )]
        for _ in range(postnet_n_convs - 2):
            convs.append(nn.Sequential(
                ConvNorm(postnet_dim, postnet_dim, postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2, w_init='tanh'),
                nn.BatchNorm1d(postnet_dim),
                nn.Tanh(),
                nn.Dropout(P_DROPOUT)
            ))
        convs.append(nn.Sequential(
            ConvNorm(postnet_dim, n_mel, postnet_kernel_size, padding=(postnet_kernel_size - 1) // 2),
            nn.Dropout(P_DROPOUT)
        ))
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

class Tacotron2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(N_SYMBOLS, SYMBOLS_EMBEDDING_DIM, ENCODER_N_CONVS, ENCODER_DIM, ENCODER_KERNEL_SIZE)
        self.decoder = Decoder(N_MEL_CHANNELS, ENCODER_DIM, ATTENTION_RNN_DIM, DECODER_RNN_DIM, PRENET_DIM, 
                               ATTENTION_DIM, ATTENTION_LOCATION_N_FILTERS, ATTENTION_LOCATION_KERNEL_SIZE, P_DROPOUT)
        self.postnet = Postnet(N_MEL_CHANNELS, POSTNET_DIM, POSTNET_KERNEL_SIZE, POSTNET_N_CONVS)

    def forward(self, text, text_lengths, mel_targets, mel_lengths):
        encoder_outputs = self.encoder(text, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mel_targets, text_lengths)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mel_out, mel_out_postnet, gate_out, mel_target, gate_target):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

class SyntheticTTSDataset(Dataset):
    def __init__(self, size, max_text_len=100, max_mel_len=200):
        self.size = size
        self.max_text_len = max_text_len
        self.max_mel_len = max_mel_len
    def __len__(self): return self.size
    def __getitem__(self, i):
        text_len = torch.randint(20, self.max_text_len, (1,)).item()
        mel_len = torch.randint(50, self.max_mel_len, (1,)).item()
        text = torch.randint(0, N_SYMBOLS, (text_len,))
        mel = torch.randn(N_MEL_CHANNELS, mel_len)
        gate = torch.zeros(mel_len)
        gate[-1] = 1.0
        return text, mel, gate

def collate_tts(batch):
    texts, mels, gates = zip(*batch)
    text_lens = torch.tensor([t.size(0) for t in texts])
    mel_lens = torch.tensor([m.size(1) for m in mels])
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    max_mel = max(m.size(1) for m in mels)
    mels_padded = torch.zeros(len(mels), N_MEL_CHANNELS, max_mel)
    gates_padded = torch.zeros(len(mels), max_mel)
    for i, (m, g) in enumerate(zip(mels, gates)):
        mels_padded[i, :, :m.size(1)] = m
        gates_padded[i, :g.size(0)] = g
    return texts_padded, text_lens, mels_padded, mel_lens, gates_padded

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    model = Tacotron2().to(dev)
    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"speech_tacotron2","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"tacotron2_32_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    
    model = DDP(model, device_ids=[rank])
    ds = SyntheticTTSDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_tts)
    
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
    
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train(); sampler.set_epoch(ep); es = time.time()
        for text, text_len, mel, mel_len, gate in loader:
            text, text_len = text.to(dev), text_len.to(dev)
            mel, mel_len, gate = mel.to(dev), mel_len.to(dev), gate.to(dev)
            optimizer.zero_grad()
            mel_out, mel_out_post, gate_out, _ = model(text, text_len, mel, text_len)
            loss = criterion(mel_out, mel_out_post, gate_out, mel, gate)
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
