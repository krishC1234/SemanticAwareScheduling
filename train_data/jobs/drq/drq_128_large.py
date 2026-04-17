#!/usr/bin/env python3
"""DRQ (Data-Regularized Q-learning) - batch=128, large params (~8M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 128
ENCODER_FEATURES = 64
HIDDEN_DIM = 512
EPOCHS = 3
NUM_SAMPLES = 5000
OBS_SHAPE = (9, 84, 84)
ACTION_DIM = 4
ENCODER_OUT_DIM = 100
DISCOUNT = 0.99
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
ALPHA_LR = 1e-4
TAU = 0.01
INIT_TEMPERATURE = 0.1
IMAGE_PAD = 4

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, nn.init.calculate_gain('relu'))
        if m.bias is not None: m.bias.data.fill_(0.0)

class Encoder(nn.Module):
    def __init__(self, obs_shape, fd, nf=ENCODER_FEATURES):
        super().__init__()
        self.feature_dim = fd
        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], nf, 3, stride=2), nn.Conv2d(nf, nf, 3, stride=1), nn.Conv2d(nf, nf, 3, stride=1), nn.Conv2d(nf, nf, 3, stride=1)])
        self.head = nn.Sequential(nn.Linear(nf * 35 * 35, fd), nn.LayerNorm(fd))
        self.apply(weight_init)
    def forward(self, obs, detach=False):
        x = obs / 255.0
        for c in self.convs: x = F.relu(c(x))
        x = x.view(x.size(0), -1)
        if detach: x = x.detach()
        return self.head(x)

class Actor(nn.Module):
    def __init__(self, enc, ad, hd=HIDDEN_DIM):
        super().__init__()
        self.encoder = enc
        self.trunk = nn.Sequential(nn.Linear(enc.feature_dim, hd), nn.ReLU(True), nn.Linear(hd, hd), nn.ReLU(True), nn.Linear(hd, 2 * ad))
        self.apply(weight_init)
    def forward(self, obs, de=False):
        h = self.encoder(obs, detach=de)
        ms = self.trunk(h)
        mu, ls = ms.chunk(2, dim=-1)
        return mu, torch.clamp(ls, -20, 2).exp()
    def sample(self, obs, de=False):
        mu, std = self.forward(obs, de)
        n = torch.randn_like(mu)
        pi = torch.tanh(mu + n * std)
        lp = (-0.5 * n.pow(2) - 0.5 * math.log(2 * math.pi) - std.log()).sum(-1, keepdim=True) - torch.log(1 - pi.pow(2) + 1e-6).sum(-1, keepdim=True)
        return pi, lp, mu

class Critic(nn.Module):
    def __init__(self, enc, ad, hd=HIDDEN_DIM):
        super().__init__()
        self.encoder = enc
        self.Q1 = nn.Sequential(nn.Linear(enc.feature_dim + ad, hd), nn.ReLU(True), nn.Linear(hd, hd), nn.ReLU(True), nn.Linear(hd, 1))
        self.Q2 = nn.Sequential(nn.Linear(enc.feature_dim + ad, hd), nn.ReLU(True), nn.Linear(hd, hd), nn.ReLU(True), nn.Linear(hd, 1))
        self.apply(weight_init)
    def forward(self, obs, action, de=False):
        h = torch.cat([self.encoder(obs, detach=de), action], -1)
        return self.Q1(h), self.Q2(h)

class DRQAgent(nn.Module):
    def __init__(self, obs_shape, ad, ef, hd):
        super().__init__()
        self.encoder = Encoder(obs_shape, ENCODER_OUT_DIM, ef)
        self.encoder_target = Encoder(obs_shape, ENCODER_OUT_DIM, ef)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        self.actor = Actor(self.encoder, ad, hd)
        self.critic = Critic(self.encoder, ad, hd)
        self.critic_target = Critic(self.encoder_target, ad, hd)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = nn.Parameter(torch.tensor(math.log(INIT_TEMPERATURE)))
        self.target_entropy = -ad

def random_shift_batch(obs, pad=IMAGE_PAD):
    n, c, h, w = obs.shape
    obs_pad = F.pad(obs, (pad, pad, pad, pad), mode='replicate')
    eps = 1.0 / (h + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=obs.device)[:h]
    ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    bg = torch.cat([ar, ar.transpose(0, 1)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
    sh = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype) * 2.0 / (h + 2 * pad)
    return F.grid_sample(obs_pad, bg + sh, padding_mode='zeros', align_corners=False)

class SyntheticRLDataset(Dataset):
    def __init__(self, sz, os, ad): self.sz, self.os, self.ad = sz, os, ad
    def __len__(self): return self.sz
    def __getitem__(self, i):
        return torch.randint(0, 256, self.os, dtype=torch.uint8).float(), torch.rand(self.ad) * 2 - 1, torch.randn(1), torch.randint(0, 256, self.os, dtype=torch.uint8).float(), torch.tensor([0.0])

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank); dev = torch.device(f"cuda:{rank}")
    agent = DRQAgent(OBS_SHAPE, ACTION_DIM, ENCODER_FEATURES, HIDDEN_DIM).to(dev)
    pc = count_params(agent)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"rl_drq","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"drq_128_large | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")
    agent.encoder = DDP(agent.encoder, device_ids=[rank])
    agent.actor = DDP(agent.actor, device_ids=[rank])
    agent.critic = DDP(agent.critic, device_ids=[rank])
    ds = SyntheticRLDataset(NUM_SAMPLES, OBS_SHAPE, ACTION_DIM)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)
    actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=CRITIC_LR)
    alpha_opt = torch.optim.Adam([agent.log_alpha], lr=ALPHA_LR)
    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        sampler.set_epoch(ep); es = time.time()
        for obs, action, reward, next_obs, done in loader:
            obs, action, reward, next_obs, done = obs.to(dev), action.to(dev), reward.to(dev), next_obs.to(dev), done.to(dev)
            obs_aug, next_obs_aug = random_shift_batch(obs), random_shift_batch(next_obs)
            with torch.no_grad():
                na, nlp, _ = agent.actor.module.sample(next_obs_aug)
                tq1, tq2 = agent.critic_target(next_obs_aug, na)
                alpha = agent.log_alpha.exp()
                tq = reward + (1 - done) * DISCOUNT * (torch.min(tq1, tq2) - alpha * nlp)
            q1, q2 = agent.critic(obs_aug, action)
            cl = F.mse_loss(q1, tq) + F.mse_loss(q2, tq)
            critic_opt.zero_grad(); cl.backward(); critic_opt.step()
            pi, lp, _ = agent.actor.module.sample(obs_aug, de=True)
            q1p, q2p = agent.critic(obs_aug, pi, de=True)
            al = (alpha.detach() * lp - torch.min(q1p, q2p)).mean()
            actor_opt.zero_grad(); al.backward(); actor_opt.step()
            alph_l = -(agent.log_alpha * (lp + agent.target_entropy).detach()).mean()
            alpha_opt.zero_grad(); alph_l.backward(); alpha_opt.step()
            with torch.no_grad():
                for p, tp in zip(agent.critic.parameters(), agent.critic_target.parameters()): tp.data.lerp_(p.data, TAU)
                for p, tp in zip(agent.encoder.parameters(), agent.encoder_target.parameters()): tp.data.lerp_(p.data, TAU)
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")
    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
