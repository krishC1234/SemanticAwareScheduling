#!/usr/bin/env python3
"""DRQ (Data-Regularized Q-learning) - batch=128, small params (~2M)"""
import time,json,math,torch,torch.nn as nn,torch.nn.functional as F,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 128
ENCODER_FEATURES = 32   # Base CNN features
HIDDEN_DIM = 256        # MLP hidden dim

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 5000
OBS_SHAPE = (9, 84, 84)  # 3 stacked frames × 3 channels (or 9 grayscale)
ACTION_DIM = 4           # Action dimension
ENCODER_OUT_DIM = 50     # Encoder output features
DISCOUNT = 0.99
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
ALPHA_LR = 1e-4
TAU = 0.01              # Soft update coefficient
INIT_TEMPERATURE = 0.1
IMAGE_PAD = 4           # Padding for random shift augmentation

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Encoder(nn.Module):
    """CNN Encoder for pixel observations"""
    def __init__(self, obs_shape, feature_dim, num_filters=ENCODER_FEATURES):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = num_filters
        self.output_logits = True
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList([
            nn.Conv2d(obs_shape[0], num_filters, 3, stride=2),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
            nn.Conv2d(num_filters, num_filters, 3, stride=1),
        ])

        self.head = nn.Sequential(
            nn.Linear(num_filters * 35 * 35, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        self.apply(weight_init)

    def forward_conv(self, obs):
        obs = obs / 255.0
        conv = obs
        for layer in self.convs:
            conv = F.relu(layer(conv))
        return conv.view(conv.size(0), -1)

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)
        if detach:
            h = h.detach()
        return self.head(h)

class Actor(nn.Module):
    """MLP Actor for continuous control"""
    def __init__(self, encoder, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = encoder
        
        self.trunk = nn.Sequential(
            nn.Linear(encoder.feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_dim)  # Mean and log_std
        )
        self.apply(weight_init)

    def forward(self, obs, detach_encoder=False):
        h = self.encoder(obs, detach=detach_encoder)
        mu_log_std = self.trunk(h)
        mu, log_std = mu_log_std.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, obs, detach_encoder=False):
        mu, std = self.forward(obs, detach_encoder)
        # Reparameterization trick
        noise = torch.randn_like(mu)
        pi = mu + noise * std
        log_pi = (-0.5 * noise.pow(2) - 0.5 * math.log(2 * math.pi) - log_std).sum(-1, keepdim=True)
        # Squash to [-1, 1]
        pi = torch.tanh(pi)
        log_pi = log_pi - torch.log(1 - pi.pow(2) + 1e-6).sum(-1, keepdim=True)
        return pi, log_pi, mu

class Critic(nn.Module):
    """Twin Q-networks"""
    def __init__(self, encoder, action_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = encoder

        self.Q1 = nn.Sequential(
            nn.Linear(encoder.feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.Q2 = nn.Sequential(
            nn.Linear(encoder.feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        h = self.encoder(obs, detach=detach_encoder)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2

class DRQAgent(nn.Module):
    """DRQ Agent combining encoder, actor, and critic"""
    def __init__(self, obs_shape, action_dim, encoder_features, hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        
        # Shared encoder for actor
        self.encoder = Encoder(obs_shape, ENCODER_OUT_DIM, encoder_features)
        # Separate encoder for critic (can share weights but separate for flexibility)
        self.encoder_target = Encoder(obs_shape, ENCODER_OUT_DIM, encoder_features)
        self.encoder_target.load_state_dict(self.encoder.state_dict())
        
        self.actor = Actor(self.encoder, action_dim, hidden_dim)
        self.critic = Critic(self.encoder, action_dim, hidden_dim)
        self.critic_target = Critic(self.encoder_target, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy temperature
        self.log_alpha = nn.Parameter(torch.tensor(math.log(INIT_TEMPERATURE)))
        self.target_entropy = -action_dim

    def forward(self, obs):
        return self.actor.sample(obs)

def random_shift(imgs, pad=IMAGE_PAD):
    """Random shift data augmentation"""
    n, c, h, w = imgs.shape
    imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
    return torch.stack([
        imgs[i, :, 
             torch.randint(0, 2*pad+1, (1,)).item():torch.randint(0, 2*pad+1, (1,)).item()+h,
             torch.randint(0, 2*pad+1, (1,)).item():torch.randint(0, 2*pad+1, (1,)).item()+w]
        for i in range(n)
    ])

def random_shift_batch(obs, pad=IMAGE_PAD):
    """Vectorized random shift"""
    n, c, h, w = obs.shape
    obs_pad = F.pad(obs, (pad, pad, pad, pad), mode='replicate')
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=obs.device)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(0, 1)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype)
    shift = shift * 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(obs_pad, grid, padding_mode='zeros', align_corners=False)

class SyntheticRLDataset(Dataset):
    """Generate synthetic RL transitions"""
    def __init__(self, size, obs_shape, action_dim):
        self.size = size
        self.obs_shape = obs_shape
        self.action_dim = action_dim

    def __len__(self): return self.size

    def __getitem__(self, i):
        obs = torch.randint(0, 256, self.obs_shape, dtype=torch.uint8).float()
        next_obs = torch.randint(0, 256, self.obs_shape, dtype=torch.uint8).float()
        action = torch.rand(self.action_dim) * 2 - 1  # [-1, 1]
        reward = torch.randn(1)
        done = torch.tensor([0.0])  # Not done
        return obs, action, reward, next_obs, done

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    agent = DRQAgent(OBS_SHAPE, ACTION_DIM, ENCODER_FEATURES, HIDDEN_DIM).to(dev)
    pc = count_params(agent)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"rl_drq","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"drq_128_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    # Wrap components for DDP
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
            obs, action, reward = obs.to(dev), action.to(dev), reward.to(dev)
            next_obs, done = next_obs.to(dev), done.to(dev)
            
            # Data augmentation
            obs_aug = random_shift_batch(obs)
            next_obs_aug = random_shift_batch(next_obs)
            
            # Critic update
            with torch.no_grad():
                next_action, next_log_pi, _ = agent.actor.module.sample(next_obs_aug)
                target_q1, target_q2 = agent.critic_target(next_obs_aug, next_action)
                target_q = torch.min(target_q1, target_q2)
                alpha = agent.log_alpha.exp()
                target_q = reward + (1 - done) * DISCOUNT * (target_q - alpha * next_log_pi)
            
            q1, q2 = agent.critic(obs_aug, action)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            # Actor update
            pi, log_pi, _ = agent.actor.module.sample(obs_aug, detach_encoder=True)
            q1_pi, q2_pi = agent.critic(obs_aug, pi, detach_encoder=True)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha.detach() * log_pi - q_pi).mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            
            # Alpha update
            alpha_loss = -(agent.log_alpha * (log_pi + agent.target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()
            
            # Soft update target networks
            with torch.no_grad():
                for p, tp in zip(agent.critic.parameters(), agent.critic_target.parameters()):
                    tp.data.lerp_(p.data, TAU)
                for p, tp in zip(agent.encoder.parameters(), agent.encoder_target.parameters()):
                    tp.data.lerp_(p.data, TAU)
        
        tsp += len(ds)
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{len(ds)/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
