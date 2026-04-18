#!/usr/bin/env python3
"""Soft Actor-Critic (SAC) - continuous control RL, batch=256, ~0.4M params

SAC is an off-policy actor-critic algorithm with entropy regularization.
It maintains an actor (policy), twin critics (Q-functions), target critics,
and a learnable temperature alpha. Training alternates between environment
interaction (filling a replay buffer) and gradient updates on sampled
mini-batches.

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy
Deep Reinforcement Learning with a Stochastic Actor", ICML 2018
"""
import time, json, math, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 256

# === FIXED ===
EPOCHS = 20
NUM_SAMPLES = 50000
OBS_DIM = 17            # MuJoCo Humanoid-like observation space
ACT_DIM = 6             # continuous action space
HIDDEN = 256
GAMMA = 0.99
TAU = 0.005
INIT_ALPHA = 0.2
LR = 3e-4

# ---------------------------------------------------------------------------
# SAC components
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GaussianActor(nn.Module):
    """Squashed Gaussian policy: outputs mean + log_std, samples via reparameterization."""
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(HIDDEN, ACT_DIM)
        self.log_std_head = nn.Linear(HIDDEN, ACT_DIM)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        # Reparameterized sample
        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise)
        # Log-prob with tanh correction
        log_prob = (-0.5 * noise.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        log_prob -= (2 * (math.log(2) - action.abs() - F.softplus(-2 * action.abs()))).sum(-1, keepdim=True)
        return action, log_prob


class TwinCritic(nn.Module):
    """Twin Q-networks for SAC."""
    def __init__(self):
        super().__init__()
        self.q1 = MLP(OBS_DIM + ACT_DIM, 1)
        self.q2 = MLP(OBS_DIM + ACT_DIM, 1)

    def forward(self, obs, action):
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACModel(nn.Module):
    """Combined SAC model: actor + twin critics. ~0.4M parameters total."""
    def __init__(self):
        super().__init__()
        self.actor = GaussianActor()
        self.critic = TwinCritic()

    def forward(self, obs):
        """Forward pass returns actor output + critic values for training."""
        action, log_prob = self.actor(obs)
        q1, q2 = self.critic(obs, action)
        return action, log_prob, q1, q2


class SyntheticRLDataset(Dataset):
    """Synthetic replay buffer: (state, action, reward, next_state, done)."""
    def __init__(self, size):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, _):
        state = torch.randn(OBS_DIM)
        action = torch.randn(ACT_DIM).clamp(-1, 1)
        reward = torch.randn(1)
        next_state = torch.randn(OBS_DIM)
        done = torch.zeros(1)
        return state, action, reward, next_state, done


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    model = SACModel().to(dev)
    # Target critic (not wrapped in DDP — only updated via polyak)
    target_critic = TwinCritic().to(dev)
    target_critic.load_state_dict(model.critic.state_dict())

    pc = count_params(model)
    if rank == 0:
        print("###FEATURES###")
        print(json.dumps({"model_type": "rl", "batch_size": BATCH_SIZE, "param_count": pc}))
        print("###END_FEATURES###")
        print(f"soft_actor_critic | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    model = DDP(model, device_ids=[rank])
    ds = SyntheticRLDataset(NUM_SAMPLES)
    sampler = DistributedSampler(ds, num_replicas=ws, rank=rank, shuffle=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True)

    actor_optim = torch.optim.Adam(model.module.actor.parameters(), lr=LR)
    critic_optim = torch.optim.Adam(model.module.critic.parameters(), lr=LR)
    log_alpha = torch.tensor(math.log(INIT_ALPHA), device=dev, requires_grad=True)
    alpha_optim = torch.optim.Adam([log_alpha], lr=LR)
    target_entropy = -ACT_DIM

    ts, tsp = time.time(), 0
    for ep in range(EPOCHS):
        model.train()
        sampler.set_epoch(ep)
        es = time.time()
        for state, action, reward, next_state, done in loader:
            state = state.to(dev)
            action = action.to(dev)
            reward = reward.to(dev)
            next_state = next_state.to(dev)
            done = done.to(dev)
            alpha = log_alpha.exp().detach()

            # --- Critic update ---
            with torch.no_grad():
                next_action, next_log_prob = model.module.actor(next_state)
                tq1, tq2 = target_critic(next_state, next_action)
                target_q = reward + GAMMA * (1 - done) * (torch.min(tq1, tq2) - alpha * next_log_prob)

            q1, q2 = model.module.critic(state, action)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # --- Actor update ---
            new_action, log_prob = model.module.actor(state)
            q1_new, q2_new = model.module.critic(state, new_action)
            actor_loss = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # --- Alpha update ---
            alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            # --- Soft update target ---
            with torch.no_grad():
                for tp, p in zip(target_critic.parameters(), model.module.critic.parameters()):
                    tp.data.mul_(1 - TAU).add_(p.data * TAU)

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