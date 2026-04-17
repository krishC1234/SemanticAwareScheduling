#!/usr/bin/env python3
"""Learning to Paint (DDPG) - batch=48, small params (~2M)"""
import time,json,torch,torch.nn as nn,torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === VARYING ===
BATCH_SIZE = 48
HIDDEN = 256

# === FIXED ===
EPOCHS = 3
NUM_SAMPLES = 2000
IMG_SIZE = 128
STATE_DIM = 9  # canvas(3) + target(3) + step(1) + coord(2)
ACTION_DIM = 13  # stroke parameters (bezier curve)
TAU = 0.001
GAMMA = 0.95 ** 5
RMSIZE = 800
MAX_STEP = 40

class Actor(nn.Module):
    """Policy network: state -> action"""
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim, hidden // 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 4, hidden // 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 16, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state):
        x = self.conv(state)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Critic(nn.Module):
    """Q-network: (state, action) -> Q-value"""
    def __init__(self, state_dim, action_dim, hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim, hidden // 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 4, hidden // 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 16 + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = self.conv(state)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros((capacity, *state_shape), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, *state_shape), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

    def push(self, state, action, reward, next_state, done):
        bs = state.size(0)
        for i in range(bs):
            idx = (self.ptr + i) % self.capacity
            self.states[idx] = state[i]
            self.actions[idx] = action[i]
            self.rewards[idx] = reward[i]
            self.next_states[idx] = next_state[i]
            self.dones[idx] = done[i]
        self.ptr = (self.ptr + bs) % self.capacity
        self.size = min(self.size + bs, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx])

class SyntheticEnv:
    """Simplified painting environment"""
    def __init__(self, batch_size, img_size, device):
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.step_count = 0
        self.max_step = MAX_STEP
        # Precompute coord channels
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(-1, 1, img_size, device=device),
            torch.linspace(-1, 1, img_size, device=device),
            indexing='ij'
        )
        self.coord = torch.stack([coord_x, coord_y], dim=0)

    def reset(self):
        self.step_count = 0
        self.canvas = torch.zeros(self.batch_size, 3, self.img_size, self.img_size, device=self.device)
        self.target = torch.rand(self.batch_size, 3, self.img_size, self.img_size, device=self.device)
        return self._get_state()

    def _get_state(self):
        step_ch = torch.full((self.batch_size, 1, self.img_size, self.img_size),
                             self.step_count / self.max_step, device=self.device)
        coord_ch = self.coord.unsqueeze(0).expand(self.batch_size, -1, -1, -1)
        return torch.cat([self.canvas, self.target, step_ch, coord_ch], dim=1)

    def step(self, action):
        # Simplified: action affects canvas randomly (real impl would render strokes)
        delta = torch.rand_like(self.canvas) * 0.1
        self.canvas = torch.clamp(self.canvas + delta, 0, 1)
        self.step_count += 1
        reward = -torch.mean((self.canvas - self.target) ** 2, dim=[1, 2, 3], keepdim=True)
        done = torch.full((self.batch_size, 1), self.step_count >= self.max_step,
                          dtype=torch.float32, device=self.device)
        return self._get_state(), reward, done

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    # Create actor/critic and targets
    actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN).to(dev)
    critic = Critic(STATE_DIM, ACTION_DIM, HIDDEN).to(dev)
    actor_target = Actor(STATE_DIM, ACTION_DIM, HIDDEN).to(dev)
    critic_target = Critic(STATE_DIM, ACTION_DIM, HIDDEN).to(dev)
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    pc = count_params(actor) + count_params(critic)
    if rank == 0:
        print("###FEATURES###"); print(json.dumps({"model_type":"rl_ddpg","batch_size":BATCH_SIZE,"param_count":pc})); print("###END_FEATURES###")
        print(f"ltp_48_small | GPUs:{ws} | Batch:{BATCH_SIZE} | Params:{pc:,}")

    actor = DDP(actor, device_ids=[rank])
    critic = DDP(critic, device_ids=[rank])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    env = SyntheticEnv(BATCH_SIZE // ws, IMG_SIZE, dev)
    buffer = ReplayBuffer(RMSIZE, (STATE_DIM, IMG_SIZE, IMG_SIZE), ACTION_DIM, dev)

    # Fill buffer with initial experiences
    state = env.reset()
    for _ in range(RMSIZE // (BATCH_SIZE // ws) + 1):
        with torch.no_grad():
            action = actor(state)
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state if not done.all() else env.reset()

    ts, tsp = time.time(), 0
    updates_per_epoch = NUM_SAMPLES // BATCH_SIZE
    for ep in range(EPOCHS):
        es = time.time()
        for _ in range(updates_per_epoch):
            # Sample from buffer
            s, a, r, ns, d = buffer.sample(BATCH_SIZE // ws)

            # Critic update
            with torch.no_grad():
                na = actor_target.module(ns) if hasattr(actor_target, 'module') else actor_target(ns)
                target_q = r + GAMMA * (1 - d) * critic_target(ns, na)
            current_q = critic(s, a)
            critic_loss = mse(current_q, target_q)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # Actor update
            actor_loss = -critic(s, actor(s)).mean()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # Soft update targets
            soft_update(actor_target, actor.module, TAU)
            soft_update(critic_target, critic.module, TAU)

            # Collect more experience
            with torch.no_grad():
                action = actor(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state if not done.all() else env.reset()

        tsp += NUM_SAMPLES
        if rank == 0: print(f"Epoch {ep+1}/{EPOCHS} | time:{time.time()-es:.2f}s | throughput:{NUM_SAMPLES/(time.time()-es):.1f} samples/sec")

    tt = time.time() - ts
    if rank == 0:
        print(f"\n==> Summary: GPUs:{ws} | Total time:{tt:.2f}s | Avg throughput:{tsp/tt:.1f} samples/sec")
        print("###RESULTS###"); print(json.dumps({"batch_size":BATCH_SIZE,"param_count":pc,"gpu_count":ws,"total_time_sec":round(tt,2),"avg_throughput":round(tsp/tt,1)})); print("###END_RESULTS###")
    dist.destroy_process_group()

if __name__ == "__main__": main()
