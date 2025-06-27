import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio
import random
from collections import deque

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# ----- Environment -----
env = gym.make('Pendulum-v1', render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# ----- Hyperparameters -----
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99
tau = 0.005
buffer_size = int(1e6)
batch_size = 64
episodes = 500
max_steps = 200

# ----- OU Noise for Exploration -----
class OUNoise:
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(act_dim)

    def reset(self):
        self.state = np.zeros(act_dim)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(act_dim)
        self.state += dx
        return self.state

# ----- Replay Buffer -----
class ReplayBuffer:
    def __init__(self, size=buffer_size):
        self.buffer = deque(maxlen=size)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# ----- Actor-Critic Models -----
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * max_action

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))

# ----- Initialization -----
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

actor = Actor()
critic = Critic()
target_actor = Actor()
target_critic = Critic()

actor.apply(init_weights)
critic.apply(init_weights)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

replay_buffer = ReplayBuffer()
noise = OUNoise()
episode_rewards = []

# ----- Training Loop -----
for ep in range(episodes):
    obs = env.reset()[0]
    total_reward = 0
    noise.reset()

    for _ in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = actor(obs_tensor).numpy()
        action += noise.sample()
        action = np.clip(action, -max_action, max_action)

        next_obs, reward, done, _, _ = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            obs_b, act_b, rew_b, next_obs_b, done_b = replay_buffer.sample(batch_size)
            obs_b = torch.tensor(obs_b, dtype=torch.float32)
            act_b = torch.tensor(act_b, dtype=torch.float32)
            rew_b = torch.tensor(rew_b, dtype=torch.float32).unsqueeze(1)
            next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32)
            done_b = torch.tensor(done_b, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                target_actions = target_actor(next_obs_b)
                target_q = target_critic(next_obs_b, target_actions)
                target_value = rew_b + gamma * (1 - done_b) * target_q
                target_value = torch.clamp(target_value, -100.0, 0.0)  # prevent Q explosion

            current_q = critic(obs_b, act_b)
            loss_critic = nn.MSELoss()(current_q, target_value)

            critic_optimizer.zero_grad()
            loss_critic.backward()
            critic_optimizer.step()

            actor_loss = -critic(obs_b, actor(obs_b)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update of target networks
            for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    episode_rewards.append(total_reward)
    if ep % 10 == 0:
        print(f"Episode {ep} | Reward: {total_reward:.2f} | Sample Action: {action}")

# ----- Plotting -----
def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.plot(smooth(episode_rewards), label="DDPG Smoothed")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DDPG on Pendulum-v1")
plt.grid()
plt.legend()
plt.show()

torch.save(actor.state_dict(), "ddpg_actor.pth")


obs = env.reset()[0]
frames = []
for _ in range(200):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = actor(obs_tensor).numpy()
    obs, _, _, _, _ = env.step(action)
    frame = env.render()
    frames.append(frame)

imageio.mimsave("ddpg_pendulum_test.mp4", frames, fps=30)
print("✅ Saved video: ddpg_pendulum_test.mp4")
