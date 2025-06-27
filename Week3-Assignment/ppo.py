import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ----- Environment -----
env = gym.make('Pendulum-v1', render_mode="rgb_array")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# ----- Hyperparameters -----
lr = 3e-4
gamma = 0.99
clip_epsilon = 0.2
update_epochs = 20
episodes = 300
max_steps = 200
entropy_coeff = 0.1

# ----- Actor-Critic Networks -----
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mu = self.net(x)
        std = self.log_std.exp()
        return mu, std

    def get_action(self, obs):
        mu, std = self(obs)
        dist = torch.distributions.Normal(mu, std)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * max_action

        log_prob = dist.log_prob(raw_action).sum(axis=-1)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-7).sum(axis=-1)
        return action, log_prob, dist

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

actor = Actor()
critic = Critic()
optimizer_actor = optim.Adam(actor.parameters(), lr=lr)
optimizer_critic = optim.Adam(critic.parameters(), lr=lr)

# ----- Return Calculation -----
def compute_returns(rewards, last_value, gamma=0.99):
    returns = []
    R = last_value
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)

episode_rewards = []

# ----- PPO Training Loop -----
for ep in range(episodes):
    obs = env.reset()[0]
    obs_buffer, act_buffer, logp_buffer, rew_buffer, val_buffer = [], [], [], [], []
    total_reward = 0

    for t in range(max_steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        obs_tensor[2] /= 8.0  # Normalize angular velocity

        value = critic(obs_tensor)
        action, logp, _ = actor.get_action(obs_tensor)
        next_obs, reward, done, _, _ = env.step(action.detach().numpy())

        obs_buffer.append(obs_tensor)
        act_buffer.append(action)
        logp_buffer.append(logp)
        rew_buffer.append(reward / 10.0)  # Scale reward
        val_buffer.append(value.squeeze(0))

        obs = next_obs
        total_reward += reward
        if done:
            break

    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        obs_tensor[2] /= 8.0
        last_value = critic(obs_tensor).item()

    returns = compute_returns(rew_buffer, last_value)
    values = torch.stack(val_buffer)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.detach()
    logp_old = torch.stack(logp_buffer).detach()

    # ----- Update Actor -----
    for _ in range(update_epochs):
        obs_stack = torch.stack(obs_buffer)
        mu, std = actor(obs_stack)
        dist = torch.distributions.Normal(mu, std)
        raw_actions = dist.rsample()
        squashed_actions = torch.tanh(raw_actions)
        new_logp = dist.log_prob(raw_actions).sum(axis=-1)
        new_logp -= torch.log(1 - squashed_actions.pow(2) + 1e-7).sum(axis=-1)

        ratio = (new_logp - logp_old).exp()
        clip_adv = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        entropy = dist.entropy().sum(axis=-1).mean()
        loss_actor = -(torch.min(ratio * advantages, clip_adv)).mean() - entropy_coeff * entropy

        optimizer_actor.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        optimizer_actor.step()

    # ----- Update Critic -----
    value_preds = critic(obs_stack).squeeze()
    loss_critic = ((returns - value_preds) ** 2).mean()
    optimizer_critic.zero_grad()
    loss_critic.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    optimizer_critic.step()

    episode_rewards.append(total_reward)
    print(f"Episode {ep} | Total Reward: {total_reward:.2f}")

# ----- Plotting -----
def smooth(x, window=10):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.plot(smooth(episode_rewards), label="PPO Smoothed")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("PPO on Pendulum-v1")
plt.legend()
plt.grid()
plt.show()

# Save trained models
torch.save(actor.state_dict(), "ppo_actor.pth")
torch.save(critic.state_dict(), "ppo_critic.pth")
