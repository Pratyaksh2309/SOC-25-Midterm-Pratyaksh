import numpy as np
import gym
import imageio
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="rgb_array")

n_states = env.observation_space.n
n_actions = env.action_space.n

def save_video(frames, filename="frozenlake_qlearning.mp4"):
    os.makedirs("videos", exist_ok=True)
    path = f"videos/{filename}"
    with imageio.get_writer(path, fps=3, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)


# -------------------
# Monte Carlo (First Visit)
# -------------------
def monte_carlo_first_visit(env, episodes=5000, gamma=0.95,
                            epsilon_start=1.0, epsilon_min=0.05, decay_rate=0.999):
    Q = np.zeros((n_states, n_actions))
    returns = [[] for _ in range(n_states * n_actions)]
    rewards = []

    epsilon = epsilon_start

    def get_action(state):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        return np.argmax(Q[state])

    for ep in range(episodes):
        state = env.reset()[0]
        episode = []
        total_reward = 0
        done = False

        while not done:
            action = get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            G = gamma * G + reward_t
            sa_pair = (state_t, action_t)
            if sa_pair not in visited:
                visited.add(sa_pair)
                idx = state_t * n_actions + action_t
                returns[idx].append(G)
                Q[state_t, action_t] = np.mean(returns[idx])

        epsilon = max(epsilon * decay_rate, epsilon_min)
        rewards.append(total_reward)

        if total_reward > 0:
            print(f"✔️ Reached goal in episode {ep}")

    return Q, rewards

# -------------------
# Evaluation & Video
# -------------------
def evaluate_policy(Q, name="qlearning"):
    state = env.reset()[0]
    frames = [env.render()]
    done = False
    total_reward = 0

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, _ = env.step(action)
        frames.append(env.render())
        total_reward += reward

    save_video(frames, f"{name}.mp4")
    return total_reward


# Record Videos BEFORE Training
# evaluate_policy(np.zeros((n_states, n_actions)), "monte_carlo_untrained")


# Train and Plot

mc_Q, mc_rewards = monte_carlo_first_visit(env)

def smooth(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.plot(smooth(mc_rewards), label="Monte Carlo (Smoothed)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("FrozenLake-v1: Reward per Episode")
plt.legend()
plt.grid()
plt.show()


# Record Videos
evaluate_policy(mc_Q, "monte_carlo")
